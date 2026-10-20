"""TR14 source lint: a string literal containing a real line break, in Python, C/C++/HIP and libr3 .rw blocks."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry import check, source_lint as sl  # noqa: E402

NL = chr(10)
BS = chr(92)


def cpp(*lines):
    return NL.join(lines) + NL


class CppLexer(unittest.TestCase):
    def test_a_string_with_a_real_line_break_is_flagged_on_its_first_line(self):
        src = cpp('void f() {', '  std::printf("r3_moe: stages' + NL + '", 1);', '}')
        self.assertEqual(sl.cpp_string_newlines(src)[0], 2)      # the continuation line opens a second broken string

    def test_the_escape_is_fine(self):
        src = cpp('void f() { std::printf("r3_moe: stages' + BS + 'n", 1); }')
        self.assertEqual(sl.cpp_string_newlines(src), [])

    def test_backslash_continuation_is_fine(self):
        src = cpp('#define MSG "first part ' + BS, 'second part"', 'int x;')
        self.assertEqual(sl.cpp_string_newlines(src), [])

    def test_raw_strings_may_span_lines(self):
        src = cpp('const char* t = R"x(line one', 'line two "quoted"', ')x";', 'int y;')
        self.assertEqual(sl.cpp_string_newlines(src), [])

    def test_quotes_and_apostrophes_in_comments_do_not_count(self):
        src = cpp('// it' + "'" + 's a "comment', '/* a "block', 'that spans */ int z;', 'int ok = 1;')
        self.assertEqual(sl.cpp_string_newlines(src), [])

    def test_char_literals_and_digit_separators(self):
        src = cpp("char q = '" + '"' + "';", "int big = 1'000'000;", 'const char* s = "ok";')
        self.assertEqual(sl.cpp_string_newlines(src), [])

    def test_an_escaped_quote_inside_a_string(self):
        src = cpp('const char* s = "say ' + BS + '"hi' + BS + '" now";')
        self.assertEqual(sl.cpp_string_newlines(src), [])

    def test_the_line_after_a_flagged_string_is_still_counted(self):
        src = cpp('int a;', 'const char* s = "open', 'int b;', 'const char* t = "also open', 'int c;')
        self.assertEqual(sl.cpp_string_newlines(src), [2, 4])


class RewriteBlocks(unittest.TestCase):
    def test_prose_outside_blocks_is_ignored_and_blocks_are_checked(self):
        src = cpp("# Why: it's the \"fast path", '', '@@ find 1', '  std::printf("a' + NL + 'b");', '@@ replace',
                  '  int ok = 1;', '@@ end', '', "# trailing prose with a lone \" quote")
        self.assertEqual(sl.rw_string_newlines(src), [4])

    def test_a_clean_block_passes(self):
        src = cpp('@@ find 1', '  int a = 0;', '@@ replace', '  int a = 1;  // "fine"', '@@ end')
        self.assertEqual(sl.rw_string_newlines(src), [])


class PythonFiles(unittest.TestCase):
    def test_an_unterminated_string_is_a_string_finding_and_other_errors_are_syntax(self):
        self.assertEqual(sl.python_problem('x = 1' + NL + 'y = "a' + NL + 'b"' + NL)[0], "string")
        self.assertEqual(sl.python_problem("def f(:" + NL)[0], "syntax")
        self.assertIsNone(sl.python_problem('y = "a' + BS + 'nb"' + NL))


class WholeCheck(unittest.TestCase):
    def test_check_reports_a_bad_file_and_skips_vendor_and_fixtures(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "engines" / "radiance" / "components" / "moe-remote").mkdir(parents=True)
            (root / "engines" / "radiance" / "vendor" / "x").mkdir(parents=True)
            (root / "tools").mkdir()
            bad = root / "engines" / "radiance" / "components" / "moe-remote" / "bad.cpp"
            bad.write_text(cpp('int f() { std::puts("oops', '"); }'))
            (root / "engines" / "radiance" / "vendor" / "x" / "skip.cpp").write_text(cpp('const char* s = "open'))
            found = [(c, p) for c, _, p in ((k, l, str(f.relative_to(root)).replace(chr(92), "/"))
                                           for f in sl.source_files(root) for k, l, _m in sl.lint_file(f))]
            self.assertTrue(found)
            self.assertEqual({p for _, p in found}, {"engines/radiance/components/moe-remote/bad.cpp"})

    def test_the_repository_itself_is_clean(self):
        root = Path(__file__).resolve().parents[3]
        problems = [(str(f.relative_to(root)), line, kind) for f in sl.source_files(root)
                    for kind, line, _m in sl.lint_file(f)]
        self.assertEqual(problems, [])

    def test_tooling_hygiene_carries_the_rule(self):
        self.assertTrue(hasattr(check, "tooling_hygiene"))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "tools").mkdir()
            d = root / "engines" / "radiance" / "components" / "c"
            d.mkdir(parents=True)
            (d / "bad.hip").write_text(cpp('void k() { printf("x', '"); }'))
            codes = {f.code for f in check.tooling_hygiene(root)}
            self.assertIn("TR14.STRING_LITERAL_NEWLINE", codes)


if __name__ == "__main__":
    unittest.main()
