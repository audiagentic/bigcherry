from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core import config


class HostValueExpansionTests(TestCase):
    def _local(self, body: str) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8")
        handle.write(body)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_process_environment_expands_and_normalizes(self):
        with mock.patch.dict(os.environ, {"BC_TEST_ROOT": "C:\\Tool\\7.1\\","BIGCHERRY_ENVIRONMENT": self._local("")}):
            self.assertEqual(config.expand_host_value("${BC_TEST_ROOT}/bin/clang.exe"), "C:/Tool/7.1/bin/clang.exe")

    def test_local_env_table_is_fallback_and_process_env_wins(self):
        local = self._local('[env]\nBC_TEST_TREE = "/from/file"\n')
        with mock.patch.dict(os.environ, {"BIGCHERRY_ENVIRONMENT": local}):
            os.environ.pop("BC_TEST_TREE", None)
            self.assertEqual(config.expand_host_value("${BC_TEST_TREE}"), "/from/file")
            with mock.patch.dict(os.environ, {"BC_TEST_TREE": "/from/env"}):
                self.assertEqual(config.expand_host_value("${BC_TEST_TREE}"), "/from/env")

    def test_unset_reference_is_kept_and_fails_closed_on_use(self):
        with mock.patch.dict(os.environ, {"BIGCHERRY_ENVIRONMENT": self._local("")}):
            os.environ.pop("BC_TEST_UNSET", None)
            value = config.expand_host_value("${BC_TEST_UNSET}/clang")
        self.assertEqual(value, "${BC_TEST_UNSET}/clang")
        with self.assertRaises(config.ConfigError):
            config.require_resolved(value, "platform.x.c-compiler")
