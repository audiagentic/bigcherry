#!/bin/sh
# usage: rows_proof.sh <src-tree> <out-prefix>
# Compiles libr4d/r4d_rows.cpp on the host (no device code needed) and dumps every section's bytes, the
# relocations (which function each row points at) and the symbol table, so two source trees can be compared byte for byte.
set -e
T=$1; P=$2
cd "$T"
g++-14 -std=c++17 -O0 -g0 -I. -Ilibr4d -Iabi -Icore -c libr4d/r4d_rows.cpp -o "$P.o"
objdump -s "$P.o" | tail -n +3 > "$P.sections.txt"
objdump -r "$P.o" | tail -n +3 > "$P.relocs.txt"
nm -C "$P.o" | sort > "$P.symbols.txt"
