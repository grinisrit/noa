#! /usr/bin/env python3

import os.path
import pathlib
import re
import sys

if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} FILE\n\nwhere FILE is a C++ source code or header file.", file=sys.stderr)
    sys.exit(1)
if not os.path.isfile(sys.argv[1]):
    print(f"error: {sys.argv[1]} is not a valid file.", file=sys.stderr)
    sys.exit(1)

src = sys.argv[1]
basename = os.path.splitext(os.path.basename(src))[0]
dirname = f"{basename}.templates"

if not os.path.isdir(dirname):
    os.mkdir(dirname)

def get_source_code(namespaces, extern_template_instantiation):
    eti = extern_template_instantiation.strip().replace("extern ", "", 1)
    # use absolute path for the include when src is an absolute path
    # (e.g. when called by CMake, because relative include does not work with
    # its separate build dir structure)
    if src == os.path.abspath(src):
        source_code = f"#include \"{src}\"\n"
    # use relative path for the include when src is relative
    else:
        relpath = os.path.relpath(src, dirname)
        source_code = f"#include \"{relpath}\"\n"
    for ns in namespaces:
        source_code += f"namespace {ns} {{\n"
    source_code += eti + "\n"
    for ns in namespaces:
        source_code += f"}} // namespace {ns}\n"
    return source_code

def check_write(content, fname):
    write = False
    if os.path.isfile(fname):
        write = open(fname, "r").read().strip() != content.strip()
    else:
        write = True

    if write is True:
        with open(fname, "w") as out:
            out.write(content)

i = 0
namespaces = []
file_names = set()
for line in open(src).readlines():
    # heuristics for namespaces
    ns_begin = re.search(r"^\s*namespace\s+(\w+)\s*\{$", line)
    if ns_begin:
        namespaces.append(ns_begin.group(1))
    ns_end = re.search(r"^\s*\}\s*\/\/\s*namespace\s+(\w+)$", line)
    if ns_end:
        namespaces.pop(-1)

    if line.strip().startswith("extern template"):
        source_code = get_source_code(namespaces, line)
        for ext in ["cpp", "cu"]:
            fname = f"{dirname}/{basename}.t{i}.{ext}"
            check_write(source_code, fname)
            file_names.add(fname)
        i += 1

# remove extraneous files from the target directory
for path in pathlib.Path(dirname).iterdir():
    if str(path) not in file_names:
        path.unlink()
