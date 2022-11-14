import re
from packaging import version
import sys

version_rgx = re.compile(r"^\s*__version__\s*=\s*['\"]([^'\"]*)['\"]")


def match_version(line):
    mm = version_rgx.match(line)
    if mm is None:
        return None
    (version_str,) = mm.groups()
    return version_str


def mk_dev_version(v, patch_number):
    *fixed, patch = version.parse(v).release
    if int(patch_number) < 0:
        next_version = int(patch) + 1
    else:
        next_version = patch_number
    return ".".join(map(str, fixed + [next_version]))


def patch_version_lines(lines, patch_number):
    for line in lines:
        v_prev = match_version(line)
        if v_prev is not None:
            v_next = mk_dev_version(v_prev, patch_number)
            line = line.replace(v_prev, v_next)
        yield line


def patch_file(fname, patch_number):
    with open(fname, "rt", encoding="utf-8") as src:
        lines = list(patch_version_lines(src, patch_number))
    with open(fname, "wt", encoding="utf-8") as dst:
        dst.writelines(lines)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print(f"Usage: {sys.argv[0]} patch_number [FILE]...")

    input_patch, *files = args
    for f in files:
        patch_file(f, input_patch)
