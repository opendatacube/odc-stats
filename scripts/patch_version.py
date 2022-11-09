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


def mk_dev_version(v, build_number):
    fixed = version.parse(v).release
    next_version = (*fixed, f"dev{build_number}")
    return ".".join(map(str, next_version))


def patch_version_lines(lines, build_number):
    for line in lines:
        v_prev = match_version(line)
        if v_prev is not None:
            v_next = mk_dev_version(v_prev, build_number)
            line = line.replace(v_prev, v_next)
        yield line


def patch_file(fname, build_number):
    with open(fname, "rt", encoding="utf-8") as src:
        lines = list(patch_version_lines(src, build_number))
    with open(fname, "wt", encoding="utf-8") as dst:
        dst.writelines(lines)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print(f"Usage: {sys.argv[0]} build-number [FILE]...")

    git_sha, *files = args
    for f in files:
        patch_file(f, git_sha)
