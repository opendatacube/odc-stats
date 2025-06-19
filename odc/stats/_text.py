from pathlib import Path
from typing import Any

PathLike = str | Path


# Copied from odc.io.text


def read_int(path: PathLike, default=None, base=10) -> int | None:
    """
    Read single integer from a text file.

    Useful for things like parsing content of /sys/ or /proc.
    """
    try:
        with open(path, encoding="utf8") as f:
            return int(f.read(), base)
    except (FileNotFoundError, ValueError):
        return default


def split_and_check(
    s: str, separator: str, n: int | tuple[int, ...]
) -> tuple[str, ...]:
    """Turn string into tuple, checking that there are exactly as many parts as expected.
    :param s: String to parse
    :param separator: Separator character
    :param n: Expected number of parts, can be a single integer value or several,
              example `(2, 3)` accepts 2 or 3 parts.
    """
    if isinstance(n, int):
        n = (n,)

    parts = s.split(separator)
    if len(parts) not in n:
        raise ValueError(f'Failed to parse "{s}"')
    return tuple(parts)


def parse_slice(s: str) -> slice:
    """
    Parse slice syntax in the form start:stop[:step]
    Examples "::4", "2:5", "2::10", "3:100:5"
    """

    def parse(part: str) -> int | None:
        if part == "":
            return None
        return int(part)

    try:
        parts = [parse(p) for p in split_and_check(s, ":", (2, 3))]
    except ValueError:
        raise ValueError(f'Expect <start>:<stop>[:<step>] syntax, got "{s}"') from None

    return slice(*parts)


def parse_yaml(s: str) -> dict[str, Any]:
    # pylint: disable=import-outside-toplevel
    import yaml

    return yaml.load(s, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))


def parse_yaml_file_or_inline(s: str) -> dict[str, Any]:
    """
    Accept on input either a path to yaml file or yaml text, return parsed yaml document.
    """
    try:
        # if file
        path = Path(s)
        with open(path, encoding="utf8") as f:
            txt = f.read()
            assert isinstance(txt, str)
    except (FileNotFoundError, OSError, ValueError):
        txt = s
    result = parse_yaml(txt)
    if isinstance(result, str):
        raise OSError(f"No such file: {s}")
    return result


def load_yaml_remote(yaml_url: str) -> dict[str, Any]:
    """
    Open a yaml file remotely and return the parsed yaml document
    """
    import fsspec
    import yaml

    try:
        with fsspec.open(yaml_url, mode="r") as f:
            return next(yaml.safe_load_all(f))
    except Exception:
        print(f"Cannot load {yaml_url}")
        raise


def parse_range2d_int(s: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Parse string like "0:3,4:5" -> ((0,3), (4,5))"""
    try:
        return tuple(
            tuple(int(x) for x in split_and_check(p, ":", 2))
            for p in split_and_check(s, ",", 2)
        )
    except ValueError:
        raise ValueError(f'Expect <int>:<int>,<int>:<int> syntax, got "{s}"') from None
