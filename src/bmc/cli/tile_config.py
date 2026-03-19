#!/usr/bin/env python3
from __future__ import annotations

import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_CONFIG_LINE_RE = re.compile(r"^(?P<key>[^\s=]+)(?:(?:\s*=\s*)|\s+)(?P<val>.*)$")


def read_config(path: Path) -> dict[str, str]:
    """
    Read a simple key/value config file.

    Format:
        key value
        key = value
        key=value

    Blank lines and lines starting with '#' are ignored. Keys are lowercased.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}

    cfg: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        match = _CONFIG_LINE_RE.match(line)
        if match is None:
            key = line
            val = ""
        else:
            key = match.group("key").strip()
            val = match.group("val").strip()

        if key:
            cfg[key.lower()] = val

    return cfg


def parse_bool(s: str) -> bool:
    v = s.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid bool {s!r}")


def split_values(s: str) -> list[str]:
    """
    Tokenize a config value into a list (e.g. for nargs='+').

    Uses shell-like splitting, so quoting is supported.
    """
    return shlex.split(s)


def format_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        return " ".join(format_value(x) for x in v)
    return str(v)


_ORDER = [
    "mode",
    "setup",
    "equi",
    "pdb_in",
    "refpdb",
    "capdb",
    "cadcd",
    "refsel",
    "othersel",
    "anchor",
    "rot",
    "flip",
    "bias",
    "biasdir",
    "biasangle",
    "biaspairs",
    "normcap",
    "k",
    "kinit",
    "kbias",
    "kbiasangle",
    "kdistx",
    "kdisty",
    "kdistz",
    "kcent",
    "knorm",
    "kdihed",
    "krot",
    "box",
    "conc",
    "surf",
    "orient",
    "ff",
    "device",
    "resources",
]


def write_config(path: Path, data: Mapping[str, str]) -> None:
    path = Path(path)
    keys = list(data.keys())

    ordered: list[str] = []
    seen = set()

    for key in _ORDER:
        if key in data:
            ordered.append(key)
            seen.add(key)

    for key in sorted(keys):
        if key not in seen:
            ordered.append(key)

    lines = []
    for key in ordered:
        value = data.get(key, "")
        if value == "":
            continue
        lines.append(f"{key} {value}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
