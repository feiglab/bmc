#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def parse_config_path(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config"),
        help="Config file (key/value) to read/write",
    )
    ns, _ = parser.parse_known_args(argv)
    return Path(ns.config)


def parse_floats(spec: str, defaults: Sequence[float], n_out: int) -> list[float]:
    values = [float(item) for item in spec.split(":") if item.strip()]
    out: list[float] = []
    last: float | None = None

    for idx in range(n_out):
        if idx < len(values):
            last = values[idx]
            out.append(last)
            continue
        if idx < len(defaults):
            last = float(defaults[idx])
            out.append(last)
            continue
        out.append(last if last is not None else 0.0)

    return out


def find_input_file(start_dir: Path | str, filename: str) -> Path:
    root = Path(start_dir).expanduser().resolve()

    for directory in (root, *root.parents):
        candidate = directory / filename
        if candidate.is_file():
            return candidate.resolve()

    candidate = Path.cwd() / filename
    if candidate.is_file():
        return candidate.resolve()

    raise FileNotFoundError(f"Could not find '{filename}' in {root} or its parents or CWD")


def split_tile_selection(selection: str) -> list[str]:
    base, dot, suffix = selection.partition(".")
    tiles = base.split(":")
    if dot:
        return [f"{tile}.{suffix}" for tile in tiles]
    return tiles


def build_anchor_selections(
    refsel: str,
    othersel: str,
    anchor: str,
) -> tuple[str, ...]:
    ref_anchor, other_anchor = anchor.split(":")

    ref_base, ref_dot, ref_suffix = refsel.partition(".")
    ref_tiles = ref_base.split(":")
    ref_tail = f".{ref_suffix}" if ref_dot else ""

    ref_index = ref_tiles.index(ref_anchor)
    ref_size = len(ref_tiles)

    asel1 = f"{ref_tiles[ref_index]}:{ref_tiles[(ref_index + 1) % ref_size]}{ref_tail}"
    asel2 = (
        f"{ref_tiles[(ref_index + 2) % ref_size]}:"
        f"{ref_tiles[(ref_index + 3) % ref_size]}{ref_tail}"
    )
    aselt = (
        f"{ref_tiles[ref_index]}:{ref_tiles[(ref_index + 2) % ref_size]}:"
        f"{ref_tiles[(ref_index + 3) % ref_size]}{ref_tail}"
    )
    as11 = f"{ref_tiles[ref_index]}{ref_tail}"
    as12 = f"{ref_tiles[(ref_index + 1) % ref_size]}{ref_tail}"

    other_base, other_dot, other_suffix = othersel.partition(".")
    other_tiles = other_base.split(":")
    other_tail = f".{other_suffix}" if other_dot else ""

    other_index = other_tiles.index(other_anchor)
    other_size = len(other_tiles)

    if other_size == 6:
        bsel1 = (
            f"{other_tiles[other_index]}:"
            f"{other_tiles[(other_index + 1) % other_size]}{other_tail}"
        )
        bsel2 = (
            f"{other_tiles[(other_index + 2) % other_size]}:"
            f"{other_tiles[(other_index + 3) % other_size]}{other_tail}"
        )
        bselt = (
            f"{other_tiles[other_index]}:"
            f"{other_tiles[(other_index + 2) % other_size]}:"
            f"{other_tiles[(other_index + 3) % other_size]}{other_tail}"
        )
    elif other_size == 5:
        bsel1 = (
            f"{other_tiles[other_index]}:"
            f"{other_tiles[(other_index + 1) % other_size]}{other_tail}"
        )
        bsel2 = (
            f"{other_tiles[(other_index + 3) % other_size]}:"
            f"{other_tiles[(other_index + 4) % other_size]}{other_tail}"
        )
        bselt = (
            f"{other_tiles[other_index]}:"
            f"{other_tiles[(other_index + 1) % other_size]}:"
            f"{other_tiles[(other_index + 4) % other_size]}{other_tail}"
        )
    elif other_size == 3:
        bsel1 = f"{other_tiles[other_index]}{other_tail}"
        bsel2 = f"{other_tiles[(other_index + 1) % other_size]}{other_tail}"
        bselt = (
            f"{other_tiles[other_index]}:"
            f"{other_tiles[(other_index + 1) % other_size]}{other_tail}"
        )
    else:
        raise SystemExit("ERROR: invalid length of other selection")

    return asel1, asel2, aselt, bsel1, bsel2, bselt, as11, as12


def format_bias_tag(bias: float, biasangle: float | None = None) -> str:
    if biasangle is None:
        return f"{bias:.2f}"
    return f"{bias:.2f}_{biasangle:.0f}"


def parse_bias_target(spec: str) -> tuple[float, float | None]:
    value = spec.strip()
    if not value:
        raise SystemExit("ERROR: empty bias value")

    if _count_pair_separators(value) == 0:
        try:
            return float(value), None
        except ValueError as exc:
            raise SystemExit(f"ERROR: invalid bias value {spec!r}") from exc

    pairs = parse_bias_pairs_arg(value)
    if len(pairs) != 1:
        raise SystemExit(
            "ERROR: --bias must define exactly one pair, got expanded input " f"{spec!r}"
        )
    bias, biasangle = pairs[0]
    return bias, biasangle


def parse_bias_pairs_arg(spec: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []

    for item in _iter_bias_pair_items(spec):
        bias_spec, angle_spec = _split_bias_item(item)
        bias_values = _parse_value_spec(bias_spec, name="bias")
        angle_values = _parse_value_spec(angle_spec, name="biasangle")

        for bias in bias_values:
            for angle in angle_values:
                pairs.append((bias, angle))

    if not pairs:
        raise SystemExit("ERROR: biaspairs must define at least one pair")

    return pairs


def normalize_bias_pairs_arg(spec: str) -> str:
    return "=".join(_iter_bias_pair_items(spec))


def build_bias_pairs(
    bias: str,
    biasangle: str | None = None,
    biaspairs: str | None = None,
) -> list[tuple[float, float | None]]:
    if biaspairs is not None:
        return [(biasval, angleval) for biasval, angleval in parse_bias_pairs_arg(biaspairs)]

    bmin, bmax, bdel = parse_floats(bias, [6.0, 9.0, 0.1], n_out=3)
    bias_values = float_range(bmin, bmax, bdel)

    if biasangle is None:
        return [(biasval, None) for biasval in bias_values]

    amin, amax, adel = parse_floats(biasangle, [90.0, 180.0, 15.0], n_out=3)
    angle_values = float_range(amin, amax, adel)
    return [(biasval, angleval) for biasval in bias_values for angleval in angle_values]


def float_range(start: float, stop: float, step: float) -> list[float]:
    if step <= 0.0:
        raise SystemExit(f"ERROR: range step must be > 0, got {step!r}")

    values: list[float] = []
    index = 0
    limit = stop + 1.0e-8

    while True:
        value = start + index * step
        if value > limit:
            break
        values.append(round(value, 12))
        index += 1

    return values


def _iter_bias_pair_items(spec: str) -> list[str]:
    items: list[str] = []

    for raw_item in _split_top_level(spec, "="):
        item = raw_item.strip()
        if not item:
            continue

        try:
            _split_bias_item(item)
        except SystemExit:
            subitems = _split_top_level_whitespace(item)
            if len(subitems) <= 1:
                raise

            for subitem in subitems:
                cleaned = subitem.strip()
                if not cleaned:
                    continue
                _split_bias_item(cleaned)
                items.append(cleaned)
        else:
            items.append(item)

    return items


def _count_pair_separators(spec: str) -> int:
    return len(_pair_separator_positions(spec))


def _split_bias_item(item: str) -> tuple[str, str]:
    positions = _pair_separator_positions(item)
    if len(positions) != 1:
        raise SystemExit(
            "ERROR: each biaspairs entry must be 'bias:biasangle', "
            "'bias_biasangle', or use braces such as "
            "'5.0:{90,120}={5.4,5.6}:{90,120}'"
        )

    idx = positions[0]
    left = item[:idx].strip()
    right = item[idx + 1 :].strip()
    if not left or not right:
        raise SystemExit(f"ERROR: invalid biaspairs entry {item!r}")

    return left, right


def _pair_separator_positions(spec: str) -> list[int]:
    positions: list[int] = []
    depth = 0

    for idx, char in enumerate(spec):
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth < 0:
                raise SystemExit(f"ERROR: unbalanced braces in {spec!r}")
            continue
        if depth == 0 and char in {":", "_"}:
            positions.append(idx)

    if depth != 0:
        raise SystemExit(f"ERROR: unbalanced braces in {spec!r}")

    return positions


def _parse_value_spec(spec: str, name: str) -> list[float]:
    token = spec.strip()
    if not token:
        raise SystemExit(f"ERROR: empty {name} in biaspairs")

    if token.startswith("{") or token.endswith("}"):
        if not (token.startswith("{") and token.endswith("}")):
            raise SystemExit(f"ERROR: invalid {name} list {spec!r}")
        inner = token[1:-1].strip()
        if not inner:
            raise SystemExit(f"ERROR: empty {name} list in {spec!r}")
        items = [part.strip() for part in _split_top_level(inner, ",")]
    else:
        items = [token]

    values: list[float] = []
    for item in items:
        if not item:
            raise SystemExit(f"ERROR: empty {name} entry in {spec!r}")
        try:
            values.append(float(item))
        except ValueError as exc:
            raise SystemExit(f"ERROR: invalid {name} value {item!r}") from exc

    return values


def _split_top_level(spec: str, sep: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0

    for idx, char in enumerate(spec):
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth < 0:
                raise SystemExit(f"ERROR: unbalanced braces in {spec!r}")
            continue
        if char == sep and depth == 0:
            parts.append(spec[start:idx])
            start = idx + 1

    if depth != 0:
        raise SystemExit(f"ERROR: unbalanced braces in {spec!r}")

    parts.append(spec[start:])
    return parts


def _split_top_level_whitespace(spec: str) -> list[str]:
    parts: list[str] = []
    token: list[str] = []
    depth = 0

    for char in spec:
        if char == "{":
            depth += 1
            token.append(char)
            continue
        if char == "}":
            depth -= 1
            if depth < 0:
                raise SystemExit(f"ERROR: unbalanced braces in {spec!r}")
            token.append(char)
            continue
        if depth == 0 and char.isspace():
            if token:
                parts.append("".join(token))
                token = []
            continue
        token.append(char)

    if depth != 0:
        raise SystemExit(f"ERROR: unbalanced braces in {spec!r}")

    if token:
        parts.append("".join(token))
    return parts
