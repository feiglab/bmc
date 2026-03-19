#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

from .tile_config import format_value, parse_bool, split_values, write_config

_PERSISTENT_DESTS = (
    "mode",
    "setup",
    "equi",
    "pdb",
    "refpdb",
    "capdb",
    "cadcd",
    "refsel",
    "othersel",
    "anchor",
    "rot",
    "bias",
    "biasdir",
    "biasangle",
    "flip",
    "normcap",
    "distcap",
    "k",
    "kinit",
    "kbias",
    "kbiasangle",
    "biaspairs",
    "kdistx",
    "kdisty",
    "kdistz",
    "kdist",
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
)


_K_INDIVIDUAL_DESTS = (
    "kinit",
    "kbias",
    "kbiasangle",
    "kdistx",
    "kdisty",
    "kdistz",
    "kdist",
    "kcent",
    "knorm",
    "kdihed",
    "krot",
)


_CONFIG_KEY_BY_DEST = {
    "pdb": "pdb_in",
}


def _config_key(dest: str) -> str:
    return _CONFIG_KEY_BY_DEST.get(dest, dest)


def _config_default(dest: str, value: str) -> object:
    if dest in {"setup", "equi", "refpdb"}:
        return Path(value)
    if dest in {
        "conc",
        "surf",
        "normcap",
        "distcap",
        "kinit",
        "kbias",
        "kbiasangle",
        "kdistx",
        "kdisty",
        "kdistz",
        "kdist",
        "kcent",
        "knorm",
        "kdihed",
        "krot",
    }:
        return float(value)
    if dest == "device":
        return int(value)
    if dest in {"orient", "flip"}:
        return parse_bool(value)
    if dest == "ff":
        return split_values(value)
    return value


def _normalize_config_value(dest: str, value: object) -> object:
    if value is None:
        return value
    if dest == "mode":
        return str(value).lower()
    if dest == "biaspairs":
        return normalize_bias_pairs_arg(str(value))
    return value


def _resolve_config_value(
    dest: str,
    args: argparse.Namespace,
    value: object,
) -> object:
    if value is not None:
        return value

    if dest == "bias":
        if getattr(args, "biaspairs", None) is None:
            return "6.0:9.0:0.1"
        return None

    if dest == "k":
        has_individual = any(getattr(args, name, None) is not None for name in _K_INDIVIDUAL_DESTS)
        if not has_individual:
            return "500:200"
        return None

    return value


def _normalize_visible(visible: Sequence[str]) -> set[str]:
    names: set[str] = set()
    for item in visible:
        name = item.strip()
        if not name:
            continue
        names.add(name)
        names.add(name.lstrip("-").replace("-", "_"))
    return names


def _help_text(visible: set[str], dest: str, text: str) -> str:
    return text if dest in visible else argparse.SUPPRESS


def apply_config_defaults(
    p: argparse.ArgumentParser,
    cfg: dict[str, str],
) -> None:
    defaults: dict[str, object] = {}

    for dest in _PERSISTENT_DESTS:
        key = _config_key(dest)
        if key not in cfg:
            continue
        defaults[dest] = _config_default(dest, cfg[key])

    if defaults:
        p.set_defaults(**defaults)


def _add_all_arguments(
    p: argparse.ArgumentParser,
    visible: set[str],
) -> None:
    mode_grp = p.add_mutually_exclusive_group()
    mode_grp.add_argument(
        "--mode",
        choices=["allatom", "cocomo"],
        default="allatom",
        help=_help_text(visible, "mode", "Simulation mode"),
    )
    mode_grp.add_argument(
        "--allatom",
        dest="mode",
        action="store_const",
        const="allatom",
        help=_help_text(visible, "mode", "Shortcut for --mode allatom"),
    )
    mode_grp.add_argument(
        "--cocomo",
        dest="mode",
        action="store_const",
        const="cocomo",
        help=_help_text(visible, "mode", "Shortcut for --mode cocomo"),
    )

    p.add_argument(
        "--setup",
        type=Path,
        default=Path("setup"),
        help=_help_text(visible, "setup", "Setup directory"),
    )
    p.add_argument(
        "--equi",
        type=Path,
        default=Path("equi"),
        help=_help_text(visible, "equi", "Equilibration output directory"),
    )
    p.add_argument(
        "--pdb",
        type=str,
        default=None,
        help=_help_text(
            visible,
            "pdb",
            "Input PDB file (searched relative to --setup and parents, " "then CWD)",
        ),
    )
    p.add_argument(
        "--refpdb",
        type=Path,
        default=None,
        help=_help_text(
            visible,
            "refpdb",
            "Reference PDB (used for restraints). Defaults depend on mode: "
            "allatom->dimer.solvated.pdb, cocomo->dimer.protein.pdb",
        ),
    )
    p.add_argument(
        "--capdb",
        type=str,
        default="CA.pdb",
        help=_help_text(visible, "capdb", "Reference trajectory (CA only)"),
    )
    p.add_argument(
        "--cadcd",
        type=str,
        default="ca.dcd",
        help=_help_text(visible, "cadcd", "Reference PDB (CA only)"),
    )
    p.add_argument(
        "--refsel",
        type=str,
        default="A:B:C:D:E:F.2-91",
        help=_help_text(visible, "refsel", "Reference selection"),
    )
    p.add_argument(
        "--othersel",
        type=str,
        default="G:H:I:J:K:L.2-91",
        help=_help_text(visible, "othersel", "Other selection"),
    )
    p.add_argument(
        "--anchor",
        type=str,
        default="A:G",
        help=_help_text(
            visible,
            "anchor",
            "Anchor points as 'X:Y' where X in ref tiles, Y in other tiles",
        ),
    )
    p.add_argument(
        "--rot",
        type=str,
        default="90:90",
        help=_help_text(
            visible,
            "rot",
            "Rotation angle reference as 'rot1:rot2' in degrees",
        ),
    )
    p.add_argument(
        "--bias",
        type=str,
        default=None,
        help=_help_text(
            visible,
            "bias",
            "Bias range as 'min:max:delta' or 'bmin:bmax:bdelta'",
        ),
    )
    p.add_argument(
        "--biasdir",
        type=str,
        default="x",
        help=_help_text(visible, "biasdir", "Bias direction 'x', 'y', 'z'"),
    )
    p.add_argument(
        "--biasangle",
        type=str,
        default=None,
        help=_help_text(
            visible,
            "biasangle",
            "Bias angle range as 'min:max:delta' in degrees",
        ),
    )
    p.add_argument(
        "--k",
        type=str,
        default=None,
        help=_help_text(
            visible,
            "k",
            "Force constants as 'kinit:kbias[:kdist[:kcent[:kangle]]]'",
        ),
    )
    p.add_argument(
        "--kinit",
        type=float,
        default=None,
        help=_help_text(visible, "kinit", "Initial force constant"),
    )
    p.add_argument(
        "--kbias",
        type=float,
        default=None,
        help=_help_text(visible, "kbias", "Force constant for distance bias"),
    )
    p.add_argument(
        "--kbiasangle",
        type=float,
        default=None,
        help=_help_text(visible, "kbiasangle", "Force constant for angle bias"),
    )
    p.add_argument(
        "--biaspairs",
        type=str,
        default=None,
        help=_help_text(
            visible,
            "biaspairs",
            "Explicit bias/biasangle pairs. Overrides --bias/--biasangle " "grid generation.",
        ),
    )
    p.add_argument(
        "--kdistx",
        type=float,
        default=None,
        help=_help_text(
            visible,
            "kdistx",
            "Force constant for distance x, if not bias",
        ),
    )
    p.add_argument(
        "--kdisty",
        type=float,
        default=None,
        help=_help_text(
            visible,
            "kdisty",
            "Force constant for distance y, if not bias",
        ),
    )
    p.add_argument(
        "--kdistz",
        type=float,
        default=None,
        help=_help_text(
            visible,
            "kdistz",
            "Force constant for distance z, if not bias",
        ),
    )
    p.add_argument(
        "--kdist",
        type=float,
        default=None,
        help=_help_text(
            visible,
            "kdist",
            "Force constant for distance between interaction units",
        ),
    )
    p.add_argument(
        "--kcent",
        type=float,
        default=None,
        help=_help_text(visible, "kcent", "Force constant for central force"),
    )
    p.add_argument(
        "--knorm",
        type=float,
        default=None,
        help=_help_text(
            visible,
            "knorm",
            "Force constant for angle norm restraint",
        ),
    )
    p.add_argument(
        "--kdihed",
        type=float,
        default=None,
        help=_help_text(
            visible,
            "kdihed",
            "Force constant for dihedral twist restraint",
        ),
    )
    p.add_argument(
        "--krot",
        type=float,
        default=None,
        help=_help_text(
            visible,
            "krot",
            "Force constant for rotation restraint",
        ),
    )

    flip_grp = p.add_mutually_exclusive_group()
    flip_grp.add_argument(
        "--flip",
        dest="flip",
        action="store_true",
        help=_help_text(visible, "flip", "Flipped second ring"),
    )
    flip_grp.add_argument(
        "--no-flip",
        dest="flip",
        action="store_false",
        help=_help_text(visible, "flip", "No flipped second ring"),
    )
    p.set_defaults(flip=False)

    p.add_argument(
        "--normcap",
        type=float,
        default=None,
        help=_help_text(visible, "normcap", "Maximum normal angle"),
    )
    p.add_argument(
        "--distcap",
        type=float,
        default=None,
        help=_help_text(visible, "distcap", "Maximum distance between interacting units"),
    )
    p.add_argument(
        "--box",
        type=str,
        default=None,
        help=_help_text(
            visible,
            "box",
            "Box size in nm: x, x:y, or x:y:z (e.g. 22:11:9).",
        ),
    )
    p.add_argument(
        "--conc",
        type=float,
        default=100.0,
        help=_help_text(
            visible,
            "conc",
            "NaCl concentration in mM (allatom only)",
        ),
    )
    p.add_argument(
        "--surf",
        type=float,
        default=0.7,
        help=_help_text(
            visible,
            "surf",
            "surface scaling parameter (COCOMO only)",
        ),
    )

    orient_grp = p.add_mutually_exclusive_group()
    orient_grp.add_argument(
        "--orient",
        dest="orient",
        action="store_true",
        help=_help_text(
            visible,
            "orient",
            "Orient using refsel plane and rotate othersel into x-axis",
        ),
    )
    orient_grp.add_argument(
        "--no-orient",
        dest="orient",
        action="store_false",
        help=_help_text(visible, "orient", "Disable orientation step"),
    )
    p.set_defaults(orient=True)

    p.add_argument(
        "--ff",
        nargs="+",
        default=None,
        help=_help_text(
            visible,
            "ff",
            "OpenMM forcefield XMLs (allatom only).",
        ),
    )
    p.add_argument(
        "--device",
        type=int,
        default=0,
        help=_help_text(
            visible,
            "device",
            "Device index (OpenMM platform device id)",
        ),
    )
    p.add_argument(
        "--resources",
        type=str,
        default="CUDA",
        help=_help_text(
            visible,
            "resources",
            "OpenMM platform/resources string",
        ),
    )

    p.add_argument(
        "--config",
        type=Path,
        default=Path("config"),
        help=_help_text(
            visible,
            "config",
            "Config file (key/value) to read/write",
        ),
    )
    p.add_argument(
        "--no-write-config",
        dest="write_config",
        action="store_false",
        help=_help_text(
            visible,
            "write_config",
            "Disable writing updated config values",
        ),
    )
    p.set_defaults(write_config=True)


def _explicit_dests(
    p: argparse.ArgumentParser,
    argv: Optional[Sequence[str]],
) -> set[str]:
    tokens = list(sys.argv[1:] if argv is None else argv)
    option_map: dict[str, str] = {}

    for action in p._actions:
        for option in action.option_strings:
            option_map[option] = action.dest

    explicit: set[str] = set()
    for token in tokens:
        if token == "--":
            break
        if not token.startswith("-"):
            continue

        option = token.split("=", 1)[0]
        dest = option_map.get(option)
        if dest not in {None, "help"}:
            explicit.add(dest)

    return explicit


def parse_args(
    cfg: dict[str, str],
    visible: Sequence[str],
    argv: Optional[Sequence[str]] = None,
    *,
    prog: str,
) -> argparse.Namespace:
    visible_names = _normalize_visible(visible)
    parser = argparse.ArgumentParser(
        prog=prog,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_all_arguments(parser, visible_names)
    apply_config_defaults(parser, cfg)

    args = parser.parse_args(argv)
    setattr(args, "_tileumbrella_visible", frozenset(visible_names))
    setattr(args, "_tileumbrella_explicit", frozenset(_explicit_dests(parser, argv)))
    return args


def write_args_config(
    cfg_path: Path,
    cfg: dict[str, str],
    args: argparse.Namespace,
    *,
    overrides: Optional[dict[str, object]] = None,
) -> dict[str, str]:
    out = dict(cfg)
    visible = set(getattr(args, "_tileumbrella_visible", ()))
    explicit = set(getattr(args, "_tileumbrella_explicit", ()))
    override_map: dict[str, object] = {}

    if overrides is not None:
        for name, value in overrides.items():
            key = _config_key(name) if name in _PERSISTENT_DESTS else name
            override_map[key] = value

    for dest in _PERSISTENT_DESTS:
        key = _config_key(dest)
        include = dest in visible or dest in explicit or key in out or key in override_map
        if not include or not hasattr(args, dest):
            continue

        if key in override_map:
            value = override_map.pop(key)
        else:
            value = getattr(args, dest)

        resolved = _resolve_config_value(dest, args, value)
        out[key] = format_value(_normalize_config_value(dest, resolved))

    for key, value in override_map.items():
        out[key] = format_value(value)

    write_config(cfg_path, out)
    return out


def parse_config_path(argv: Optional[Sequence[str]] = None) -> Path:
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
    last: Optional[float] = None

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


def find_input_file(start_dir: Union[Path, str], filename: str) -> Path:
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
        bselc = (
            f"{other_tiles[(other_index + 2) % other_size]}:"
            f"{other_tiles[(other_index + 5) % other_size]}{other_tail}"
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
        bselc = bsel2
    elif other_size == 3:
        bsel1 = f"{other_tiles[other_index]}{other_tail}"
        bsel2 = f"{other_tiles[(other_index + 1) % other_size]}{other_tail}"
        bselt = (
            f"{other_tiles[other_index]}:"
            f"{other_tiles[(other_index + 1) % other_size]}{other_tail}"
        )
        bselc = bsel2
    else:
        raise SystemExit("ERROR: invalid length of other selection")

    return asel1, asel2, aselt, bsel1, bsel2, bselt, as11, as12, bselc


def format_bias_tag(bias: float, biasangle: Optional[float] = None) -> str:
    if biasangle is None:
        return f"{bias:.2f}"
    return f"{bias:.2f}_{biasangle:.0f}"


def parse_bias_target(spec: str) -> tuple[float, Optional[float]]:
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
    bias: Optional[str],
    biasangle: Optional[str] = None,
    biaspairs: Optional[str] = None,
) -> list[tuple[float, Optional[float]]]:
    if biaspairs is not None:
        return [(biasval, angleval) for biasval, angleval in parse_bias_pairs_arg(biaspairs)]

    bias_spec = "6.0:9.0:0.1" if bias is None else bias
    bmin, bmax, bdel = parse_floats(bias_spec, [6.0, 9.0, 0.1], n_out=3)
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
