from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Optional, Union

import numpy as np

ArrayLike = Union[float, Sequence[float], np.ndarray]


@dataclass(frozen=True)
class SwitchTerms:
    """Expanded radial switch terms, one row per interacting pair."""

    pairs: np.ndarray
    eps: np.ndarray
    r0: np.ndarray
    alpha: np.ndarray
    term_id: np.ndarray

    def __post_init__(self) -> None:
        pairs = np.asarray(self.pairs, dtype=np.int64)
        eps = np.asarray(self.eps, dtype=np.float64)
        r0 = np.asarray(self.r0, dtype=np.float64)
        alpha = np.asarray(self.alpha, dtype=np.float64)
        term_id = np.asarray(self.term_id, dtype=np.int64)

        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must have shape (n_pairs, 2)")

        n_pairs = int(pairs.shape[0])
        for name, arr in (
            ("eps", eps),
            ("r0", r0),
            ("alpha", alpha),
            ("term_id", term_id),
        ):
            if arr.shape != (n_pairs,):
                raise ValueError(f"{name} must have shape ({n_pairs},)")

        object.__setattr__(self, "pairs", pairs)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "r0", r0)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "term_id", term_id)

    @property
    def n_pairs(self) -> int:
        return int(self.pairs.shape[0])

    @property
    def atom_indices(self) -> np.ndarray:
        if self.n_pairs == 0:
            return np.empty((0,), dtype=np.int64)
        return np.unique(self.pairs.reshape(-1))


def switch_terms_from_interactions(
    interactions: Sequence[Any],
    *,
    include_angle: bool = False,
    default_alpha: float = 10.0,
) -> SwitchTerms:
    """
    Expand COCOMO switch interactions into one row per pair.

    Notes
    -----
    - Only radial 'switch' terms are included by default.
    - Angle-gated switch terms are skipped unless include_angle=True.
    - A zero/None alpha is replaced by default_alpha to match COCOMO.
    """

    pair_rows: list[tuple[int, int]] = []
    eps_rows: list[float] = []
    r0_rows: list[float] = []
    alpha_rows: list[float] = []
    term_rows: list[int] = []

    for term_id, intr in enumerate(interactions):
        func = str(getattr(intr, "function", "switch") or "switch").lower()
        if func != "switch":
            continue

        if bool(getattr(intr, "angle", False)) and not include_angle:
            continue

        eps = float(getattr(intr, "strength", 0.0))
        r0 = float(getattr(intr, "distance", 0.0))
        alpha = float(getattr(intr, "parameter", 0.0) or 0.0)
        if alpha == 0.0:
            alpha = float(default_alpha)

        for i_raw, j_raw in getattr(intr, "pairs", ()) or ():
            i = int(i_raw)
            j = int(j_raw)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            pair_rows.append((i, j))
            eps_rows.append(eps)
            r0_rows.append(r0)
            alpha_rows.append(alpha)
            term_rows.append(int(term_id))

    if not pair_rows:
        return SwitchTerms(
            pairs=np.empty((0, 2), dtype=np.int64),
            eps=np.empty((0,), dtype=np.float64),
            r0=np.empty((0,), dtype=np.float64),
            alpha=np.empty((0,), dtype=np.float64),
            term_id=np.empty((0,), dtype=np.int64),
        )

    return SwitchTerms(
        pairs=np.asarray(pair_rows, dtype=np.int64),
        eps=np.asarray(eps_rows, dtype=np.float64),
        r0=np.asarray(r0_rows, dtype=np.float64),
        alpha=np.asarray(alpha_rows, dtype=np.float64),
        term_id=np.asarray(term_rows, dtype=np.int64),
    )


def switch_terms_from_assembly(
    assembly: Any,
    *,
    include_angle: bool = False,
    default_alpha: float = 10.0,
) -> SwitchTerms:
    """Build SwitchTerms from assembly.get_interactions()."""
    interactions = assembly.get_interactions()
    return switch_terms_from_interactions(
        interactions,
        include_angle=include_angle,
        default_alpha=default_alpha,
    )


def update_switch_terms(
    terms: SwitchTerms,
    *,
    eps: Optional[ArrayLike] = None,
    r0: Optional[ArrayLike] = None,
    alpha: Optional[ArrayLike] = None,
    pair_mask: Optional[np.ndarray] = None,
    term_ids: Optional[Sequence[int]] = None,
) -> SwitchTerms:
    """
    Return a modified copy of SwitchTerms.

    Parameters
    ----------
    eps, r0, alpha
        Accepted forms:
        - scalar: applied to all selected pairs
        - length-n_selected: one value per selected pair
        - length-n_terms: one value per selected term_id, broadcast to all
          selected pairs in that term. This mode is enabled when term_ids
          is provided.
    pair_mask
        Boolean mask over pairs.
    term_ids
        Select all pairs belonging to one or more original interaction ids.
    """

    mask = np.ones((terms.n_pairs,), dtype=bool)
    if pair_mask is not None:
        pair_mask = np.asarray(pair_mask, dtype=bool)
        if pair_mask.shape != (terms.n_pairs,):
            raise ValueError("pair_mask has wrong shape")
        mask &= pair_mask

    term_ids_arr: Optional[np.ndarray] = None
    if term_ids is not None:
        term_ids_arr = np.asarray(term_ids, dtype=np.int64).reshape(-1)
        mask &= np.isin(terms.term_id, term_ids_arr)

    eps_out = terms.eps.copy()
    r0_out = terms.r0.copy()
    alpha_out = terms.alpha.copy()

    if eps is not None:
        eps_out[mask] = _broadcast_selected_or_terms(
            eps,
            mask=mask,
            selected_term_ids=term_ids_arr,
            pair_term_id=terms.term_id,
            name="eps",
        )
    if r0 is not None:
        r0_out[mask] = _broadcast_selected_or_terms(
            r0,
            mask=mask,
            selected_term_ids=term_ids_arr,
            pair_term_id=terms.term_id,
            name="r0",
        )
    if alpha is not None:
        alpha_out[mask] = _broadcast_selected_or_terms(
            alpha,
            mask=mask,
            selected_term_ids=term_ids_arr,
            pair_term_id=terms.term_id,
            name="alpha",
        )

    return replace(
        terms,
        eps=eps_out,
        r0=r0_out,
        alpha=alpha_out,
    )


def _broadcast_selected_or_terms(
    values: ArrayLike,
    *,
    mask: np.ndarray,
    selected_term_ids: Optional[np.ndarray],
    pair_term_id: np.ndarray,
    name: str,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    n_selected = int(mask.sum())

    if arr.ndim == 0:
        return np.full((n_selected,), float(arr), dtype=np.float64)

    arr = arr.reshape(-1)

    if arr.shape == (n_selected,):
        return arr.astype(np.float64, copy=False)

    if selected_term_ids is None:
        raise ValueError(f"{name} expected scalar or {n_selected} values, " f"got {arr.shape[0]}")

    if arr.shape != (selected_term_ids.shape[0],):
        raise ValueError(
            f"{name} expected scalar, {n_selected} pair values, or "
            f"{selected_term_ids.shape[0]} term values, got {arr.shape[0]}"
        )

    out = np.empty((n_selected,), dtype=np.float64)
    selected_pair_terms = pair_term_id[mask]

    for i, term_id in enumerate(selected_term_ids):
        out[selected_pair_terms == term_id] = arr[i]

    return out


def update_switch_terms_by_term(
    terms: SwitchTerms,
    *,
    eps: Optional[ArrayLike] = None,
    r0: Optional[ArrayLike] = None,
    alpha: Optional[ArrayLike] = None,
    term_ids: Optional[Sequence[int]] = None,
) -> SwitchTerms:
    if term_ids is None:
        term_ids = np.unique(terms.term_id)
    return update_switch_terms(
        terms,
        eps=eps,
        r0=r0,
        alpha=alpha,
        term_ids=term_ids,
    )


def switch_energy_from_distances(
    dist_nm: np.ndarray,
    terms: SwitchTerms,
) -> np.ndarray:
    """
    Evaluate the radial switch energy from precomputed distances.

    Parameters
    ----------
    dist_nm
        Shape (n_pairs,) or (n_frames, n_pairs), in nm.
    terms
        SwitchTerms with matching pair dimension.

    Returns
    -------
    np.ndarray
        Shape () or (n_frames,), in kJ/mol.
    """

    dist = np.asarray(dist_nm, dtype=np.float64)
    if dist.ndim == 1:
        if dist.shape != (terms.n_pairs,):
            raise ValueError("dist_nm has wrong shape")
        x = terms.alpha * (dist - terms.r0)
        x = np.clip(x, -50.0, 50.0)
        e = -terms.eps / (1.0 + np.exp(x))
        return np.asarray(e.sum(), dtype=np.float64)

    if dist.ndim != 2 or dist.shape[1] != terms.n_pairs:
        raise ValueError("dist_nm must have shape (n_frames, n_pairs)")

    x = terms.alpha.reshape(1, -1) * (dist - terms.r0.reshape(1, -1))
    x = np.clip(x, -50.0, 50.0)
    e = -terms.eps.reshape(1, -1) / (1.0 + np.exp(x))
    return e.sum(axis=1, dtype=np.float64)


def switch_energies_many_from_distances(
    dist_nm: np.ndarray,
    *,
    eps: np.ndarray,
    r0: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """
    Evaluate many parameter variants from one distance matrix.

    Parameters
    ----------
    dist_nm
        Shape (n_frames, n_pairs).
    eps, r0, alpha
        Shape (n_variants, n_pairs) or (n_pairs,).

    Returns
    -------
    np.ndarray
        Shape (n_variants, n_frames), in kJ/mol.
    """

    dist = np.asarray(dist_nm, dtype=np.float64)
    if dist.ndim != 2:
        raise ValueError("dist_nm must have shape (n_frames, n_pairs)")

    eps_2d = _as_2d_params(eps, dist.shape[1], "eps")
    r0_2d = _as_2d_params(r0, dist.shape[1], "r0")
    alpha_2d = _as_2d_params(alpha, dist.shape[1], "alpha")

    if not (eps_2d.shape == r0_2d.shape == alpha_2d.shape):
        raise ValueError("eps, r0, and alpha must have matching shapes")

    x = alpha_2d[:, None, :] * (dist[None, :, :] - r0_2d[:, None, :])
    x = np.clip(x, -50.0, 50.0)
    e = -eps_2d[:, None, :] / (1.0 + np.exp(x))
    return e.sum(axis=2, dtype=np.float64)


def switch_energy_offsets_from_distances(
    dist_nm: np.ndarray,
    reference: SwitchTerms,
    target: SwitchTerms,
) -> np.ndarray:
    """Return target - reference switch energy, in kJ/mol."""
    _check_same_pairs(reference, target)
    return switch_energy_from_distances(dist_nm, target) - switch_energy_from_distances(
        dist_nm, reference
    )


def umbrella_keep_mask(
    n_frames: int,
    *,
    n_umbrellas: int,
    umbrella_skip: int = 0,
    frames_per_umbrella: Optional[int] = None,
) -> np.ndarray:
    """
    Boolean mask for a concatenated umbrella trajectory.

    Parameters
    ----------
    n_frames
        Total number of frames in the concatenated trajectory.
    n_umbrellas
        Number of umbrella windows in the concatenation.
    umbrella_skip
        Number of leading frames to discard from each umbrella block.
    frames_per_umbrella
        Frames in each umbrella block. If omitted, inferred from n_frames.

    Returns
    -------
    np.ndarray
        Shape (n_frames,), dtype bool.
    """

    n_frames = int(n_frames)
    n_umbrellas = int(n_umbrellas)
    umbrella_skip = int(umbrella_skip)

    if n_frames < 0:
        raise ValueError("n_frames must be >= 0")
    if n_umbrellas <= 0:
        raise ValueError("n_umbrellas must be >= 1")
    if umbrella_skip < 0:
        raise ValueError("umbrella_skip must be >= 0")

    if n_frames == 0:
        return np.empty((0,), dtype=bool)

    if frames_per_umbrella is None:
        if n_frames % n_umbrellas != 0:
            raise ValueError(
                "n_frames is not divisible by n_umbrellas; "
                "provide frames_per_umbrella explicitly"
            )
        frames_per_umbrella = n_frames // n_umbrellas
    else:
        frames_per_umbrella = int(frames_per_umbrella)
        if frames_per_umbrella <= 0:
            raise ValueError("frames_per_umbrella must be >= 1")
        expected = n_umbrellas * frames_per_umbrella
        if n_frames != expected:
            raise ValueError(
                f"expected {expected} frames from "
                f"n_umbrellas={n_umbrellas} and "
                f"frames_per_umbrella={frames_per_umbrella}, got {n_frames}"
            )

    if umbrella_skip >= frames_per_umbrella:
        raise ValueError("umbrella_skip must be smaller than frames_per_umbrella")

    frame_in_block = np.arange(n_frames, dtype=np.int64) % frames_per_umbrella
    return frame_in_block >= umbrella_skip


def apply_umbrella_skip(
    values: np.ndarray,
    *,
    n_umbrellas: int,
    umbrella_skip: int = 0,
    frames_per_umbrella: Optional[int] = None,
    axis: int = 0,
) -> np.ndarray:
    """
    Drop the first umbrella_skip frames from each umbrella block.

    Parameters
    ----------
    values
        Array with the frame axis along ``axis``.
    n_umbrellas
        Number of umbrella windows in the concatenation.
    umbrella_skip
        Number of leading frames to discard from each umbrella block.
    frames_per_umbrella
        Frames in each umbrella block. If omitted, inferred from
        ``values.shape[axis]``.
    axis
        Axis corresponding to frames.

    Returns
    -------
    np.ndarray
        A view/copy with the skipped frames removed.
    """

    arr = np.asarray(values)
    if arr.ndim == 0:
        raise ValueError("values must have at least one dimension")

    axis = int(axis)
    if axis < 0:
        axis += arr.ndim
    if axis < 0 or axis >= arr.ndim:
        raise ValueError("axis is out of bounds")

    mask = umbrella_keep_mask(
        arr.shape[axis],
        n_umbrellas=n_umbrellas,
        umbrella_skip=umbrella_skip,
        frames_per_umbrella=frames_per_umbrella,
    )
    return np.compress(mask, arr, axis=axis)


def pair_distances_nm(
    xyz_nm: np.ndarray,
    pairs: np.ndarray,
    *,
    box_nm: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Pair distances for one frame or many frames.

    Parameters
    ----------
    xyz_nm
        Shape (n_atoms, 3) or (n_frames, n_atoms, 3), in nm.
    pairs
        Shape (n_pairs, 2), integer atom indices.
    box_nm
        None, shape (3,), or shape (n_frames, 3), in nm.
    """

    xyz = np.asarray(xyz_nm, dtype=np.float64)
    pair_idx = np.asarray(pairs, dtype=np.int64)

    if xyz.ndim == 2:
        d = xyz[pair_idx[:, 0], :] - xyz[pair_idx[:, 1], :]
        if box_nm is not None:
            box = _as_box_1d(box_nm)
            d -= np.rint(d / box.reshape(1, 3)) * box.reshape(1, 3)
        return np.linalg.norm(d, axis=1)

    if xyz.ndim != 3:
        raise ValueError("xyz_nm must have shape (n_atoms, 3) or (n_frames, n_atoms, 3)")

    d = xyz[:, pair_idx[:, 0], :] - xyz[:, pair_idx[:, 1], :]
    if box_nm is not None:
        box = np.asarray(box_nm, dtype=np.float64)
        if box.ndim == 1:
            box = _as_box_1d(box).reshape(1, 1, 3)
        elif box.ndim == 2 and box.shape[0] == xyz.shape[0] and box.shape[1] == 3:
            box = box.reshape(xyz.shape[0], 1, 3)
        else:
            raise ValueError("box_nm must have shape (3,) or (n_frames, 3)")
        d -= np.rint(d / box) * box
    return np.linalg.norm(d, axis=2)


def precompute_switch_distances_from_structure(
    trajectory: Any,
    terms: SwitchTerms,
    *,
    box_nm: Optional[ArrayLike] = None,
    n_umbrellas: Optional[int] = None,
    umbrella_skip: int = 0,
    frames_per_umbrella: Optional[int] = None,
) -> np.ndarray:
    """
    Precompute all switch-pair distances from a loaded trajectory.

    Notes
    -----
    molecule_data.load_dcd stores coordinates but not box lengths, so exact PBC
    matching requires passing box_nm explicitly or using iter_dcd-based streaming.

    If ``n_umbrellas`` is given, the first ``umbrella_skip`` frames of each
    umbrella block are removed from the returned distance matrix.
    """

    coords = getattr(trajectory, "_coords_nm", None)
    if coords is None:
        coords = _coords_from_models(trajectory)

    dist = pair_distances_nm(
        coords,
        terms.pairs,
        box_nm=None if box_nm is None else box_nm,
    )
    return _maybe_apply_umbrella_skip(
        dist,
        n_umbrellas=n_umbrellas,
        umbrella_skip=umbrella_skip,
        frames_per_umbrella=frames_per_umbrella,
        axis=0,
    )


def precompute_switch_distances_from_dcd(
    dcd_file: Any,
    template: Any,
    terms: SwitchTerms,
    *,
    stride: int = 1,
    chunk: int = 500,
    box_nm: Optional[ArrayLike] = None,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    n_umbrellas: Optional[int] = None,
    umbrella_skip: int = 0,
    frames_per_umbrella: Optional[int] = None,
) -> np.ndarray:
    """
    Stream a DCD and precompute all switch-pair distances.

    Only atoms needed by the switch pairs are read from the DCD.

    Umbrella skipping is applied after ``stride`` and ``frame_start/frame_stop``.
    If ``frames_per_umbrella`` is omitted, it is inferred from the number of
    selected frames and ``n_umbrellas``.
    """

    atom_idx, local_pairs = _compressed_pairs(terms.pairs)
    d_blocks: list[np.ndarray] = []

    for i_frame, (xyz_nm, box_frame_nm) in enumerate(
        _iter_dcd(
            dcd_file,
            template,
            chunk=int(chunk),
            stride=int(stride),
            atom_indices=atom_idx.tolist(),
        )
    ):
        if i_frame < int(frame_start):
            continue
        if frame_stop is not None and i_frame >= int(frame_stop):
            break

        box_use = _select_box(box_frame_nm, box_nm)
        d_frame = pair_distances_nm(
            xyz_nm,
            local_pairs,
            box_nm=box_use,
        )
        d_blocks.append(np.asarray(d_frame, dtype=np.float64))

    if not d_blocks:
        dist = np.empty((0, terms.n_pairs), dtype=np.float64)
    else:
        dist = np.vstack(d_blocks)

    return _maybe_apply_umbrella_skip(
        dist,
        n_umbrellas=n_umbrellas,
        umbrella_skip=umbrella_skip,
        frames_per_umbrella=frames_per_umbrella,
        axis=0,
    )


def switch_energies_from_dcd(
    dcd_file: Any,
    template: Any,
    terms: SwitchTerms,
    *,
    stride: int = 1,
    chunk: int = 500,
    box_nm: Optional[ArrayLike] = None,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    n_umbrellas: Optional[int] = None,
    umbrella_skip: int = 0,
    frames_per_umbrella: Optional[int] = None,
) -> np.ndarray:
    """Direct streaming energy evaluation for one SwitchTerms object."""
    dist = precompute_switch_distances_from_dcd(
        dcd_file,
        template,
        terms,
        stride=stride,
        chunk=chunk,
        box_nm=box_nm,
        frame_start=frame_start,
        frame_stop=frame_stop,
        n_umbrellas=n_umbrellas,
        umbrella_skip=umbrella_skip,
        frames_per_umbrella=frames_per_umbrella,
    )
    return switch_energy_from_distances(dist, terms)


def switch_energy_offsets_from_dcd(
    dcd_file: Any,
    template: Any,
    reference: SwitchTerms,
    target: SwitchTerms,
    *,
    stride: int = 1,
    chunk: int = 500,
    box_nm: Optional[ArrayLike] = None,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    n_umbrellas: Optional[int] = None,
    umbrella_skip: int = 0,
    frames_per_umbrella: Optional[int] = None,
) -> np.ndarray:
    """
    Stream a DCD once and return target - reference switch energies.
    """
    _check_same_pairs(reference, target)
    dist = precompute_switch_distances_from_dcd(
        dcd_file,
        template,
        reference,
        stride=stride,
        chunk=chunk,
        box_nm=box_nm,
        frame_start=frame_start,
        frame_stop=frame_stop,
        n_umbrellas=n_umbrellas,
        umbrella_skip=umbrella_skip,
        frames_per_umbrella=frames_per_umbrella,
    )
    return switch_energy_offsets_from_distances(dist, reference, target)


def _maybe_apply_umbrella_skip(
    values: np.ndarray,
    *,
    n_umbrellas: Optional[int] = None,
    umbrella_skip: int = 0,
    frames_per_umbrella: Optional[int] = None,
    axis: int = 0,
) -> np.ndarray:
    arr = np.asarray(values)
    if n_umbrellas is None and int(umbrella_skip) == 0 and frames_per_umbrella is None:
        return arr

    if n_umbrellas is None:
        raise ValueError("n_umbrellas is required when umbrella frame skipping is requested")

    return apply_umbrella_skip(
        arr,
        n_umbrellas=n_umbrellas,
        umbrella_skip=umbrella_skip,
        frames_per_umbrella=frames_per_umbrella,
        axis=axis,
    )


def _iter_dcd(*args: Any, **kwargs: Any) -> Any:
    try:
        from mdsim.molecule_data import iter_dcd
    except ImportError:  # pragma: no cover
        from mdsim.molecule_data import iter_dcd
    return iter_dcd(*args, **kwargs)


def _coords_from_models(trajectory: Any) -> np.ndarray:
    models = getattr(trajectory, "models", None)
    if models is None:
        raise ValueError("trajectory does not expose _coords_nm or models")

    coords_list: list[np.ndarray] = []
    for model in models:
        pos = model.positions()
        if hasattr(pos, "value_in_unit") and hasattr(pos, "unit"):
            pos_nm = pos.value_in_unit(pos.unit)
        else:
            pos_nm = pos
        coords_list.append(np.asarray(pos_nm, dtype=np.float64))
    if not coords_list:
        return np.empty((0, 0, 3), dtype=np.float64)
    return np.stack(coords_list, axis=0)


def _compressed_pairs(pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    atoms = np.unique(np.asarray(pairs, dtype=np.int64).reshape(-1))
    idx_map = {int(old): i for i, old in enumerate(atoms.tolist())}
    local = np.empty_like(pairs, dtype=np.int64)
    for i, (a, b) in enumerate(np.asarray(pairs, dtype=np.int64)):
        local[i, 0] = idx_map[int(a)]
        local[i, 1] = idx_map[int(b)]
    return atoms, local


def _as_box_1d(box_nm: ArrayLike) -> np.ndarray:
    box = np.asarray(box_nm, dtype=np.float64).reshape(-1)
    if box.shape != (3,):
        raise ValueError("box_nm must be a length-3 sequence")
    return box.copy()


def _select_box(
    box_frame_nm: Optional[np.ndarray],
    box_fallback_nm: Optional[ArrayLike],
) -> Optional[np.ndarray]:
    if box_frame_nm is not None:
        return _as_box_1d(box_frame_nm)
    if box_fallback_nm is not None:
        return _as_box_1d(box_fallback_nm)
    return None


def _broadcast_selected(values: ArrayLike, n_selected: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        return np.full((n_selected,), float(arr), dtype=np.float64)
    arr = arr.reshape(-1)
    if arr.shape != (n_selected,):
        raise ValueError(f"expected {n_selected} values, got {arr.shape[0]}")
    return arr


def _as_2d_params(values: np.ndarray, n_pairs: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape != (n_pairs,):
            raise ValueError(f"{name} must have shape ({n_pairs},)")
        return arr.reshape(1, n_pairs)
    if arr.ndim != 2 or arr.shape[1] != n_pairs:
        raise ValueError(f"{name} must have shape ({n_pairs},) or (n_variants, {n_pairs})")
    return arr


def _check_same_pairs(reference: SwitchTerms, target: SwitchTerms) -> None:
    if reference.pairs.shape != target.pairs.shape:
        raise ValueError("reference and target do not have the same pair layout")
    if not np.array_equal(reference.pairs, target.pairs):
        raise ValueError("reference and target pairs differ")
