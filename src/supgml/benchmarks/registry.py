"""Stable names for constructing benchmark problems."""

from .problems import (
    bump,
    curved_wave,
    curved_waves,
    cylinder,
    falloff,
    hemker,
    lifted_edge,
    wedge,
)


BENCHMARKS = {
    "wedge": wedge,
    "bump": bump,
    "lifted-edge": lifted_edge,
    "cylinder": cylinder,
    "falloff": falloff,
    "curved-wave": curved_wave,
    "curved-waves": curved_waves,
    "hemker": hemker,
}

_LEGACY_ORDER = tuple(BENCHMARKS)


def create(name, mesh=None, **parameters):
    """Construct a benchmark by readable name or legacy integer ID."""

    if not isinstance(name, str):
        try:
            name = _LEGACY_ORDER[int(name)]
        except (IndexError, TypeError, ValueError) as error:
            raise KeyError("unknown benchmark ID: {!r}".format(name)) from error
    normalized = name.lower().replace("_", "-")
    try:
        benchmark = BENCHMARKS[normalized]
    except KeyError as error:
        choices = ", ".join(BENCHMARKS)
        raise KeyError("unknown benchmark {!r}; choose from {}".format(name, choices)) from error
    return benchmark(mesh=mesh, **parameters)


def int_to_prblm(idx, mesh):
    """Compatibility wrapper for the original integer registry."""

    return create(idx, mesh=mesh)
