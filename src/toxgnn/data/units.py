"""LC50 unit conversion utilities."""

from __future__ import annotations

import math


UNIT_TO_MOLAR = {
    "g/L": lambda v, mw: v / mw,
    "mg/L": lambda v, mw: v * 1e-3 / mw,
    "ug/L": lambda v, mw: v * 1e-6 / mw,
    "ng/L": lambda v, mw: v * 1e-9 / mw,
    "mol/L": lambda v, mw: v,
    "mmol/L": lambda v, mw: v * 1e-3,
    "umol/L": lambda v, mw: v * 1e-6,
    "nmol/L": lambda v, mw: v * 1e-9,
}

ACCEPTED_UNITS = set(UNIT_TO_MOLAR.keys())


def normalize_unit(unit: str) -> str:
    """Normalize unit string to standard form."""
    u = str(unit).strip().replace("µ", "u").replace("μ", "u")
    aliases = {
        "mg/l": "mg/L",
        "ug/l": "ug/L",
        "ng/l": "ng/L",
        "g/l": "g/L",
        "mol/l": "mol/L",
        "mmol/l": "mmol/L",
        "umol/l": "umol/L",
        "nmol/l": "nmol/L",
    }
    return aliases.get(u.lower(), u)


def convert_lc50_to_pLC50(
    value: float,
    unit: str,
    mw: float,
    qualifier: str = "=",
) -> dict:
    """Convert LC50 to pLC50 = -log10(LC50 [mol/L]).

    Returns dict with LC50_mol_L, pLC50, unit_normalized, train_allowed, qualifier.
    Raises ValueError for invalid inputs.
    """
    if value is None or float(value) <= 0:
        raise ValueError(f"LC50 must be positive, got {value}")
    if mw is None or float(mw) <= 0:
        raise ValueError(f"Molecular weight must be positive, got {mw}")

    norm_unit = normalize_unit(unit)
    if norm_unit not in UNIT_TO_MOLAR:
        raise ValueError(f"Unsupported LC50 unit: {unit}")

    mol_l = float(UNIT_TO_MOLAR[norm_unit](float(value), float(mw)))
    if mol_l <= 0 or not math.isfinite(mol_l):
        raise ValueError(f"Invalid converted LC50 mol/L: {mol_l}")

    return {
        "LC50_mol_L": mol_l,
        "pLC50": -math.log10(mol_l),
        "unit_normalized": norm_unit,
        "train_allowed": str(qualifier).strip() == "=",
        "qualifier": str(qualifier).strip(),
    }


def pLC50_to_lc50_mgL(pLC50: float, mw: float) -> float:
    """Convert pLC50 back to LC50 in mg/L."""
    mol_l = math.pow(10.0, -float(pLC50))
    return mol_l * float(mw) * 1000.0


def batch_convert_lc50(
    values: list[float],
    units: list[str],
    mws: list[float],
    qualifiers: list[str] | None = None,
) -> list[dict]:
    """Batch convert LC50 values. Returns list of conversion results."""
    if qualifiers is None:
        qualifiers = ["="] * len(values)

    results = []
    for val, unit, mw, qual in zip(values, units, mws, qualifiers):
        try:
            result = convert_lc50_to_pLC50(val, unit, mw, qual)
            result["status"] = "success"
            result["error"] = None
        except ValueError as e:
            result = {
                "LC50_mol_L": None,
                "pLC50": None,
                "unit_normalized": normalize_unit(unit),
                "train_allowed": False,
                "qualifier": str(qual).strip(),
                "status": "error",
                "error": str(e),
            }
        results.append(result)

    return results
