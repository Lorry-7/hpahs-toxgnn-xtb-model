"""Parser for xTB output files.

Reference: ToxGNN_Project_2 compute_stage3_xtb_descriptors.py
Parses 10 quantum chemical descriptors from xTB output:
  total_energy, homo, lumo, gap, dipole, q_min, q_max, q_abs_max, charge_range,
  isotropic_polarizability
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def parse_xtb_output(output_dir: str | Path) -> dict[str, float]:
    """Parse xTB output directory for quantum chemical descriptors.

    Reads the combined output file (xtb_full_output.txt) and the charges file.
    Returns dict with 10 raw descriptors (including isotropic_polarizability).

    Reference: compute_stage3_xtb_descriptors.py parse_xtb_output() + parse_charges()
    """
    output_dir = Path(output_dir)

    # Read combined output (stdout+stderr)
    full_output_path = output_dir / "xtb_full_output.txt"
    if full_output_path.exists():
        text = full_output_path.read_text(encoding="utf-8", errors="ignore")
    else:
        # Fallback: try reading stdout.txt + stderr.txt
        stdout_path = output_dir / "stdout.txt"
        stderr_path = output_dir / "stderr.txt"
        stdout = stdout_path.read_text(encoding="utf-8", errors="ignore") if stdout_path.exists() else ""
        stderr = stderr_path.read_text(encoding="utf-8", errors="ignore") if stderr_path.exists() else ""
        text = stdout + "\n" + stderr

    if not text.strip():
        raise FileNotFoundError(f"xTB output is empty in: {output_dir}")

    result = {}

    # Total energy (Hartree) - matches "TOTAL ENERGY  -32.0747  Eh" or ":: total energy ... Eh ::"
    m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", text, flags=re.IGNORECASE)
    if m:
        result["total_energy"] = float(m.group(1))

    # HOMO-LUMO gap (eV) - matches "HOMO-LUMO GAP  3.7181  eV" or ":: HOMO-LUMO gap ... eV ::"
    m = re.search(r"HOMO-LUMO GAP\s+([\d\.]+)", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"HL-Gap\s+[\d\.]+\s+Eh\s+([\d\.]+)\s+eV", text, flags=re.IGNORECASE)
    if m:
        result["gap"] = float(m.group(1))

    # HOMO / LUMO (eV) - formats: "-10.4846 (HOMO)" or "HOMO    -6.54 eV"
    homo_matches = re.findall(r"(-?\d+\.\d+)\s+\(HOMO\)", text, flags=re.IGNORECASE)
    lumo_matches = re.findall(r"(-?\d+\.\d+)\s+\(LUMO\)", text, flags=re.IGNORECASE)
    if not homo_matches:
        m = re.search(r"HOMO\s+(-?\d+\.\d+)\s+eV", text, flags=re.IGNORECASE)
        if m:
            homo_matches = [m.group(1)]
    if not lumo_matches:
        m = re.search(r"LUMO\s+(-?\d+\.\d+)\s+eV", text, flags=re.IGNORECASE)
        if m:
            lumo_matches = [m.group(1)]
    if homo_matches:
        result["homo"] = float(homo_matches[-1])
    if lumo_matches:
        result["lumo"] = float(lumo_matches[0])

    # Dipole moment (Debye) - in "molecular dipole:" block, "full: x y z total" line
    dipole_block = re.search(
        r"molecular dipole:(.+?)(?=molecular quadrupole|\Z)",
        text, flags=re.DOTALL | re.IGNORECASE,
    )
    if dipole_block:
        full_match = re.search(
            r"full:\s+[\d\.\-]+\s+[\d\.\-]+\s+[\d\.\-]+\s+([\d\.]+)",
            dipole_block.group(1),
        )
        if full_match:
            result["dipole"] = float(full_match.group(1))
    # Fallback: table format "x y z tot" then "1.234 2.345 3.456 4.567 Debye"
    if "dipole" not in result:
        m = re.search(
            r"molecular dipole.+?^\s*[\d\.\-]+\s+[\d\.\-]+\s+[\d\.\-]+\s+([\d\.]+)\s+Debye",
            text, flags=re.DOTALL | re.IGNORECASE | re.MULTILINE,
        )
        if m:
            result["dipole"] = float(m.group(1))

    # Atomic charges from charges file
    charges_result = _parse_charges_file(output_dir / "charges")
    # Fallback: parse Mulliken charges from text if charges file missing
    if np.isnan(charges_result.get("min_charge", np.nan)):
        charges_from_text = _parse_charges_from_text(text)
        if charges_from_text is not None:
            charges_result = charges_from_text
    result.update(charges_result)

    # Isotropic average polarizability (atomic units)
    # Matches: "Mol. α(0) /au        :        105.233576"
    m = re.search(r"Mol\.\s*α\(0\)\s*/au\s*:\s+(-?\d+\.\d+)", text)
    if m:
        result["isotropic_polarizability"] = float(m.group(1))

    return result


def _parse_charges_file(charges_path: Path) -> dict[str, float]:
    """Parse Mulliken atomic charges from xTB charges file.

    Reference: compute_stage3_xtb_descriptors.py parse_charges()
    File format: atomic_number  charge (one per line)
    """
    result = {
        "min_charge": np.nan,
        "max_charge": np.nan,
        "max_abs_charge": np.nan,
        "charge_range": np.nan,
    }

    if not charges_path.exists():
        return result

    charges = []
    with open(charges_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split()
                charges.append(float(parts[-1]))
            except Exception:
                continue

    if len(charges) == 0:
        return result

    charges = np.array(charges, dtype=float)
    result["min_charge"] = float(np.min(charges))
    result["max_charge"] = float(np.max(charges))
    result["max_abs_charge"] = float(np.max(np.abs(charges)))
    result["charge_range"] = float(np.max(charges) - np.min(charges))
    return result


def _parse_charges_from_text(text: str) -> dict[str, float] | None:
    """Parse Mulliken charges from xTB stdout text block.

    Matches lines like: '1  C    -0.123' or '2  C     0.456'
    """
    charge_block = re.search(
        r"Mulliken charges(.+?)(?:::$|\Z)",
        text, flags=re.DOTALL | re.IGNORECASE,
    )
    if not charge_block:
        return None
    charges = []
    for m in re.finditer(r"^\s*\d+\s+\w+\s+(-?\d+\.\d+)", charge_block.group(1), re.MULTILINE):
        charges.append(float(m.group(1)))
    if not charges:
        return None
    charges = np.array(charges, dtype=float)
    return {
        "min_charge": float(np.min(charges)),
        "max_charge": float(np.max(charges)),
        "max_abs_charge": float(np.max(np.abs(charges))),
        "charge_range": float(np.max(charges) - np.min(charges)),
    }


def extract_qs5_descriptors(raw_descriptors: dict[str, float]) -> dict[str, float]:
    """Extract QS5 subset from raw 9 descriptors."""
    qs5_keys = ["total_energy", "homo", "lumo", "gap", "charge_range"]
    return {k: raw_descriptors.get(k, float("nan")) for k in qs5_keys}


def validate_descriptors(descriptors: dict[str, float], required_keys: list[str]) -> bool:
    """Check that all required keys are present and finite."""
    for key in required_keys:
        if key not in descriptors:
            return False
        if not isinstance(descriptors[key], (int, float)):
            return False
        if not (-1e10 < descriptors[key] < 1e10):
            return False
    return True
