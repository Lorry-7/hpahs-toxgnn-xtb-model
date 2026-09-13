"""xTB calculation runner for quantum chemical descriptors.

Reference: ToxGNN_Project_2 compute_stage3_xtb_descriptors.py
Uses --sp (single-point) + --alpb water for robust descriptor extraction.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


def sanitize_mol_id(mol_id: str) -> str:
    """Sanitize molecule ID for use in file paths.

    Replaces characters that are invalid in Windows/Linux file paths.
    """
    # Replace invalid file path characters with underscore
    sanitized = re.sub(r'[<>:"/\\|?*\[\]]', '_', mol_id)
    # Replace multiple spaces/underscores with single underscore
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip('. ')
    # Truncate to reasonable length (Windows max path is 260)
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    return sanitized


@dataclass
class XTBConfig:
    """Configuration for xTB calculations."""

    executable: str = "xtb"
    timeout_sec: int = 300
    method: str = "gfn2"
    solvent_model: str = "alpb"
    solvent: str = "water"
    uhf: int = 0
    use_sp: bool = True  # single-point (True) vs geometry optimization (False)


@dataclass
class XTBJobResult:
    """Result of a single xTB job."""

    mol_id: str
    status: str
    command: str
    runtime_sec: float
    input_xyz: str
    output_dir: str
    error_message: str | None
    descriptors: dict[str, float]


def generate_lowest_energy_xyz(
    smiles: str,
    xyz_path: Path,
    n_confs: int = 20,
    seed: int = 42,
) -> int:
    """Generate XYZ file from lowest energy conformer.

    Returns formal charge of the molecule.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params))

    if not conf_ids:
        raise RuntimeError("RDKit conformer generation failed")

    # Optimize and score each conformer
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    energies = []
    for cid in conf_ids:
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
        if ff is not None:
            ff.Initialize()
            ff.Minimize(maxIts=500)
            energies.append((float(ff.CalcEnergy()), cid))
        else:
            energies.append((float("inf"), cid))

    best_cid = min(energies, key=lambda x: x[0])[1]
    conf = mol.GetConformer(best_cid)

    # Write XYZ file
    xyz_path.parent.mkdir(parents=True, exist_ok=True)
    with xyz_path.open("w", encoding="utf-8") as f:
        f.write(f"{mol.GetNumAtoms()}\n")
        f.write(f"generated_by_toxgnn conf={best_cid}\n")
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            f.write(f"{atom.GetSymbol()} {pos.x:.8f} {pos.y:.8f} {pos.z:.8f}\n")

    # Get formal charge
    charge = Chem.GetFormalCharge(mol)
    return charge


def run_xtb_job(
    mol_id: str,
    xyz_path: Path,
    out_dir: Path,
    charge: int,
    cfg: XTBConfig,
) -> XTBJobResult:
    """Run a single xTB job.

    Uses subprocess.PIPE to capture stdout+stderr, then combines them
    for parsing (xTB writes output to both streams depending on platform).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy XYZ to run directory as mol.xyz (reference approach)
    local_xyz = out_dir / "mol.xyz"
    shutil.copyfile(xyz_path, local_xyz)

    # Clean stale restart files
    for fname in ["restart", "xtbrestart", "xtblast.xyz", "xtbopt.xyz", "charges"]:
        p = out_dir / fname
        if p.exists():
            p.unlink()

    # Build command
    cmd = [cfg.executable, "mol.xyz", "--gfn", "2", "--chrg", str(charge)]
    if cfg.uhf:
        cmd.extend(["--uhf", str(cfg.uhf)])
    if cfg.use_sp:
        cmd.append("--sp")
    if cfg.solvent_model and cfg.solvent:
        cmd.extend([f"--{cfg.solvent_model}", cfg.solvent])

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(out_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=cfg.timeout_sec,
        )

        # Combine stdout + stderr (reference approach)
        out_text = (
            proc.stdout.decode("utf-8", errors="replace")
            + "\n"
            + proc.stderr.decode("utf-8", errors="replace")
        )

        # Save combined output
        (out_dir / "xtb_full_output.txt").write_text(out_text, encoding="utf-8", errors="ignore")

        status = "success" if proc.returncode == 0 else "failed"
        error = None if status == "success" else f"xTB return code {proc.returncode}"

    except subprocess.TimeoutExpired:
        out_text = ""
        status, error = "timeout", f"timeout after {cfg.timeout_sec} sec"
        (out_dir / "xtb_full_output.txt").write_text(error, encoding="utf-8")

    except FileNotFoundError:
        out_text = ""
        status, error = "not_found", f"xTB executable not found: {cfg.executable}"
        (out_dir / "xtb_full_output.txt").write_text(error, encoding="utf-8")

    runtime = time.time() - t0

    return XTBJobResult(
        mol_id=mol_id,
        status=status,
        command=" ".join(cmd),
        runtime_sec=runtime,
        input_xyz=str(xyz_path),
        output_dir=str(out_dir),
        error_message=error,
        descriptors={},
    )


def run_xtb_single(
    mol_id: str,
    smiles: str,
    xtb_dir: Path,
    charge: int,
    cfg: XTBConfig,
    n_confs: int = 20,
    seed: int = 42,
) -> XTBJobResult:
    """Run xTB job for a single molecule (no retry).

    Matches reference approach: single attempt, NaN on failure.
    """
    # Sanitize mol_id for file paths
    safe_id = sanitize_mol_id(mol_id)

    xyz_dir = xtb_dir / "inputs"
    out_dir = xtb_dir / "outputs" / safe_id

    # Generate XYZ
    xyz_path = xyz_dir / f"{safe_id}.xyz"
    try:
        generate_lowest_energy_xyz(smiles, xyz_path, n_confs, seed)
    except Exception as e:
        return XTBJobResult(
            mol_id=mol_id,
            status="conformer_failed",
            command="",
            runtime_sec=0.0,
            input_xyz=str(xyz_path),
            output_dir=str(out_dir),
            error_message=str(e),
            descriptors={},
        )

    return run_xtb_job(mol_id, xyz_path, out_dir, charge, cfg)
