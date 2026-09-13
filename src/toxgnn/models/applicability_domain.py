"""Applicability Domain assessment for model predictions."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import pairwise_distances

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def morgan_fp(smiles: str, radius: int = 2, nbits: int = 2048):
    """Generate Morgan fingerprint for a molecule."""
    if not HAS_RDKIT:
        raise ImportError("RDKit required for fingerprint generation")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def max_tanimoto(fp, ref_fps) -> float:
    """Calculate maximum Tanimoto similarity to reference set."""
    if not ref_fps:
        return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
    return float(max(sims)) if sims else 0.0


@dataclass
class ADThresholds:
    """Applicability domain thresholds."""

    structural_min_sim: float
    embedding_max_dist: float
    xtb_max_mahalanobis: float


class ApplicabilityDomain:
    """Applicability Domain assessment using structural, embedding, and xTB criteria."""

    def __init__(self, k: int = 5):
        self.k = k
        self.thresholds: ADThresholds | None = None
        self.ref_fps = None
        self.ref_embeddings = None
        self.xtb_cov = None
        self.ref_xtb = None

    def fit(
        self,
        ref_smiles: list[str],
        ref_embeddings: np.ndarray,
        ref_xtb: np.ndarray,
    ) -> "ApplicabilityDomain":
        """Fit AD thresholds from reference training data."""
        self.ref_fps = [morgan_fp(s) for s in ref_smiles]
        self.ref_embeddings = np.asarray(ref_embeddings, dtype=float)
        self.ref_xtb = np.asarray(ref_xtb, dtype=float)

        # Leave-one-out nearest structural similarity
        sims = []
        for i, fp in enumerate(self.ref_fps):
            others = self.ref_fps[:i] + self.ref_fps[i + 1:]
            sims.append(max_tanimoto(fp, others))
        structural_min = float(np.percentile(sims, 5))

        # Leave-one-out kNN distance in embedding space
        D = pairwise_distances(self.ref_embeddings)
        np.fill_diagonal(D, np.nan)
        kth_dist = np.nanmean(np.sort(D, axis=1)[:, :self.k], axis=1)
        emb_max = float(np.percentile(kth_dist, 95))

        # xTB Mahalanobis distance
        self.xtb_cov = EmpiricalCovariance().fit(self.ref_xtb)
        maha = self.xtb_cov.mahalanobis(self.ref_xtb)
        xtb_max = float(np.percentile(maha, 95))

        self.thresholds = ADThresholds(structural_min, emb_max, xtb_max)
        return self

    def assess(
        self,
        smiles: str,
        embedding: np.ndarray,
        xtb: np.ndarray | None,
    ) -> dict:
        """Assess if a sample is within the applicability domain."""
        if self.thresholds is None:
            raise RuntimeError("AD must be fitted before assess")

        # Structural similarity
        sim = max_tanimoto(morgan_fp(smiles), self.ref_fps)

        # Embedding distance
        d = pairwise_distances(
            np.asarray(embedding).reshape(1, -1), self.ref_embeddings
        )
        emb_d = float(np.mean(np.sort(d.ravel())[:self.k]))

        # xTB Mahalanobis distance
        if xtb is None or np.any(~np.isfinite(xtb)):
            xtb_d, xtb_flag = np.nan, "missing"
        else:
            xtb_d = float(
                self.xtb_cov.mahalanobis(np.asarray(xtb).reshape(1, -1))[0]
            )
            xtb_flag = "inside" if xtb_d <= self.thresholds.xtb_max_mahalanobis else "outside"

        # Flags
        structural_flag = "inside" if sim >= self.thresholds.structural_min_sim else "outside"
        emb_flag = "inside" if emb_d <= self.thresholds.embedding_max_dist else "outside"

        # Majority voting
        votes = sum([
            structural_flag == "inside",
            emb_flag == "inside",
            xtb_flag == "inside",
        ])
        if votes >= 2:
            overall = "inside"
        elif votes == 1:
            overall = "boundary"
        else:
            overall = "outside"

        return {
            "nearest_tanimoto": sim,
            "embedding_distance": emb_d,
            "xtb_mahalanobis": xtb_d,
            "AD_structural": structural_flag,
            "AD_embedding": emb_flag,
            "AD_xtb": xtb_flag,
            "AD_overall": overall,
        }

    def batch_assess(
        self,
        smiles_list: list[str],
        embeddings: np.ndarray,
        xtb_descriptors: np.ndarray | None = None,
    ) -> list[dict]:
        """Assess multiple samples."""
        results = []
        for i, smi in enumerate(smiles_list):
            xtb = xtb_descriptors[i] if xtb_descriptors is not None else None
            results.append(self.assess(smi, embeddings[i], xtb))
        return results
