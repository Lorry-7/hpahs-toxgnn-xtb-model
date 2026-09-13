"""Data schema definitions for processed data files."""

from __future__ import annotations

import pandera as pa
from pandera import Column, Check, DataFrameSchema

# LogP master schema
LOGP_SCHEMA = DataFrameSchema({
    "mol_id": Column(str, Check.str_length(min_value=1)),
    "raw_smiles": Column(str, nullable=False),
    "canonical_smiles": Column(str, Check.str_length(min_value=1)),
    "isomeric_smiles": Column(str, nullable=True),
    "inchi_key": Column(str, nullable=True),
    "skeleton_key": Column(str, nullable=True),
    "tautomer_family_key": Column(str, nullable=True),
    "molecular_weight": Column(float, Check.greater_than(0), nullable=True),
    "exp_logp": Column(float, nullable=False),
    "split_logp": Column(str, Check.isin(["train", "val", "test", "unused"])),
})

# LC50 FHM 96h schema
LC50_SCHEMA = DataFrameSchema({
    "mol_id": Column(str, Check.str_length(min_value=1)),
    "raw_smiles": Column(str, nullable=False),
    "canonical_smiles": Column(str, Check.str_length(min_value=1)),
    "isomeric_smiles": Column(str, nullable=True),
    "inchi_key": Column(str, nullable=True),
    "skeleton_key": Column(str, nullable=True),
    "tautomer_family_key": Column(str, nullable=True),
    "molecular_weight": Column(float, Check.greater_than(0), nullable=True),
    "LC50_mol_L": Column(float, Check.greater_than(0), nullable=True),
    "pLC50": Column(float, nullable=False),
    "split_lc50": Column(str, Check.isin(["train", "val", "unused"])),
    "train_allowed": Column(bool, nullable=False),
    "qualifier": Column(str, nullable=True),
})

# Source domain schema (39 HPAH-relevant compounds)
SOURCE_DOMAIN_SCHEMA = DataFrameSchema({
    "mol_id": Column(str, Check.str_length(min_value=1)),
    "canonical_smiles": Column(str, Check.str_length(min_value=1)),
    "pLC50": Column(float, nullable=False),
})

# HPAH targets schema (57 compounds for inference)
HPAH_TARGETS_SCHEMA = DataFrameSchema({
    "hpah_id": Column(str, Check.str_length(min_value=1)),
    "canonical_smiles": Column(str, Check.str_length(min_value=1)),
})


def validate_logp_schema(df) -> None:
    """Validate LogP master DataFrame schema."""
    LOGP_SCHEMA.validate(df)


def validate_lc50_schema(df) -> None:
    """Validate LC50 DataFrame schema."""
    LC50_SCHEMA.validate(df)


def validate_source_domain_schema(df) -> None:
    """Validate source domain DataFrame schema."""
    SOURCE_DOMAIN_SCHEMA.validate(df)


def validate_hpah_targets_schema(df) -> None:
    """Validate HPAH targets DataFrame schema."""
    HPAH_TARGETS_SCHEMA.validate(df)
