"""Physics feature groups + symmetry types for HEP datasets.

Exact column layout from Baldi 2014 (HIGGS, SUSY) and Baldi 2016 (HEPMASS).
Indices are 0-based and refer to the *feature* columns (after stripping the label).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class FeatureGroupConfig:
    """One group = one physics object (lepton / jet_i / MET / ...).

    type_id: same id => permutation-equivariant (e.g. all 4 jets share type_id=1).
    invariant_inputs: for invariant_augmentation. Each entry is a 4-momentum
        index list (pT, eta, phi, m or b_tag) — or a pair of objects to compute
        ΔR / m_inv between.
    """
    name: str
    indices: List[int]
    type_id: int


HIGGS_LOW_LEVEL_GROUPS: List[FeatureGroupConfig] = [
    FeatureGroupConfig("lepton", [0, 1, 2], type_id=0),
    FeatureGroupConfig("met",    [3, 4],    type_id=1),
    FeatureGroupConfig("jet1",   [5, 6, 7, 8],   type_id=2),
    FeatureGroupConfig("jet2",   [9, 10, 11, 12], type_id=2),
    FeatureGroupConfig("jet3",   [13, 14, 15, 16], type_id=2),
    FeatureGroupConfig("jet4",   [17, 18, 19, 20], type_id=2),
]

# (a, b) pairs to compute invariant mass / ΔR. Each item is two group names whose
# (pT, eta, phi) we will combine.
HIGGS_LOW_LEVEL_INVARIANT_PAIRS: List[tuple] = [
    ("jet1", "jet2"),
    ("jet1", "jet3"),
    ("jet1", "jet4"),
    ("jet2", "jet3"),
    ("jet2", "jet4"),
    ("jet3", "jet4"),
    ("lepton", "jet1"),
    ("lepton", "jet2"),
]

SUSY_LOW_LEVEL_GROUPS: List[FeatureGroupConfig] = [
    FeatureGroupConfig("lepton1", [0, 1, 2], type_id=0),
    FeatureGroupConfig("lepton2", [3, 4, 5], type_id=0),
    FeatureGroupConfig("met",     [6, 7],    type_id=1),
]

SUSY_LOW_LEVEL_INVARIANT_PAIRS: List[tuple] = [
    ("lepton1", "lepton2"),
]


@dataclass
class DatasetGroupSpec:
    groups: List[FeatureGroupConfig]
    invariant_pairs: List[tuple] = field(default_factory=list)
    feature_layout: str = "pT_eta_phi_btag"  # used by invariant computation


REGISTRY: Dict[str, DatasetGroupSpec] = {
    "higgs:low_level":  DatasetGroupSpec(HIGGS_LOW_LEVEL_GROUPS, HIGGS_LOW_LEVEL_INVARIANT_PAIRS),
    "susy:low_level":   DatasetGroupSpec(SUSY_LOW_LEVEL_GROUPS,  SUSY_LOW_LEVEL_INVARIANT_PAIRS),
}


def get_groups(dataset: str, setup: str) -> DatasetGroupSpec | None:
    """Return group spec, or None if PIFT cannot inject physics priors for this combo
    (e.g. high-level setup where invariants are already computed)."""
    return REGISTRY.get(f"{dataset}:{setup}")
