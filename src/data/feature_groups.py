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
    kinematic_kind: layout of `indices` for v3 NSI 4-momentum reconstruction.
        - "full":      (pT, eta, phi[, btag]) — leptons/jets
        - "transverse": (mag, phi)            — MET (no eta)
        - "engineered": opaque scalars, do NOT pass to NSI 4-momentum builder
    """
    name: str
    indices: List[int]
    type_id: int
    kinematic_kind: str = "full"


HIGGS_LOW_LEVEL_GROUPS: List[FeatureGroupConfig] = [
    FeatureGroupConfig("lepton", [0, 1, 2], type_id=0, kinematic_kind="full"),
    FeatureGroupConfig("met",    [3, 4],    type_id=1, kinematic_kind="transverse"),
    FeatureGroupConfig("jet1",   [5, 6, 7, 8],   type_id=2, kinematic_kind="full"),
    FeatureGroupConfig("jet2",   [9, 10, 11, 12], type_id=2, kinematic_kind="full"),
    FeatureGroupConfig("jet3",   [13, 14, 15, 16], type_id=2, kinematic_kind="full"),
    FeatureGroupConfig("jet4",   [17, 18, 19, 20], type_id=2, kinematic_kind="full"),
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
    FeatureGroupConfig("lepton1", [0, 1, 2], type_id=0, kinematic_kind="full"),
    FeatureGroupConfig("lepton2", [3, 4, 5], type_id=0, kinematic_kind="full"),
    FeatureGroupConfig("met",     [6, 7],    type_id=1, kinematic_kind="transverse"),
]

SUSY_LOW_LEVEL_INVARIANT_PAIRS: List[tuple] = [
    ("lepton1", "lepton2"),
]


@dataclass
class DatasetGroupSpec:
    groups: List[FeatureGroupConfig]
    invariant_pairs: List[tuple] = field(default_factory=list)
    feature_layout: str = "pT_eta_phi_btag"  # used by invariant computation


HIGGS_ALL_GROUPS: List[FeatureGroupConfig] = HIGGS_LOW_LEVEL_GROUPS + [
    # 7 engineered Lorentz invariants (m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb)
    FeatureGroupConfig("engineered", list(range(21, 28)), type_id=3, kinematic_kind="engineered"),
]
SUSY_ALL_GROUPS: List[FeatureGroupConfig] = SUSY_LOW_LEVEL_GROUPS + [
    # 10 engineered features (MET_rel, axial_MET, M_R, M_TR_2, R, MT2, S_R, M_Delta_R, dPhi_r_b, cos_theta_r1)
    FeatureGroupConfig("engineered", list(range(8, 18)), type_id=2, kinematic_kind="engineered"),
]


# Top Tagging: 8 subjets × 8 features = 64 features.  Each subjet is one group.
# All subjets share type_id=0 → weight tying = single MLP shared across 8 subjets.
# kinematic_kind="subjet" — first 4 indices are (E, px, py, pz), last 4 are
# substructure scalars (log_n_const, mass, width, log_pt).
TOP_TAGGING_SUBJET_GROUPS: List[FeatureGroupConfig] = [
    FeatureGroupConfig(f"sj{i}", list(range(8 * i, 8 * i + 8)),
                       type_id=0, kinematic_kind="subjet")
    for i in range(8)
]


REGISTRY: Dict[str, DatasetGroupSpec] = {
    "higgs:low_level":  DatasetGroupSpec(HIGGS_LOW_LEVEL_GROUPS, HIGGS_LOW_LEVEL_INVARIANT_PAIRS),
    "higgs:all":        DatasetGroupSpec(HIGGS_ALL_GROUPS,       HIGGS_LOW_LEVEL_INVARIANT_PAIRS),
    "susy:low_level":   DatasetGroupSpec(SUSY_LOW_LEVEL_GROUPS,  SUSY_LOW_LEVEL_INVARIANT_PAIRS),
    "susy:all":         DatasetGroupSpec(SUSY_ALL_GROUPS,        SUSY_LOW_LEVEL_INVARIANT_PAIRS),
    "top_tagging:all":  DatasetGroupSpec(TOP_TAGGING_SUBJET_GROUPS),
}


def get_groups(dataset: str, setup: str) -> DatasetGroupSpec | None:
    """Return group spec, or None if PIFT cannot inject physics priors for this combo
    (e.g. high-level setup where invariants are already computed)."""
    return REGISTRY.get(f"{dataset}:{setup}")
