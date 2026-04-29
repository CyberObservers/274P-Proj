"""Anti-kT / C/A subjet clustering wrapper around fastjet.

Used by `preprocess_top_tagging` to convert per-jet 200 constituent 4-momenta
into K=8 subjet tokens, each with (E, px, py, pz, n_const, mass, width, log_pt).

The PIFT-Subjet variant treats each subjet as one group token (kinematic_kind="subjet"),
similar to how HIGGS treats each lepton/jet as a group.  The difference: HIGGS
groups are *hand-spec'd*, Top Tagging subjets are *physics-algorithm-generated*.
"""
from __future__ import annotations

from typing import List

import awkward as ak
import fastjet
import numpy as np


def cluster_subjets_batch(
    constituents: np.ndarray,
    n_subjets: int = 8,
    R: float = 1.5,
    algorithm: str = "ca",
) -> np.ndarray:
    """Cluster a batch of jets into K subjets using fastjet.

    Parameters
    ----------
    constituents : (N_events, N_max=200, 4) float32
        Per-jet constituent 4-momenta in (E, px, py, pz) order.
        Padded with zeros (E=0) for jets with <200 valid constituents.
    n_subjets : int
        Number of exclusive subjets per jet (target K).
    R : float
        Jet radius — for C/A on Top Tagging, use 1.5 (large enough to encompass
        the whole top jet which itself has R~0.8).
    algorithm : str
        "ca" (Cambridge-Aachen, p=0) is recommended — natively supports
        exclusive_jets.  "kt" also works.  "antikt" gives a fastjet warning.

    Returns
    -------
    subjet_features : (N_events, K, 8) float32
        Per-subjet 8-D features:
            [0..3] (E, px, py, pz)  — total 4-momentum of subjet
            [4]    log(n_constituents + 1)   — log-count
            [5]    subjet mass = sqrt(E² - |p|²)  (clamped at 0 for numerical)
            [6]    width = E-weighted RMS of ΔR from subjet axis
            [7]    log(pT + 1)
        Empty subjets (when a jet has <K constituents) are zero-padded.
    """
    if algorithm == "ca":
        jet_alg = fastjet.cambridge_algorithm
    elif algorithm == "kt":
        jet_alg = fastjet.kt_algorithm
    elif algorithm == "antikt":
        jet_alg = fastjet.antikt_algorithm
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")
    jetdef = fastjet.JetDefinition(jet_alg, R)

    N, N_max, _ = constituents.shape
    K = n_subjets
    out = np.zeros((N, K, 8), dtype=np.float32)

    # Bucket events by valid-count >= K and < K
    E_all = constituents[..., 0]
    valid = E_all > 1e-6
    n_valid = valid.sum(axis=1)
    big_idx = np.where(n_valid >= K)[0]
    small_idx = np.where((n_valid > 0) & (n_valid < K))[0]

    # ----- big events (>= K constituents): batch-cluster via awkward + fastjet -----
    events_list = []
    for i in big_idx:
        m = valid[i]
        events_list.append([
            {"E":  float(constituents[i, j, 0]),
             "px": float(constituents[i, j, 1]),
             "py": float(constituents[i, j, 2]),
             "pz": float(constituents[i, j, 3])}
            for j in np.where(m)[0]
        ])
    if len(events_list) > 0:
        events = ak.Array(events_list)
        cs = fastjet.ClusterSequence(events, jetdef)
        sub_jets = cs.exclusive_jets(n_jets=K)
        sub_const = cs.exclusive_jets_constituents(K)
        for ii, i in enumerate(big_idx):
            sj_E_i  = sub_jets["E"][ii].tolist()
            sj_px_i = sub_jets["px"][ii].tolist()
            sj_py_i = sub_jets["py"][ii].tolist()
            sj_pz_i = sub_jets["pz"][ii].tolist()
            for k in range(min(len(sj_E_i), K)):
                _fill_subjet_features(out, i, k, sj_E_i[k], sj_px_i[k], sj_py_i[k],
                                      sj_pz_i[k], sub_const[ii][k])
        del cs, events

    # ----- small events (<K constituents): each constituent IS its own subjet -----
    for i in small_idx:
        m = valid[i]
        idxs = np.where(m)[0]
        for k, j in enumerate(idxs[:K]):
            E, px, py, pz = (float(constituents[i, j, 0]), float(constituents[i, j, 1]),
                             float(constituents[i, j, 2]), float(constituents[i, j, 3]))
            _fill_subjet_features_simple(out, i, k, E, px, py, pz)
    return out


def _fill_subjet_features(out, i, k, E, px, py, pz, cs_const):
    out[i, k, 0] = E
    out[i, k, 1] = px
    out[i, k, 2] = py
    out[i, k, 3] = pz
    n_const = len(cs_const)
    out[i, k, 4] = np.log(n_const + 1)
    mass_sq = max(E**2 - (px**2 + py**2 + pz**2), 0.0)
    out[i, k, 5] = float(np.sqrt(mass_sq))
    pT_sj = np.sqrt(px**2 + py**2)
    if n_const >= 2 and pT_sj > 1e-6:
        eta_sj = 0.5 * np.log((E + pz) / max(E - pz, 1e-9))
        phi_sj = np.arctan2(py, px)
        dR_sum, w_sum = 0.0, 0.0
        for c in cs_const.tolist():
            cE, cpx, cpy, cpz = c["E"], c["px"], c["py"], c["pz"]
            cpT = np.sqrt(cpx**2 + cpy**2)
            if cpT < 1e-6:
                continue
            c_eta = 0.5 * np.log((cE + cpz) / max(cE - cpz, 1e-9))
            c_phi = np.arctan2(cpy, cpx)
            dphi = (c_phi - phi_sj + np.pi) % (2 * np.pi) - np.pi
            dR2 = (c_eta - eta_sj)**2 + dphi**2
            dR_sum += cE * dR2
            w_sum += cE
        out[i, k, 6] = float(np.sqrt(dR_sum / max(w_sum, 1e-9)))
    out[i, k, 7] = float(np.log(pT_sj + 1))


def _fill_subjet_features_simple(out, i, k, E, px, py, pz):
    """Single-constituent subjet (small-event fallback): mass=0, width=0, n=1."""
    out[i, k, 0] = E
    out[i, k, 1] = px
    out[i, k, 2] = py
    out[i, k, 3] = pz
    out[i, k, 4] = np.log(2)   # log(1 + 1)
    out[i, k, 5] = 0.0
    out[i, k, 6] = 0.0
    pT_sj = np.sqrt(px**2 + py**2)
    out[i, k, 7] = float(np.log(pT_sj + 1))
