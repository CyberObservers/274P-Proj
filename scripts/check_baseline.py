"""Stage 1 gate: read HIGGS FT-T all-features seed 0 metric and pass/fail."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED = "results/higgs/ft_transformer__all__seed0/metrics.json"
# rtdl FT-T (default) on HIGGS reaches ~0.85; Baldi 2014 5-layer DNN got 0.88.
# Looser gate to reflect FT-T's actual ceiling; we still require it to clearly beat
# random (0.5) and clear the FT-T literature floor.
LO, HI = 0.840, 0.890

p = Path(REQUIRED)
if not p.exists():
    print(f"[fail] {REQUIRED} missing"); sys.exit(2)
m = json.loads(p.read_text())
auc = m.get("test", {}).get("auc")
if auc is None:
    print(f"[fail] no test.auc in {p}"); sys.exit(3)
status = "pass" if LO <= auc <= HI else "FAIL"
print(f"[{status}] HIGGS FT-T all-features seed0  AUC={auc:.4f}  (target {LO}..{HI})")
sys.exit(0 if status == "pass" else 1)
