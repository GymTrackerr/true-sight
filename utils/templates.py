# utils/templates.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import json
import math

@dataclass
class JointTemplate:
    min: float
    max: float
    delta: float      # expected change (your "diff")
    weight: float     # importance weight
    tol: float        # tolerance in degrees

@dataclass
class ExerciseTemplate:
    name: str
    joints: Dict[str, JointTemplate]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "joints": {k: asdict(v) for k, v in self.joints.items()}}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExerciseTemplate":
        return ExerciseTemplate(
            name=d["name"],
            joints={k: JointTemplate(**v) for k, v in d["joints"].items()}
        )

def _soft_weights(deltas: Dict[str, float], tau: float = 35.0, floor: float = 0.15) -> Dict[str, float]:
    keys = list(deltas.keys())
    xs = [max(0.0, float(deltas[k])) for k in keys]
    exps = [math.exp(x / tau) for x in xs]
    s = sum(exps) or 1.0
    ws = [e / s for e in exps]
    ws = [max(floor, w) for w in ws]
    s2 = sum(ws) or 1.0
    ws = [w / s2 for w in ws]
    return {keys[i]: float(ws[i]) for i in range(len(keys))}

def build_template_from_export(
    exercise_name: str,
    rows: List[List[Any]],
    *,
    min_delta: float = 12.0,
    keep_top_k: int = 6,
    tau: float = 35.0,
    floor: float = 0.15,
) -> ExerciseTemplate:
    """
    rows: [name, index, diff, angle1, angle2]
    Produces a reusable ExerciseTemplate.
    """
    items = []
    for name, _idx, diff, a1, a2 in rows:
        a1 = float(a1); a2 = float(a2)
        delta = abs(a2 - a1)  # same as diff, but consistent
        if delta < min_delta:
            continue
        mn = min(a1, a2)
        mx = max(a1, a2)
        items.append((name, mn, mx, delta))

    items.sort(key=lambda x: x[3], reverse=True)
    items = items[:keep_top_k]

    deltas = {name: delta for (name, _mn, _mx, delta) in items}
    weights = _soft_weights(deltas, tau=tau, floor=floor)

    joints: Dict[str, JointTemplate] = {}
    for name, mn, mx, delta in items:
        tol = max(8.0, 0.12 * delta)  # degrees
        joints[name] = JointTemplate(min=mn, max=mx, delta=delta, weight=weights[name], tol=tol)

    return ExerciseTemplate(name=exercise_name, joints=joints)

def save_template(path: str, tpl: ExerciseTemplate) -> None:
    with open(path, "w") as f:
        json.dump(tpl.to_dict(), f, indent=2)

def load_template(path: str) -> ExerciseTemplate:
    with open(path, "r") as f:
        return ExerciseTemplate.from_dict(json.load(f))
