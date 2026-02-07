# live_metrics.py
# Online (live) rolling-window metrics for pose angles + torso center
# Uses only past frames: each update() returns metrics for "what has happened so far".

from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Optional, Tuple, Any, List


@dataclass
class LiveMetrics:
    frame_index: int
    window_start: int
    window_end: int  # inclusive end for convenience

    primary: Optional[Tuple[str, float]]     # (key, ROM)
    secondary: Optional[Tuple[str, float]]   # (key, ROM)
    rom: Dict[str, float]

    symmetry: Dict[str, Optional[float]]     # 0..1 higher better
    phase_offset: Optional[float]            # 0..1 lower better

    smoothness: Optional[float]              # jerk proxy lower better
    torso_drift: Optional[float]             # normalized drift lower better

    view_badness: Optional[float]            # 0..1 higher = more foreshortened/rotated

    reps: int
    rep_event: Optional[dict]   # populated only when a rep is counted
    rep_rom: Optional[float]
    rep_key: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.frame_index,
            "window": {"start": self.window_start, "end": self.window_end},
            "primary": list(self.primary) if self.primary else None,
            "secondary": list(self.secondary) if self.secondary else None,
            "rom": self.rom,
            "symmetry": self.symmetry,
            "phase_offset": self.phase_offset,
            "smoothness": self.smoothness,
            "torso_drift": self.torso_drift,
            "view_badness": self.view_badness,
            "reps": self.reps,
            "rep_event": self.rep_event,
            "rep_rom": self.rep_rom,
            "rep_key": self.rep_key,
        }


class LiveWindowAnalyzer:
    """
    Rolling-window analyzer:
    - angles: dict[str,float] per frame (e.g. {"LEFT_ARM": 87, ...})
    - torso: (x,y) normalized per frame or None
    - view_segments: normalized lengths per frame (optional) to estimate rotation/foreshortening

    Call update(...) once per frame. Returns LiveMetrics or None until enough data.
    """

    def __init__(
        self,
        fps: int,
        window_seconds: float = 1.0,
        min_angle_samples: int = 5,
        symmetry_pairs: Optional[List[Tuple[str, str, str]]] = None,
        view_badness_threshold: float = 0.25,
    ):
        self.fps = max(1, int(fps))
        self.win = max(5, int(round(window_seconds * self.fps)))
        self.min_angle_samples = min_angle_samples
        self.view_badness_threshold = view_badness_threshold

        # --- rep counting state ---
        self.rep_count = 0
        self._active_key = None          # which angle we are counting reps from
        self._ema = None                 # smoothed angle value
        self._ema_alpha = 0.25           # smoothing strength (0..1), higher = less smooth
        self._phase = "search"           # "search" | "down" | "up"
        self._min_val = None
        self._max_val = None
        self._min_i = None
        self._max_i = None

        self._last_rep_frame = -10**9
        self._min_rep_frames = max(6, int(0.35 * self.fps))   # shortest plausible rep
        self._min_rom_deg = 18.0                              # ignore tiny movements
        self._hysteresis_deg = 6.0                             # how much reversal needed
        self._key_hold_frames = max(10, int(0.35 * self.fps))  # keep key stable a bit
        self._key_hold_left = 0

        # (label, left_key, right_key)
        self.symmetry_pairs = symmetry_pairs or [
            ("arm", "LEFT_ARM", "RIGHT_ARM"),
            ("leg", "LEFT_LEG", "RIGHT_LEG"),
            ("hip", "LEFT_HIP", "RIGHT_HIP"),
            ("shoulder", "LEFT_SHOULDER", "RIGHT_SHOULDER"),
        ]

        self._angles_q: Deque[Dict[str, float]] = deque(maxlen=self.win)
        self._torso_q: Deque[Optional[Tuple[float, float]]] = deque(maxlen=self.win)

        # For view_badness: store per-frame normalized segment lengths (optional)
        # keys like: "UPPER_ARM_L", "UPPER_ARM_R", "THIGH_L", etc.
        self._seglen_q: Deque[Optional[Dict[str, float]]] = deque(maxlen=self.win)
        self._seg_baseline: Dict[str, float] = {}  # running baseline for segments

        self.frame_index = -1

    def update(
        self,
        angles: Dict[str, float],
        torso_center_norm: Optional[Tuple[float, float]],
        seglens_norm: Optional[Dict[str, float]] = None,
    ) -> Optional[LiveMetrics]:
        """
        angles: dict of angle name -> degrees (float)
        torso_center_norm: (x/scale, y/scale) or None
        seglens_norm: optional dict of normalized segment lengths for view_badness
        """
        self.frame_index += 1

        # Store latest frame
        self._angles_q.append({k: float(v) for k, v in angles.items()})
        self._torso_q.append(torso_center_norm)
        self._seglen_q.append({k: float(v) for k, v in seglens_norm.items()} if seglens_norm else None)

        if len(self._angles_q) < self.min_angle_samples:
            return None

        # Compute metrics on current window (only history)
        window_start = max(0, self.frame_index - len(self._angles_q) + 1)
        window_end = self.frame_index

        rom = self._rom_current_window(min_samples=self.min_angle_samples)
        primary, secondary = self._pick_primary_secondary(rom)

        symmetry = {}
        for label, lk, rk in self.symmetry_pairs:
            symmetry[label] = self._symmetry_from_rom(rom, lk, rk)

        phase = None
        if primary and secondary:
            phase = self._phase_offset(primary[0], secondary[0])

        smooth = self._smoothness_jerk(primary[0]) if primary else None
        drift = self._torso_drift()

        view_badness = self._view_badness()
        rep_info = self._update_rep_counter(primary_key=primary[0] if primary else None, angles=angles)

        return LiveMetrics(
            frame_index=self.frame_index,
            window_start=window_start,
            window_end=window_end,
            primary=primary,
            secondary=secondary,
            rom=rom,
            symmetry=symmetry,
            phase_offset=phase,
            smoothness=smooth,
            torso_drift=drift,
            view_badness=view_badness,
            reps=self.rep_count,
            rep_event=rep_info.get("event"),
            rep_rom=rep_info.get("rom"),
            rep_key=rep_info.get("key"),
        )

    # ---------- core computations (window-based) ----------
    def _rom_current_window(self, min_samples: int) -> Dict[str, float]:
        vals_by_key: Dict[str, List[float]] = {}
        for d in self._angles_q:
            for k, v in d.items():
                vals_by_key.setdefault(k, []).append(v)

        rom: Dict[str, float] = {}
        for k, vals in vals_by_key.items():
            if len(vals) >= min_samples:
                rom[k] = max(vals) - min(vals)
        return rom

    @staticmethod
    def _pick_primary_secondary(rom: Dict[str, float]) -> Tuple[Optional[Tuple[str, float]], Optional[Tuple[str, float]]]:
        if not rom:
            return None, None
        items = sorted(rom.items(), key=lambda kv: kv[1], reverse=True)
        primary = items[0]
        secondary = items[1] if len(items) > 1 else None
        return primary, secondary

    @staticmethod
    def _symmetry_from_rom(rom: Dict[str, float], left_key: str, right_key: str) -> Optional[float]:
        if left_key not in rom or right_key not in rom:
            return None
        a, b = rom[left_key], rom[right_key]
        m = max(a, b)
        if m < 1e-9:
            return None
        return 1.0 - abs(a - b) / m  # 1 good, 0 bad

    def _phase_offset(self, primary_key: str, secondary_key: str) -> Optional[float]:
        # Find peak timestamps within the window using only present samples.
        idx = []
        p = []
        s = []
        # local indices 0..len(window)-1
        for i, d in enumerate(self._angles_q):
            if primary_key in d and secondary_key in d:
                idx.append(i)
                p.append(d[primary_key])
                s.append(d[secondary_key])

        if len(idx) < 6:
            return None

        p_peak = idx[p.index(max(p))]
        s_peak = idx[s.index(max(s))]
        dur = max(1, len(self._angles_q) - 1)
        return abs(p_peak - s_peak) / dur  # 0..1

    def _smoothness_jerk(self, key: str) -> Optional[float]:
        vals = []
        for d in self._angles_q:
            if key in d:
                vals.append(d[key])

        if len(vals) < 8:
            return None

        v = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
        if len(v) < 3:
            return None
        jerk = [abs(v[i] - v[i - 1]) for i in range(1, len(v))]
        return sum(jerk) / len(jerk) if jerk else None

    def _torso_drift(self) -> Optional[float]:
        pts = [p for p in self._torso_q if p is not None]
        if len(pts) < 5:
            return None
        x0, y0 = pts[0]
        dists = [((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 for (x, y) in pts]
        return max(dists) if dists else None

    # ---------- view / rotation quality (optional) ----------

    def _view_badness(self) -> Optional[float]:
        """
        Uses normalized segment lengths to detect foreshortening (rotation out of plane).
        Requires seglens_norm passed into update(). If not provided, returns None.
        """
        # Find the most recent seglen dict
        recent = None
        for d in reversed(self._seglen_q):
            if d is not None and len(d) > 0:
                recent = d
                break
        if recent is None:
            return None

        # Update baseline with slow EMA-like approach (robust enough)
        for k, v in recent.items():
            if v <= 0:
                continue
            if k not in self._seg_baseline:
                self._seg_baseline[k] = v
            else:
                # slow adapt so baseline doesn't chase quick rotations
                self._seg_baseline[k] = 0.98 * self._seg_baseline[k] + 0.02 * v

        # Foreshortening score per segment
        badness = 0.0
        for k, v in recent.items():
            b = self._seg_baseline.get(k)
            if not b or b <= 1e-9:
                continue
            # 0 good, 1 very foreshortened
            f = 1.0 - (v / b)
            if f > badness:
                badness = f

        # clamp 0..1
        if badness < 0.0:
            badness = 0.0
        if badness > 1.0:
            badness = 1.0
        return badness
    def _update_rep_counter(self, primary_key: Optional[str], angles: Dict[str, float]) -> Dict[str, Any]:
        """
        Generic rep counter on a single angle signal.
        Detects cycles using hysteresis and ROM thresholds.
        Returns {"key":..., "rom":..., "event": {...} or None}
        """
        # --- choose / hold active key ---
        if self._active_key is None:
            self._active_key = primary_key

        # keep key stable for a bit to avoid flapping
        if primary_key and primary_key != self._active_key:
            if self._key_hold_left <= 0:
                # allow switch only if current active key isn't present
                if self._active_key not in angles:
                    self._active_key = primary_key
                    self._key_hold_left = self._key_hold_frames
            else:
                self._key_hold_left -= 1
        else:
            if self._key_hold_left > 0:
                self._key_hold_left -= 1

        key = self._active_key
        if not key or key not in angles:
            return {"key": key, "rom": None, "event": None}

        x = float(angles[key])

        # --- EMA smoothing ---
        if self._ema is None:
            self._ema = x
        else:
            a = self._ema_alpha
            self._ema = a * x + (1 - a) * self._ema

        v = self._ema
        i = self.frame_index

        # --- init extrema ---
        if self._min_val is None:
            self._min_val = v; self._min_i = i
        if self._max_val is None:
            self._max_val = v; self._max_i = i

        event = None

        # --- state machine ---
        if self._phase == "search":
            # Start tracking by assuming we're going "down" toward a min
            self._phase = "down"
            self._min_val = v; self._min_i = i
            self._max_val = v; self._max_i = i

        elif self._phase == "down":
            # Update min while descending
            if v < self._min_val:
                self._min_val = v; self._min_i = i

            # If we've risen enough from the min, we reversed -> now going up
            if v > self._min_val + self._hysteresis_deg:
                self._phase = "up"
                self._max_val = v; self._max_i = i

        elif self._phase == "up":
            # Update max while ascending
            if v > self._max_val:
                self._max_val = v; self._max_i = i

            # If we've dropped enough from the max, we reversed -> potential rep complete
            if v < self._max_val - self._hysteresis_deg:
                rom = float(self._max_val - self._min_val)
                frames_since_last = i - self._last_rep_frame
                rep_duration = (self._max_i - self._min_i) if (self._max_i is not None and self._min_i is not None) else None

                ok_duration = (frames_since_last >= self._min_rep_frames)
                ok_rom = (rom >= self._min_rom_deg)

                if ok_duration and ok_rom:
                    self.rep_count += 1
                    self._last_rep_frame = i
                    event = {
                        "rep": self.rep_count,
                        "key": key,
                        "rom": rom,
                        "min": self._min_val,
                        "max": self._max_val,
                        "min_frame": self._min_i,
                        "max_frame": self._max_i,
                        "counted_at": i,
                    }

                # reset for next rep: start searching for next minimum
                self._phase = "down"
                self._min_val = v; self._min_i = i
                self._max_val = v; self._max_i = i

        return {"key": key, "rom": (self._max_val - self._min_val) if (self._max_val is not None and self._min_val is not None) else None,
                "event": event}
