# live_metrics.py
# Online (live) rolling-window metrics for pose angles + torso center
# Uses only past frames: each update() returns metrics for "what has happened so far".

from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Optional, Tuple, Any, List
from utils.templates import ExerciseTemplate


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

    rep_progress: Optional[float]            # 0..100 percentage through current rep

    reps: int
    rep_event: Optional[dict]   # populated only when a rep is counted
    rep_rom: Optional[float]
    rep_key: Optional[str]
    rep_score: Optional[dict]   # populated when rep is scored against template

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
            "rep_progress": self.rep_progress,
            "reps": self.reps,
            "rep_event": self.rep_event,
            "rep_rom": self.rep_rom,
            "rep_key": self.rep_key,
            "rep_score": self.rep_score,
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
        template: Optional[ExerciseTemplate] = None,
    ):
        self.fps = max(1, int(fps))
        self.win = max(5, int(round(window_seconds * self.fps)))
        self.template = template  # Optional exercise template for scoring reps
        self.min_angle_samples = min_angle_samples
        self.view_badness_threshold = view_badness_threshold

        # --- sticky primary/secondary selection ---
        self._prim_key = None
        self._sec_key = None
        self._switch_margin = 8.0          # degrees of ROM advantage required to switch
        self._switch_confirm_frames = max(6, int(0.5 * self.fps))  # ~0.5s at current fps
        self._switch_counter = 0
        self._candidate_key = None

        # --- rep counting state ---
        self.rep_count = 0
        self._active_key = None          # which angle we are counting reps from
        self._rep_signal_key = None      # locked signal for rep counting (left/right averaged)
        self._rep_signal_lock_frames = 0  # frames remaining to keep signal locked
        self._rep_signal_lock_duration = max(150, int(5.0 * self.fps))  # lock for 5 seconds
        self._ema = None                 # smoothed angle value
        self._ema_alpha = 0.25           # smoothing strength (0..1), higher = less smooth
        self._phase = "search"           # "search" | "down" | "up"
        self._min_val = None
        self._max_val = None
        self._min_i = None
        self._max_i = None

        self._last_rep_frame = -10**9
        self._min_rep_frames = max(6, int(0.35 * self.fps))   # shortest plausible rep
        self._min_rom_deg = 12.0                              # base min ROM (was 18.0, lowered for 15 fps)
        self._hysteresis_deg = 6.0                             # how much reversal needed
        self._key_hold_frames = max(10, int(0.35 * self.fps))  # keep key stable a bit
        self._key_hold_left = 0
        
        # Rep frame tracking for template scoring
        self._rep_frames: Deque[Dict[str, float]] = deque(maxlen=500)  # store frames for current rep

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
        primary, secondary = self._stable_primary_secondary(rom)

        symmetry = {}
        for label, lk, rk in self.symmetry_pairs:
            symmetry[label] = self._symmetry_from_rom(rom, lk, rk)

        phase = None
        if primary and secondary:
            phase = self._phase_offset(primary[0], secondary[0])

        smooth = self._smoothness_jerk(primary[0]) if primary else None
        drift = self._torso_drift()

        view_badness = self._view_badness()
        rep_info = self._update_rep_counter(primary_key=primary[0] if primary else None, angles=angles, rom=rom, view_badness=view_badness)
        
        # Calculate rep progress
        rep_progress = self._calculate_rep_progress(angles)
        
        # Track frames for rep scoring
        self._rep_frames.append({k: float(v) for k, v in angles.items()})
        
        # Score rep against template if available
        rep_score = None
        if rep_info.get("event") is not None and self.template is not None:
            # A rep was just completed, score it
            rep_score = score_rep_against_template(self.template, list(self._rep_frames))
            self._rep_frames.clear()  # Reset for next rep
            print(f"\n🎯 REP #{rep_info['event']['rep']} SCORED:")
            print(f"   Overall Quality: {rep_score['overall_score']:.0%}")
            print(f"   Matches Template: {'✓' if rep_score['matches_template'] else '✗'}")
            for joint, scores in rep_score['joint_scores'].items():
                status = "✓" if (scores['is_within_range'] and scores['is_within_tolerance']) else "✗"
                print(f"   {joint}: {scores['score']:.0%} {status}")

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
            rep_progress=rep_progress,
            reps=self.rep_count,
            rep_event=rep_info.get("event"),
            rep_rom=rep_info.get("rom"),
            rep_key=rep_info.get("key"),
            rep_score=rep_score,
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

    def _stable_primary_secondary(self, rom: dict[str, float]) -> Tuple[Optional[Tuple[str, float]], Optional[Tuple[str, float]]]:
        """
        Sticky primary/secondary selection: only switch if new candidate wins by margin
        and stays consistent for N frames. Prevents flapping due to jitter.
        """
        if not rom:
            return None, None

        ranked = sorted(rom.items(), key=lambda kv: kv[1], reverse=True)
        best_key, best_rom = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None

        # Initialize on first call
        if self._prim_key is None:
            self._prim_key = best_key
            self._sec_key = second[0] if second else None
            return (self._prim_key, rom[self._prim_key]), (self._sec_key, rom[self._sec_key]) if self._sec_key else None

        # If current primary missing, swap immediately
        if self._prim_key not in rom:
            self._prim_key = best_key
            self._sec_key = second[0] if second else None
            self._switch_counter = 0
            self._candidate_key = None
            return (self._prim_key, rom[self._prim_key]), (self._sec_key, rom[self._sec_key]) if self._sec_key else None

        cur_rom = rom[self._prim_key]

        # Only consider switching if new best clearly better (margin threshold)
        if best_key != self._prim_key and (best_rom - cur_rom) >= self._switch_margin:
            if self._candidate_key != best_key:
                self._candidate_key = best_key
                self._switch_counter = 1
            else:
                self._switch_counter += 1

            if self._switch_counter >= self._switch_confirm_frames:
                self._prim_key = best_key
                self._sec_key = second[0] if second else None
                self._switch_counter = 0
                self._candidate_key = None
        else:
            self._switch_counter = 0
            self._candidate_key = None

        # Recompute secondary based on ranked list excluding primary
        sec_key = None
        for k, _ in ranked:
            if k != self._prim_key:
                sec_key = k
                break
        self._sec_key = sec_key

        primary = (self._prim_key, rom.get(self._prim_key, 0.0))
        secondary = (self._sec_key, rom.get(self._sec_key, 0.0)) if self._sec_key else None
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

    def _get_dominant_joint_pair(self) -> Optional[str]:
        """
        Identify which joint pair (arm, leg, hip, shoulder) has highest ROM.
        Returns the label name, or None if insufficient data.
        """
        rom = self._rom_current_window(min_samples=self.min_angle_samples)
        if not rom:
            return None
        
        pair_roms = {}
        for label, left_key, right_key in self.symmetry_pairs:
            left_rom = rom.get(left_key, 0.0)
            right_rom = rom.get(right_key, 0.0)
            pair_roms[label] = max(left_rom, right_rom)  # take the larger of the pair
        
        if not pair_roms:
            return None
        
        return max(pair_roms, key=pair_roms.get)

    def _get_rep_signal_averaged(self, joint_label: str, angles: Dict[str, float]) -> Optional[float]:
        """
        Get averaged left/right signal for a joint (e.g., "leg" -> (LEFT_LEG + RIGHT_LEG)/2).
        Reduces jitter from single joint tracking.
        """
        # Map label to keys
        label_to_keys = {
            "arm": ("LEFT_ARM", "RIGHT_ARM"),
            "leg": ("LEFT_LEG", "RIGHT_LEG"),
            "hip": ("LEFT_HIP", "RIGHT_HIP"),
            "shoulder": ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
        }
        
        if joint_label not in label_to_keys:
            return None
        
        left_key, right_key = label_to_keys[joint_label]
        left_val = angles.get(left_key)
        right_val = angles.get(right_key)
        
        if left_val is None or right_val is None:
            return None
        
        return (float(left_val) + float(right_val)) / 2.0

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
    def _update_rep_counter(self, primary_key: Optional[str], angles: Dict[str, float], rom: Dict[str, float], view_badness: Optional[float]) -> Dict[str, Any]:
        """
        Improved rep counter with:
        - Locked rep signal (doesn't change mid-set)
        - Left/right averaging for smoother signal
        - Dynamic ROM threshold based on recent window ROM
        - View badness gating (don't count if quality too low)
        - Velocity-based peak/trough detection
        """
        # --- gate: skip rep counting if view quality too bad ---
        if view_badness is not None and view_badness > self.view_badness_threshold:
            return {"key": self._rep_signal_key, "rom": None, "event": None}

        # --- choose/lock rep signal (only once per ~5 second window) ---
        if self._rep_signal_key is None and primary_key is not None:
            # On first call, pick dominant joint pair and lock it
            dominant_joint = self._get_dominant_joint_pair()
            if dominant_joint is not None:
                # Check if we can get the averaged signal
                avg_signal = self._get_rep_signal_averaged(dominant_joint, angles)
                if avg_signal is not None:
                    self._rep_signal_key = dominant_joint  # label like "leg", "arm", etc.
                    self._rep_signal_lock_frames = self._rep_signal_lock_duration
                else:
                    # Fallback to primary key if averaging fails
                    self._rep_signal_key = primary_key
                    self._rep_signal_lock_frames = self._rep_signal_lock_duration

        # Decrement lock timer
        if self._rep_signal_lock_frames > 0:
            self._rep_signal_lock_frames -= 1

        # If lock expired and primary changed substantially, allow re-selection
        if self._rep_signal_lock_frames <= 0:
            self._rep_signal_key = None

        # Get the actual signal value
        signal_value = None
        if self._rep_signal_key is not None:
            # Try averaged signal first
            if self._rep_signal_key in ["arm", "leg", "hip", "shoulder"]:
                signal_value = self._get_rep_signal_averaged(self._rep_signal_key, angles)
            # Fallback to direct angle key
            if signal_value is None and self._rep_signal_key in angles:
                signal_value = angles.get(self._rep_signal_key)

        if signal_value is None:
            return {"key": self._rep_signal_key, "rom": None, "event": None}

        x = float(signal_value)

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

        # --- dynamic ROM threshold based on window ROM ---
        window_rom_value = max(rom.values()) if rom else 0.0
        dynamic_min_rom = max(self._min_rom_deg, 0.35 * window_rom_value)

        # --- state machine ---
        if self._phase == "search":
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
                rom_value = float(self._max_val - self._min_val)
                frames_since_last = i - self._last_rep_frame
                rep_duration = (self._max_i - self._min_i) if (self._max_i is not None and self._min_i is not None) else None

                ok_duration = (frames_since_last >= self._min_rep_frames)
                ok_rom = (rom_value >= dynamic_min_rom)

                if ok_duration and ok_rom:
                    self.rep_count += 1
                    self._last_rep_frame = i
                    event = {
                        "rep": self.rep_count,
                        "key": self._rep_signal_key,
                        "rom": rom_value,
                        "min": self._min_val,
                        "max": self._max_val,
                        "min_frame": self._min_i,
                        "max_frame": self._max_i,
                        "counted_at": i,
                    }

                # reset for next rep
                self._phase = "down"
                self._min_val = v; self._min_i = i
                self._max_val = v; self._max_i = i

        return {"key": self._rep_signal_key, "rom": (self._max_val - self._min_val) if (self._max_val is not None and self._min_val is not None) else None,
                "event": event}

    def _calculate_rep_progress(self, angles: Dict[str, float]) -> Optional[float]:
        """
        Calculate percentage progress through current rep based on template min/max.
        Returns 0-100 percentage.
        """
        if not self.template or not angles:
            return None
        
        try:
            joints_info = self.template.joints
            if not joints_info:
                return None
            
            progress_scores = []
            
            for joint_name, joint_config in joints_info.items():
                if joint_name in angles:
                    current_angle = angles[joint_name]
                    min_angle = float(joint_config.min)
                    max_angle = float(joint_config.max)
                    weight = float(joint_config.weight)
                    
                    # Calculate progress as percentage within the range
                    if current_angle < min_angle:
                        progress = 0
                    elif current_angle > max_angle:
                        progress = 100
                    else:
                        range_size = max_angle - min_angle
                        current_progress = current_angle - min_angle
                        progress = (current_progress / range_size) * 100 if range_size > 0 else 0
                    
                    # Weight by joint importance
                    progress_scores.append((progress, weight))
            
            if progress_scores:
                # Calculate weighted average
                total_weighted = sum(p * w for p, w in progress_scores)
                total_weight = sum(w for _, w in progress_scores)
                return total_weighted / total_weight if total_weight > 0 else None
        except Exception as e:
            print(f"Error calculating rep progress: {e}")
        
        return None


def score_rep_against_template(
    template: ExerciseTemplate,
    rep_angle_frames: List[Dict[str, float]],
) -> Dict[str, Any]:
    """
    Score how well a rep (sequence of angle frames) matches an exercise template.
    
    Args:
        template: ExerciseTemplate with joint ranges and tolerances
        rep_angle_frames: List of angle dicts from start to end of rep (one per frame)
    
    Returns:
        {
            "overall_score": 0.0-1.0,  # weighted average of joint scores
            "joint_scores": {
                "JOINT_NAME": {
                    "score": 0.0-1.0,
                    "rom": actual_rom,
                    "expected_rom": template_delta,
                    "is_within_range": bool,
                    "is_within_tolerance": bool,
                }
            },
            "matches_template": bool,  # all joints within acceptable range
            "num_frames": int,
        }
    """
    if not rep_angle_frames:
        return {"overall_score": 0.0, "joint_scores": {}, "matches_template": False, "num_frames": 0}
    
    # Collect all angle values for each joint across the rep
    joint_values: Dict[str, List[float]] = {}
    for frame in rep_angle_frames:
        for joint_name, angle in frame.items():
            if joint_name not in joint_values:
                joint_values[joint_name] = []
            joint_values[joint_name].append(float(angle))
    
    # Score each joint
    joint_scores = {}
    weighted_scores = []
    
    for joint_name, template_info in template.joints.items():
        if joint_name not in joint_values:
            # Joint missing from rep data
            joint_scores[joint_name] = {
                "score": 0.0,
                "rom": None,
                "expected_rom": float(template_info.delta),
                "is_within_range": False,
                "is_within_tolerance": False,
            }
            continue
        
        values = joint_values[joint_name]
        actual_min = min(values)
        actual_max = max(values)
        actual_rom = actual_max - actual_min
        
        template_min = float(template_info.min)
        template_max = float(template_info.max)
        template_rom = float(template_info.delta)
        tolerance = float(template_info.tol)
        weight = float(template_info.weight)
        
        # Check if range is within template bounds
        is_within_range = (actual_min >= template_min and actual_max <= template_max)
        
        # Check if ROM is within tolerance of expected
        rom_diff = abs(actual_rom - template_rom)
        is_within_tolerance = (rom_diff <= tolerance)
        
        # Score: combination of range match and ROM match
        # ROM match: how close the actual ROM is to expected (0..1)
        if template_rom > 1e-6:
            rom_score = max(0.0, 1.0 - (rom_diff / template_rom))
        else:
            rom_score = 1.0 if abs(actual_rom) < 1e-6 else 0.0
        
        # Range match: penalize if outside template bounds
        if is_within_range:
            range_score = 1.0
        else:
            # Penalize proportionally to how far outside
            out_of_bounds = 0.0
            if actual_min < template_min:
                out_of_bounds += (template_min - actual_min)
            if actual_max > template_max:
                out_of_bounds += (actual_max - template_max)
            template_span = template_max - template_min
            if template_span > 1e-6:
                range_score = max(0.0, 1.0 - (out_of_bounds / template_span))
            else:
                range_score = 0.5
        
        # Combined score (favor range adherence slightly)
        joint_score = 0.6 * range_score + 0.4 * rom_score
        joint_score = max(0.0, min(1.0, joint_score))
        
        joint_scores[joint_name] = {
            "score": joint_score,
            "rom": actual_rom,
            "expected_rom": template_rom,
            "is_within_range": is_within_range,
            "is_within_tolerance": is_within_tolerance,
        }
        
        weighted_scores.append((joint_score, weight))
    
    # Calculate weighted overall score
    if weighted_scores:
        total_weight = sum(w for _, w in weighted_scores)
        if total_weight > 1e-9:
            overall_score = sum(s * w for s, w in weighted_scores) / total_weight
        else:
            overall_score = sum(s for s, _ in weighted_scores) / len(weighted_scores)
    else:
        overall_score = 0.0
    
    overall_score = max(0.0, min(1.0, overall_score))
    
    # Rep matches template if all joints are within tolerance and range
    all_within = all(
        js["is_within_tolerance"] and js["is_within_range"]
        for js in joint_scores.values()
        if js["rom"] is not None
    )
    
    return {
        "overall_score": overall_score,
        "joint_scores": joint_scores,
        "matches_template": all_within,
        "num_frames": len(rep_angle_frames),
    }
