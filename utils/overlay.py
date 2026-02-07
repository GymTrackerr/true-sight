import cv2
from utils.joints import Joints

# simple skeleton edges
EDGES = [
    (Joints.LEFT_SHOULDER, Joints.RIGHT_SHOULDER),
    (Joints.LEFT_HIP, Joints.RIGHT_HIP),

    (Joints.LEFT_SHOULDER, Joints.LEFT_ELBOW),
    (Joints.LEFT_ELBOW, Joints.LEFT_WRIST),

    (Joints.RIGHT_SHOULDER, Joints.RIGHT_ELBOW),
    (Joints.RIGHT_ELBOW, Joints.RIGHT_WRIST),

    (Joints.LEFT_HIP, Joints.LEFT_KNEE),
    (Joints.LEFT_KNEE, Joints.LEFT_ANKLE),

    (Joints.RIGHT_HIP, Joints.RIGHT_KNEE),
    (Joints.RIGHT_KNEE, Joints.RIGHT_ANKLE),

    (Joints.LEFT_SHOULDER, Joints.LEFT_HIP),
    (Joints.RIGHT_SHOULDER, Joints.RIGHT_HIP),
]

def draw_pose_and_metrics(frame_bgr, frame_obj, metrics):
    """
    frame_obj: your frame_save for person 0
    metrics: m.to_dict() or None
    """
    out = frame_bgr

    # 1) draw joints
    if frame_obj and frame_obj.has_detections():
        # circles
        for j in Joints:
            p = frame_obj.get_joint_xy(j)
            if p is None:
                continue
            cv2.circle(out, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)

        # edges
        for a, b in EDGES:
            pa = frame_obj.get_joint_xy(a)
            pb = frame_obj.get_joint_xy(b)
            if pa is None or pb is None:
                continue
            cv2.line(out, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 255, 0), 2)

    # 2) draw text overlay
    if metrics:
        y = 24
        def put(line):
            nonlocal y
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            y += 26

        reps = metrics.get("reps")
        primary = metrics.get("primary")
        smooth = metrics.get("smoothness")
        drift = metrics.get("torso_drift")
        view = metrics.get("view_badness")

        put(f"reps: {reps if reps is not None else '-'}")
        if primary:
            put(f"primary: {primary[0]} ROM={primary[1]:.1f}")
        if smooth is not None:
            put(f"smoothness: {smooth:.2f}")
        if drift is not None:
            put(f"torso drift: {drift:.3f}")
        if view is not None:
            put(f"view badness: {view:.2f}")

        # flash when a rep is counted (if you implemented rep_event)
        rep_event = metrics.get("rep_event")
        if rep_event:
            cv2.putText(out, f"+1 rep ({rep_event['rep']})", (10, y+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3, cv2.LINE_AA)

    return out


def draw_rep_score(frame_bgr, rep_score):
    """
    Draw the previous rep's score on the right side of the frame.
    
    rep_score: dict from score_rep_against_template() or None
    """
    if not rep_score:
        return frame_bgr
    
    out = frame_bgr
    height, width = out.shape[:2]
    
    # Right side panel: start at 80% of width
    panel_x = int(width * 0.75)
    panel_y = 20
    line_height = 28
    
    # Draw semi-transparent background for readability
    overlay = out.copy()
    cv2.rectangle(overlay, (panel_x - 10, 0), (width, height), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.3, out, 0.7, 0)
    
    # Title
    cv2.putText(out, "REP SCORE", (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    
    y = panel_y + line_height
    
    # Overall score with color coding
    overall = rep_score.get("overall_score", 0.0)
    if overall >= 0.85:
        color = (0, 255, 0)  # Green
        grade = "A"
    elif overall >= 0.70:
        color = (0, 255, 255)  # Yellow
        grade = "B"
    elif overall >= 0.50:
        color = (0, 165, 255)  # Orange
        grade = "C"
    else:
        color = (0, 0, 255)  # Red
        grade = "D"
    
    cv2.putText(out, f"{overall:.0%} ({grade})", (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    y += line_height + 10
    
    # Template match status
    matches = rep_score.get("matches_template", False)
    match_text = "✓ Template Match" if matches else "✗ Needs Work"
    match_color = (0, 255, 0) if matches else (0, 0, 255)
    cv2.putText(out, match_text, (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, match_color, 2, cv2.LINE_AA)
    y += line_height
    
    # Joint scores
    cv2.putText(out, "---", (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    y += line_height - 5
    
    joint_scores = rep_score.get("joint_scores", {})
    for joint_name, scores in sorted(joint_scores.items()):
        score = scores.get("score", 0.0)
        is_ok = scores.get("is_within_range", False) and scores.get("is_within_tolerance", False)
        
        # Color based on pass/fail
        color = (0, 255, 0) if is_ok else (0, 0, 255)
        
        # Abbreviate joint names
        short_name = joint_name.replace("LEFT_", "L_").replace("RIGHT_", "R_")
        text = f"{short_name}: {score:.0%}"
        
        cv2.putText(out, text, (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
        y += line_height - 5
        
        if y > height - 20:
            break
    
    return out
