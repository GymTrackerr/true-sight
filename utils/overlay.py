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
