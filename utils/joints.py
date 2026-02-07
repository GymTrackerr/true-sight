from ultralytics import YOLO
from enum import Enum
from math import degrees, atan2

class Joints(Enum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

class ConnectedJoints(Enum):
    LEFT_ARM = ((0), (Joints.LEFT_SHOULDER, Joints.LEFT_ELBOW, Joints.LEFT_WRIST))
    RIGHT_ARM = ((1), (Joints.RIGHT_SHOULDER, Joints.RIGHT_ELBOW, Joints.RIGHT_WRIST))
    LEFT_LEG = ((0), (Joints.LEFT_HIP, Joints.LEFT_KNEE, Joints.LEFT_ANKLE))
    RIGHT_LEG = ((1), (Joints.RIGHT_HIP, Joints.RIGHT_KNEE, Joints.RIGHT_ANKLE))
    LEFT_HIP = ((0), (Joints.LEFT_SHOULDER, Joints.LEFT_HIP, Joints.LEFT_KNEE))
    RIGHT_HIP = ((1), (Joints.RIGHT_SHOULDER, Joints.RIGHT_HIP, Joints.RIGHT_KNEE))
    LEFT_SHOULDER = ((0), (Joints.LEFT_EAR, Joints.LEFT_SHOULDER, Joints.LEFT_ELBOW))
    RIGHT_SHOULDER = ((1), (Joints.RIGHT_EAR, Joints.RIGHT_SHOULDER, Joints.RIGHT_ELBOW))

class frame_save:
    def __init__(self, result):
        self.result = result
    
    def has_detections(self):
        """Check if this frame has any keypoint detections"""
        return (self.result.keypoints is not None and 
                self.result.keypoints.xy is not None and 
                len(self.result.keypoints.xy) > 0)

    def get_joint_xy(self, joint:Joints):
        if not self.has_detections():
            return None
        pt = self.result.keypoints.xy[0][joint.value]
        # Conver to python floats
        return (float(pt[0]), float(pt[1]))
    
    def get_connected_joints(self, connected:ConnectedJoints):
        return (
            self.get_joint_xy(connected.value[1][0]),
            self.get_joint_xy(connected.value[1][1]),
            self.get_joint_xy(connected.value[1][2]),
        )
    
    def midpoint(self, j1:Joints, j2:Joints):
        """Calculate midpoint between two joints"""
        loc1 = self.get_joint_xy(j1)
        loc2 = self.get_joint_xy(j2)
        if loc1 is None or loc2 is None:
            return None
        return ((loc1[0] + loc2[0]) / 2, (loc1[1] + loc2[1]) / 2)
    
    def shoulder_width(self):
        """Calculate shoulder width as distance between left and right shoulder"""
        return self.find_width(Joints.LEFT_SHOULDER, Joints.RIGHT_SHOULDER)
    
    def hip_width(self):
        """Calculate hip width as distance between left and right hip"""
        return self.find_width(Joints.LEFT_HIP, Joints.RIGHT_HIP)
    
    def torso_center(self):
        sm = self.midpoint(Joints.LEFT_SHOULDER, Joints.RIGHT_SHOULDER)
        hm = self.midpoint(Joints.LEFT_HIP, Joints.RIGHT_HIP)
        if sm is None or hm is None:
            return None
        return ((sm[0] + hm[0]) / 2.0, (sm[1] + hm[1]) / 2.0)

    def find_width(self, j1:Joints, j2:Joints):
        """Calculate width between two joints"""
        loc1 = self.get_joint_xy(j1)
        loc2 = self.get_joint_xy(j2)
        if loc1 is None or loc2 is None:
            return None
        dx = loc1[0] - loc2[0]
        dy = loc1[1] - loc2[1]
        return (dx**2 + dy**2)**0.5
    
    def _dist(self, j1: Joints, j2: Joints):
        a = self.get_joint_xy(j1)
        b = self.get_joint_xy(j2)
        if a is None or b is None:
            return None
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx*dx + dy*dy) ** 0.5

    def segment_lengths_norm(self):
        """
        Normalized segment lengths for view/rotation detection.
        Returns None if no scale.
        """
        sc = self.scale()
        if sc is None or sc <= 1e-6:
            return None

        segs = {
            "UPPER_ARM_L": self._dist(Joints.LEFT_SHOULDER, Joints.LEFT_ELBOW),
            "FOREARM_L":   self._dist(Joints.LEFT_ELBOW, Joints.LEFT_WRIST),
            "UPPER_ARM_R": self._dist(Joints.RIGHT_SHOULDER, Joints.RIGHT_ELBOW),
            "FOREARM_R":   self._dist(Joints.RIGHT_ELBOW, Joints.RIGHT_WRIST),

            "THIGH_L":     self._dist(Joints.LEFT_HIP, Joints.LEFT_KNEE),
            "SHIN_L":      self._dist(Joints.LEFT_KNEE, Joints.LEFT_ANKLE),
            "THIGH_R":     self._dist(Joints.RIGHT_HIP, Joints.RIGHT_KNEE),
            "SHIN_R":      self._dist(Joints.RIGHT_KNEE, Joints.RIGHT_ANKLE),
        }

        out = {}
        for k, v in segs.items():
            if v is not None:
                out[k] = float(v) / float(sc)
        return out if out else None

    def scale(self):
        shoulder_w = self.shoulder_width()
        if (shoulder_w is not None and shoulder_w > 0):
            return shoulder_w
        hip_w = self.hip_width()
        if (hip_w is not None and hip_w > 0):
            return hip_w
        
        # Final Fallback, Torso length (shoulder to hip)
        torso_length = self.find_width(Joints.LEFT_SHOULDER, Joints.LEFT_HIP)
        if (torso_length is not None and torso_length > 0):
            return torso_length
        return None

    def angles_dict(self):
        """Calculate angles for all connected joints and return as dict (floats only)."""
        angles = {}
        if not self.has_detections():
            return angles

        for connected in ConnectedJoints:
            a = self.get_angle_diff(
                connected.value[1][0],
                connected.value[1][1],
                connected.value[1][2]
            )
            if a is not None:
                angles[connected.name] = float(a)
        return angles


    def find_changing_joints(self, frame2):
        """Array of [name, index, difference, angle1, angle2] for connecting joints"""
        changes = []
        
        # Skip if either frame has no detections
        if not self.has_detections() or not frame2.has_detections():
            return changes

        for i in range(len(ConnectedJoints)):
            connected = list(ConnectedJoints)[i]
            # Get angle differences between frames
            angle1 = self.get_angle_diff(connected.value[1][0], connected.value[1][1], connected.value[1][2])
            angle2 = frame2.get_angle_diff(connected.value[1][0], connected.value[1][1], connected.value[1][2])
            
            if angle1 is not None and angle2 is not None:
                DIF = abs(angle1 - angle2)
                changes.append([connected.name, i, DIF, angle1, angle2])

        # changes.sort()
        # sort by changes[2]
        changes.sort(key=lambda x: x[2], reverse=True)
        if changes:
            print(changes)
        return changes

    def find_changing_loc(self, frame2):
        highest_change = [-1, -1]
        second_highest = [-1, -1]
        for i in range(len(Joints)):
            joint = Joints(i)
            # compare joint locations between two frames
            loc1 = self.get_joint_xy(joint)
            loc2 = frame2.get_joint_xy(joint)
            if loc1 is None or loc2 is None:
                continue
            DIF = abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
            if DIF > highest_change[1]:
                second_highest = highest_change
                highest_change = [i, DIF]
        
        print(f"Highest location change found: {Joints(highest_change[0]).name} = {highest_change[1]}")
        
        if second_highest[0] != -1:
            print(f"Second highest location change found: {Joints(second_highest[0]).name} = {second_highest[1]}")
        return highest_change

    def get_angle_diff(self, joint1, joint2, joint3):
        """Calculate the angle AT joint2 (the bend angle)"""
        loc_1 = self.get_joint_xy(joint1)
        loc_2 = self.get_joint_xy(joint2)
        loc_3 = self.get_joint_xy(joint3)
        
        # Return None if any joint is missing
        if loc_1 is None or loc_2 is None or loc_3 is None:
            return None
        
        # Vectors from joint2 to joint1 and joint2 to joint3
        v1 = [loc_1[0] - loc_2[0], loc_1[1] - loc_2[1]]
        v2 = [loc_3[0] - loc_2[0], loc_3[1] - loc_2[1]]
        
        # Dot product and magnitudes for angle between vectors
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = (v1[0]**2 + v1[1]**2)**0.5
        mag2 = (v2[0]**2 + v2[1]**2)**0.5
        
        if mag1 == 0 or mag2 == 0:
            return None
        
        cos_angle = dot / (mag1 * mag2)
        angle = degrees(atan2((v1[0]*v2[1] - v1[1]*v2[0]), dot))
        return round(abs(angle))  # 0-180°, always positive
