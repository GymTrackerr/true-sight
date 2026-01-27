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
    LEFT_ARM = (Joints.LEFT_SHOULDER, Joints.LEFT_ELBOW, Joints.LEFT_WRIST)
    RIGHT_ARM = (Joints.RIGHT_SHOULDER, Joints.RIGHT_ELBOW, Joints.RIGHT_WRIST)
    LEFT_LEG = (Joints.LEFT_HIP, Joints.LEFT_KNEE, Joints.LEFT_ANKLE)
    RIGHT_LEG = (Joints.RIGHT_HIP, Joints.RIGHT_KNEE, Joints.RIGHT_ANKLE)

class frame_save:
    def __init__(self, result):
        self.result = result

    def get_joint(self, joint:Joints):
        # print(joint)
        # if (joint < 0 or joint >= len(self.result['keypoints'])):
        #     raise ValueError("Invalid joint index")
        return self.result.keypoints.xy[0][joint.value]
    
    def get_connected_joints(self, connected:ConnectedJoints):
        return (
            self.get_joint(connected.value[0]),
            self.get_joint(connected.value[1]),
            self.get_joint(connected.value[2]),
        )
    
    def find_changing_angle(self, frame2):
        highest_change = [-1, -1, -1, -1]
        second_highest = [-1, -1, -1, -1]

        for i in range(len(ConnectedJoints)):
            connected = list(ConnectedJoints)[i]
            # Get angle differences between frames
            angle1 = self.get_angle_diff(connected.value[0], connected.value[1], connected.value[2])
            angle2 = frame2.get_angle_diff(connected.value[0], connected.value[1], connected.value[2])
            DIF = abs(angle1 - angle2)
            
            if DIF > highest_change[1]:
                second_highest = highest_change
                highest_change = [i, DIF, angle1, angle2]
            elif (DIF > second_highest[1]):
                second_highest = [i, DIF, angle1, angle2]
        print(f"Highest change found: {list(ConnectedJoints)[highest_change[0]].name} = {highest_change[1]}, before: {highest_change[2]}, after: {highest_change[3]}")
        
        # if second_highest[0] != -1:
        print(f"Second highest change found: {list(ConnectedJoints)[second_highest[0]].name} = {second_highest[1]}, before: {second_highest[2]}, after: {second_highest[3]}")
        return highest_change

    def find_changing_loc(self, frame2):
        highest_change = [-1, -1]
        second_highest = [-1, -1]
        for i in range(len(Joints)):
            joint = Joints(i)
            # compare joint locations between two frames
            loc1 = self.get_joint(joint)
            loc2 = frame2.get_joint(joint)
            DIF = abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
            if DIF > highest_change[1]:
                second_highest = highest_change
                highest_change = [i, DIF]
        
        print(f"Highest location change found: {Joints(highest_change[0]).name} = {highest_change[1]}")
        
        if second_highest[0] != -1:
            print(f"Second highest location change found: {Joints(second_highest[0]).name} = {second_highest[1]}")
        return highest_change

    def get_angle_diff(self, joint1, joint2, joint3):
        # Placeholder for angle difference calculation
        loc_1 = self.get_joint(joint1)
        loc_2 = self.get_joint(joint2)
        loc_3 = self.get_joint(joint3)

        angle = degrees(atan2(loc_3[1]-loc_2[1], loc_3[0]-loc_2[0]) - atan2(loc_1[1]-loc_2[1], loc_1[0]-loc_2[0]))

        if angle < 0:
            angle += 360
        return angle
    
