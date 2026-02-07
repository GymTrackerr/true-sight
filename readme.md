# TrueSight
Created by Daniel Kravec on January 27, 2026


## Real-time feedback
- have real time feedback on form during exercise talking
- e.g. "Keep your back straight", "Lower the weight slowly", etc.
- Have pacing notes, e.g. "Lift for 2 seconds, hold for 1 second, lower for 3 seconds"
- Can tap watch to mark rep completion or pacing
- how many reps to go
- Option to record video with overlayed feedback text and rep counts in real time

## Implementation notes
- use standard angle ranges for arms and legs, should be within 0/10/15/20/45/90/110/135/150/160/170/180 degrees etc ish
- one time buy to download


### Yolo Model 
The model used for pose estimation is `yolo11m-pose.pt` from Ultralytics YOLO. This model is capable of detecting 17 keypoints on the human body.
---
- Nose
- Left Eye
- Right Eye
- Left Ear
- Right Ear
- Left Shoulder
- Right Shoulder
- Left Elbow
- Right Elbow
- Left Wrist
- Right Wrist
- Left Hip
- Right Hip
- Left Knee
- Right Knee
- Left Ankle
- Right Ankle