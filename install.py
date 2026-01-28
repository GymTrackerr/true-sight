from ultralytics import YOLO
from utils.joints import Joints, frame_save
from utils.exercisedb import ExerciseDBSearch
# https://docs.ultralytics.com/tasks/pose/
api_base_url = "http://192.168.3.21:5002"

# load model
model = YOLO("yolo26s-pose.pt")

# Predict with the model
# results = model.track(source="./input/test2.mov", show=True, save=True)




def get_results(imagePath):
    results = model(source=imagePath, save=True, save_txt=True)
    # Access the results, one per person in frame
    curs = []

    for result in results:
        curs.append(frame_save(result))
        # DIF = cur.get_joint(Joints.RIGHT_ELBOW)

        # DIF = cur.get_angle_diff(Joints.RIGHT_WRIST, Joints.RIGHT_ELBOW, Joints.RIGHT_SHOULDER)
        # print(DIF)

        
        # boxes = result.boxes  # Boxes object for bounding box outputs
        # masks = result.masks  # Masks object for segmentation masks outputs
        # keypoints = result.keypoints  # Keypoints object for pose outputs
        # # keypoints = results[0].keypoints.xy[0] # Get all keypoints for the first person
        # right_elbow = keypoints[8] # Select index 8
        # print(right_elbow)
        # # kpts_data = keypoints.data 
        # # print(f"Keypoints:\n{keypoints.data}")
        # # print(f"Number of keypoints detected: {kpts_data.shape[0]}")
        # # print(f"keypoints : {kpts_data}")
        # # probs = result.probs  # Probs object for classification outputs
        # # obb = result.obb  # Oriented boxes object for OBB outputs
        # # result.show()  # display to screen
        result.save(filename="result.jpg")  # save to disk

        # xy = result.keypoints.xy  # x and y coordinates
        # xyn = result.keypoints.xyn  # normalized
        # kpts = result.keypoints.data  # x, y, visibility (if available)
        # print(f"Keypoints:\n{kpts}")

    return curs

def find_exercise():
    # find exercise from db search
    EDB = ExerciseDBSearch()

    found = EDB.search("Dumbbell Alternate Bicep Curl")
    # curs = []

    cur0 = get_results(api_base_url+found.get('images')[0])
    cur1 = get_results(api_base_url+found.get('images')[1])


    change = cur0[0].find_changing_angle(cur1[0])
    print(f"Change between two frames: {change[0]} with diff {change[1]}")
    
    # curs = cur0 + cur1

    # for i in range(len(curs) - 1):
    #     if (i + 1) > len(curs):
    #         pass
    #     DIF = curs[i].find_changing_angle(curs[i+1])
    #     print(DIF)

    # get key muscle and joints
    # primary_muscles = found.get('primaryMuscles', [])
    # print(primary_muscles)
    # clip = model(source=found.get('images'), save=True, save_txt=True)
    # get_results(found.get('images')[0], primary_muscles)

    # get_results(found.get('images')[1], primary_muscles)




    # get reference poses by using API

    pass
if __name__ == "__main__":
    find_exercise()
    # get_results("./tests/0.jpg")
    # get_results("./tests/1.jpg")