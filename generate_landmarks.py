import json
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

ROOT_DIR = Path(__file__).parent
MODEL_DIR = ROOT_DIR / "model"
DATA_DIR = ROOT_DIR / "data"

HAND_LANDMARK_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
LANDMARK_DATA_PATH = DATA_DIR / "landmarks.json"

print("Loading Model...")

base_options = python.BaseOptions(model_asset_path=str(HAND_LANDMARK_MODEL_PATH))
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

def landmarks_to_dict(result):
    return {
        "handedness": 
        [
            [
                {"category": h.category_name, "score": h.score} 
                for h in hand
            ]
            for hand in result.handedness
        ],
        "hand_landmarks": 
        [
            [
                {"x": lm.x, "y": lm.y, "z": lm.z} 
                for lm in hand
            ]
            for hand in result.hand_landmarks
        ],
        "hand_world_landmarks": 
        [
            [
                {"x": lm.x, "y": lm.y, "z": lm.z} 
                for lm in hand
            ]
            for hand in result.hand_world_landmarks
        ]
    }

print("Detecting Landmarks...")

# subdirectoies are class names
subdirs = [p for p in DATA_DIR.iterdir() if p.is_dir()]
detections = []
for sd in subdirs:
    class_name = sd.name
    print(f" Class: {class_name}")

    for img_path in sd.glob("*.jpg"):
        image = mp.Image.create_from_file(str(img_path))    
        detection_result = detector.detect(image)
        detection_dict = landmarks_to_dict(detection_result)

        # cv.imshow("x", cv.imread(img_path))
        # cv.waitKey(0)
        # cv.destroyWindow("x")

        # print(detection_dict.keys())
        # print(len(detection_dict["handedness"]))
        # print(len(detection_dict["hand_world_landmarks"][0]))
        # print(detection_dict["hand_world_landmarks"][0][0])
        # exit()

        detection_dict["label"] = class_name
        detections.append(detection_dict)

print(f"Total Images Processed: {len(detections)}")
print("Writing to json...")

# save detection landmarks to json
with open(LANDMARK_DATA_PATH, "w") as f:
    json.dump(detections, f, separators=(",", ":"))
