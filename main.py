"""
Gesture Classification Pipeline Inference

Install:
pip install -r requirements.txt

Run:
(1) execute: python main.py
(2) navigate to Gradio URL in a web browser.
(3) upload image
(4) click submit
"""

import joblib
import cv2 as cv
import gradio as gr
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

CLASSES = ["call", "dislike", "like", "take_picture"]

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model"
OUTPUT_DIR = ROOT_DIR / "outputs"

HAND_LANDMARK_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
GESTURE_CLASSIFIER_PATH = MODEL_DIR / "gesture_classifier.joblib"
GC_SCALER_PATH = MODEL_DIR / "gc_scaler.joblib"
ERROR_IMG_PATH = DATA_DIR / "detection_failed.png"

# For annotation
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

# Load Mediapipe hand landmarks model.
base_options = python.BaseOptions(model_asset_path=str(HAND_LANDMARK_MODEL_PATH))
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Load feature scaler
gc_scaler = joblib.load(GC_SCALER_PATH)

# Load gesture classifier
gesture_classifier = joblib.load(GESTURE_CLASSIFIER_PATH)

def landmarks_to_dict(result):
    # Convert Hand Landmarks object into dictionary 
    # for convenient serialization.
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

def featurize(entry):
    """
    Turn one MediaPipe hand_landmarks record into a fixed size list of
    numbers that a model can learn from.

    What it does, step by step:
      1. Look at the first hand found in the picture.
      2. Move all points so the wrist point becomes zero (position 0,0,0).
      3. Shrink or grow all points using the size of the hand, so a hand
         close to the camera and a hand far from the camera look the same.
      4. Turn the 21 points, each with x, y, z, into one long list of 63 numbers.

    If no hand was found in the picture, this returns None instead, so we
    can skip that picture later instead of guessing wrong numbers for it.
    """
    # get the list of hands found in this picture, if any
    lm_list = entry.get("hand_landmarks")
    # if there is no hand list, or it is empty, we have nothing to use
    if not lm_list or len(lm_list) == 0:
        # tell the caller this picture has no usable hand
        return None
    # take just the first hand found (some pictures could have more than one)
    pts = lm_list[0]
    # if that first hand somehow has no points, skip it too
    if not pts:
        return None

    # build a table of numbers: one row per point, three columns for x, y, z
    coords = np.array([[p["x"], p["y"], p["z"]] for p in pts], dtype=np.float64)
    # remember where the wrist point is (point number 0)
    wrist = coords[0].copy()
    # move every point so the wrist becomes the new zero point
    coords -= wrist  # this makes the hand's position in the picture not matter

    # measure the distance from the wrist to the middle finger's base knuckle
    scale = np.linalg.norm(coords[9])  # this tells us roughly how big the hand looks
    # avoid dividing by a number that is basically zero
    if scale < 1e-8:
        scale = 1e-8
    # shrink or grow all points using that distance, so hand size does not matter
    coords /= scale  # this makes the hand's size or distance from camera not matter

    # turn the table of 21 rows and 3 columns into one flat list of 63 numbers
    return coords.flatten()

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    mp_drawing.draw_landmarks(
      annotated_image,
      hand_landmarks,
      mp_hands.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style())

    # # Get the top left corner of the detected hand's bounding box.
    # height, width, _ = annotated_image.shape
    # x_coordinates = [landmark.x for landmark in hand_landmarks]
    # y_coordinates = [landmark.y for landmark in hand_landmarks]
    # text_x = int(min(x_coordinates) * width)
    # text_y = int(min(y_coordinates) * height) - MARGIN

    # # Draw handedness (left or right hand) on the image.
    # cv.putText(annotated_image, f"{handedness[0].category_name}",
    #             (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
    #             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

  return annotated_image

def run_inference(input_img_path):
    output_img_path = OUTPUT_DIR / f"{Path(input_img_path).stem}_annotated.jpg"

    # Inference landmarks from image
    image = mp.Image.create_from_file(str(input_img_path))    
    detection_result = detector.detect(image)
    
    # Preprocess hand landmarks
    detection_dict = landmarks_to_dict(detection_result)
    feature_vector = featurize(detection_dict)

    # If no hands are detected, we cannot proceed
    if feature_vector is None:
        return ERROR_IMG_PATH

    feature_mat = feature_vector.reshape(1, -1)
    scaled_input_vector = gc_scaler.transform(feature_mat)

    # Inference gesture class from feature vector
    prediction = gesture_classifier.predict(scaled_input_vector)

    # Display / save result - landmarks + classification
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    annotated_image = cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR)
    
    # Add class label to annotated image
    text_x = 50; text_y = 50 # top left corner-ish, can be improved later
    cv.putText(annotated_image, 
               f"Prediction = {str(prediction[0]).upper()}", 
               (text_x, text_y), 
               cv.FONT_HERSHEY_DUPLEX, 
               FONT_SIZE, 
               HANDEDNESS_TEXT_COLOR, 
               FONT_THICKNESS, 
               cv.LINE_AA)

    cv.imwrite(output_img_path, annotated_image)
    return output_img_path

def main():
    app_ui = gr.Interface(
        title="AIG210 Project Demo",
        fn=run_inference,
        inputs=gr.Image(type="filepath", label="Upload Image"),
        outputs=gr.Image(label="Result"),
    )

    app_ui.launch()

if __name__=="__main__":
    main()