import torch
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

CLASSES = ["call", "dislike", "like", "take_picture"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = Path(__file__).parent
MODEL_DIR = ROOT_DIR / "model"

HAND_LANDMARK_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
GESTURE_CLASSIFY_MODEL_PATH = MODEL_DIR / "model.pt"

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

# TODO: load gesture classification dnn

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

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

  return annotated_image


def main():
    # TODO: get input image
    # input_img_path =
    # output_img_path = 

    # TODO: inference landmarks from image
    # image = mp.Image.create_from_file(str(input_img_path))    
    # detection_result = detector.detect(image)
    
    # TODO: pass hand_world_landmarks to gesture classifier dnn

    # TODO: inference on hand_world_landmarks

    # TODO: display result - landmarks + classification (?)
    # class_label = CLASSES[output_logits.argmax]
    # 
    # annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    # annotated_image = cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR)
    #
    # Add class label to annotated image
    # text_x = 50, text_y = 50 # top left corner-ish, can be improved later
    # cv.putText(annotated_image, f"{class_label}", (text_x, text_y), cv.FONT_HERSHEY_DUPLEX, FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)
    # cv.imwrite(output_img_path, annotated_image)
    # cv.imshow("result", annotated_image)
    # cv.waitKey(0)
    # cv.destroyWindow("result")
    pass

if __name__=="__main__":
    main()