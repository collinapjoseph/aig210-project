"""
Hand Gesture Recognition from MediaPipe Landmarks
This program reads hand landmark data, builds a small dataset,
trains a few models to guess the gesture, and saves charts and numbers
about how well each model did.

Gestures it learns to tell apart:
    call, dislike, like, take_picture

How to run it:
    python train_gesture_classifier.py project_data_folder output_folder

What it saves in the output folder:
    class_distribution.png
    missing_detection_rate.png
    confusion_matrix.png
    model_comparison.png
    per_class_metrics.png
    metrics_summary.json
"""

# json lets us read and write .json files
import json
# os lets us build file paths and make folders
import os
# argparse lets the user type extra options when running this file
import argparse
# numpy helps us do math on lists of numbers quickly
import numpy as np
# matplotlib is the library we use to draw charts
import matplotlib
# this tells matplotlib to save charts to files instead of trying to pop up a window
matplotlib.use("Agg")
# pyplot is the part of matplotlib we actually draw with
import matplotlib.pyplot as plt

# this splits our data into a training group and a testing group
from sklearn.model_selection import train_test_split
# this rescales numbers so they are easier for models to learn from
from sklearn.preprocessing import StandardScaler
# this is one type of model we will train, called a Random Forest
from sklearn.ensemble import RandomForestClassifier
# this is another type of model, called a Support Vector Machine
from sklearn.svm import SVC
# this is a third type of model, a small neural network
from sklearn.neural_network import MLPClassifier
# these are tools that measure how good our models are
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_fscore_support
)

# a fixed number so every run of this program splits and trains the same way
RANDOM_STATE = 42
# the four gesture names we are trying to recognize, in a fixed order
CLASSES = ["call", "dislike", "like", "take_picture"]


# this function opens the landmarks file and gives back its contents
def load_landmarks(json_path):
    # open the file for reading
    with open(json_path, "r") as f:
        # turn the file's text into a Python list/dictionary
        data = json.load(f)
    # send that data back to whoever called this function
    return data


# this function turns one image's hand data into a simple list of numbers
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


# this function builds the full list of feature rows and their matching labels
def build_dataset(records):
    # X will hold the feature numbers, y will hold the matching gesture name
    X, y, dropped = [], [], 0
    # go through every picture's record one at a time
    for r in records:
        # turn this record into a list of numbers, or None if no hand was found
        feat = featurize(r)
        # if there was no usable hand, count it and skip to the next picture
        if feat is None:
            dropped += 1
            continue
        # otherwise save the numbers
        X.append(feat)
        # and save the correct gesture name for those numbers
        y.append(r["label"])
    # turn the plain lists into numpy arrays and also return how many we dropped
    return np.array(X), np.array(y), dropped


# this function draws a bar chart showing how many pictures are in each class
def plot_class_distribution(records, out_dir):
    # count how many pictures belong to each gesture
    counts = {c: sum(1 for r in records if r["label"] == c) for c in CLASSES}
    # start a new blank chart of a certain size
    plt.figure(figsize=(6, 4))
    # draw one bar per gesture, using the counts we just made
    plt.bar(counts.keys(), counts.values(), color="#4C72B0")
    # give the chart a title
    plt.title("Class distribution of raw dataset")
    # label the up and down axis
    plt.ylabel("Number of samples")
    # tidy up the spacing so labels are not cut off
    plt.tight_layout()
    # save the chart as an image file in the output folder
    plt.savefig(os.path.join(out_dir, "class_distribution.png"), dpi=150)
    # close the chart so it does not use extra memory
    plt.close()


# this function draws a chart of how often MediaPipe failed to find a hand
def plot_missing_detection(records, out_dir):
    # start a counter of missed pictures for each gesture, starting at zero
    miss = {c: 0 for c in CLASSES}
    # start a counter of total pictures for each gesture, starting at zero
    total = {c: 0 for c in CLASSES}
    # look at every picture's record
    for r in records:
        # add one to the total count for this gesture
        total[r["label"]] += 1
        # if this record has no hand landmarks at all
        if not r.get("hand_landmarks"):
            # add one to the missed count for this gesture
            miss[r["label"]] += 1
    # turn the raw counts into percentages for each gesture
    rates = {c: 100.0 * miss[c] / total[c] for c in CLASSES}
    # start a new blank chart
    plt.figure(figsize=(6, 4))
    # draw one bar per gesture showing its percentage of missed detections
    plt.bar(rates.keys(), rates.values(), color="#C44E52")
    # give the chart a title
    plt.title("MediaPipe hand detection failure rate by class")
    # label the up and down axis
    plt.ylabel("Percent of images with no hand detected")
    # tidy up spacing
    plt.tight_layout()
    # save the chart to a file
    plt.savefig(os.path.join(out_dir, "missing_detection_rate.png"), dpi=150)
    # close the chart
    plt.close()
    # send back the raw counts in case the caller wants them too
    return miss, total


# this function draws a confusion matrix, which shows correct versus wrong guesses
def plot_confusion(cm, labels, out_dir, model_name):
    # start a new blank chart of a certain size
    plt.figure(figsize=(5.5, 5))
    # show the confusion matrix as a grid of colored squares
    plt.imshow(cm, cmap="Blues")
    # give the chart a title that includes the model's name
    plt.title(f"Confusion matrix for {model_name}")
    # add a color scale bar on the side
    plt.colorbar()
    # make a list of positions for the tick marks, one per gesture
    tick = np.arange(len(labels))
    # put gesture names along the bottom, tilted so they fit
    plt.xticks(tick, labels, rotation=45, ha="right")
    # put gesture names along the side
    plt.yticks(tick, labels)
    # go through every row of the grid
    for i in range(len(labels)):
        # go through every column of the grid
        for j in range(len(labels)):
            # write the actual number in that square, using white or black text
            # depending on how dark the square's background color is
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    # label the side axis
    plt.ylabel("True label")
    # label the bottom axis
    plt.xlabel("Predicted label")
    # tidy up spacing
    plt.tight_layout()
    # pick a plain file name for the Random Forest model, or a custom one for others
    fname = "confusion_matrix.png" if model_name == "Random Forest" else f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    # save the chart to a file
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    # close the chart
    plt.close()


# this function draws a bar chart comparing the accuracy of each model
def plot_model_comparison(results, out_dir):
    # get the list of model names
    names = list(results.keys())
    # get the matching accuracy score for each model
    accs = [results[n]["accuracy"] for n in names]
    # start a new blank chart
    plt.figure(figsize=(6, 4))
    # draw one bar per model, using its accuracy as the bar height
    bars = plt.bar(names, accs, color=["#55A868", "#4C72B0", "#C44E52"])
    # fix the up and down axis to go from 0 to 1, since accuracy is a fraction
    plt.ylim(0, 1.0)
    # label the up and down axis
    plt.ylabel("Test accuracy")
    # give the chart a title
    plt.title("Model comparison on held out test set")
    # write the exact accuracy number above each bar
    for b, a in zip(bars, accs):
        plt.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.3f}", ha="center")
    # tidy up spacing
    plt.tight_layout()
    # save the chart to a file
    plt.savefig(os.path.join(out_dir, "model_comparison.png"), dpi=150)
    # close the chart
    plt.close()


# this function draws a chart comparing precision, recall, and F1 for each class
def plot_per_class_metrics(y_test, y_pred, labels, out_dir):
    # calculate precision, recall, and F1 score for every gesture class
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=labels, zero_division=0)
    # make a list of positions, one per gesture, to place bars at
    x = np.arange(len(labels))
    # decide how wide each small bar should be
    width = 0.25
    # start a new blank chart
    plt.figure(figsize=(7, 4))
    # draw the precision bars, shifted slightly left
    plt.bar(x - width, p, width, label="Precision")
    # draw the recall bars, in the middle
    plt.bar(x, r, width, label="Recall")
    # draw the F1 bars, shifted slightly right
    plt.bar(x + width, f1, width, label="F1")
    # label each group of bars with its gesture name
    plt.xticks(x, labels)
    # fix the up and down axis so it is easy to compare charts
    plt.ylim(0, 1.05)
    # give the chart a title
    plt.title("Per class precision, recall, and F1 for Random Forest")
    # show a small legend explaining the bar colors
    plt.legend()
    # tidy up spacing
    plt.tight_layout()
    # save the chart to a file
    plt.savefig(os.path.join(out_dir, "per_class_metrics.png"), dpi=150)
    # close the chart
    plt.close()
    # send back the three lists of numbers in case they are needed later
    return p, r, f1


# this is the main function that runs everything in order
def main():
    # create a helper that can read extra typed in options
    parser = argparse.ArgumentParser()
    # this option is the folder that holds the project data and landmarks.json
    parser.add_argument("data_dir", nargs="?", default=".")
    # this option is the folder where charts and results should be saved
    parser.add_argument("out_dir", nargs="?", default="./outputs")
    # actually read whatever the user typed when running this program
    args = parser.parse_args()

    # make the output folder if it does not already exist
    os.makedirs(args.out_dir, exist_ok=True)

    # load all the landmark records from the json file
    records = load_landmarks(os.path.join(args.data_dir, "landmarks.json"))
    # print how many records we loaded, just so we can see progress
    print(f"Loaded {len(records)} landmark records.")

    # draw and save the chart showing how many pictures are in each class
    plot_class_distribution(records, args.out_dir)
    # draw and save the chart showing missed hand detections, and keep the counts
    miss, total = plot_missing_detection(records, args.out_dir)
    # print the missed detection counts so we can see them right away
    print("Missing hand detection counts per class:", miss)

    # turn all the records into a table of numbers (X) and matching labels (y)
    X, y, dropped = build_dataset(records)
    # print the shape of our data and how many pictures we had to skip
    print(f"Built feature matrix: X={X.shape}, dropped={dropped} samples with no hand detected")

    # split the data into 70 percent for training and 30 percent for later use
    # stratify keeps the same class balance in both pieces
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    # split that remaining 30 percent in half: 15 percent validation, 15 percent test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    # print how many pictures ended up in each of the three groups
    print(f"Split sizes: train {len(X_train)}, val {len(X_val)}, test {len(X_test)}")

    # create a tool that will rescale our numbers to a standard range
    scaler = StandardScaler()
    # learn the rescaling from the training data, and apply it to the training data
    X_train_s = scaler.fit_transform(X_train)
    # apply that same rescaling to the validation data, without relearning it
    X_val_s = scaler.transform(X_val)
    # apply that same rescaling to the test data too
    X_test_s = scaler.transform(X_test)

    # set up the three different models we want to try, with their settings
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=1000, random_state=RANDOM_STATE
        ),
    }

    # this will store the results for every model we try
    results = {}
    # these will keep track of whichever model turns out to be the best so far
    best_name, best_acc, best_model = None, -1, None
    # go through each model one at a time
    for name, model in models.items():
        # teach the model using the training data
        model.fit(X_train_s, y_train)
        # check how accurate it is on the validation data
        val_acc = accuracy_score(y_val, model.predict(X_val_s))
        # use the model to guess labels for the test data
        y_pred_test = model.predict(X_test_s)
        # check how accurate those guesses were compared to the real labels
        test_acc = accuracy_score(y_test, y_pred_test)
        # save both accuracy numbers for this model
        results[name] = {"val_accuracy": val_acc, "accuracy": test_acc}
        # print the accuracy numbers so we can see progress as it runs
        print(f"{name}: val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")
        # build a confusion matrix comparing real labels to guessed labels
        cm = confusion_matrix(y_test, y_pred_test, labels=CLASSES)
        # draw and save a chart of that confusion matrix
        plot_confusion(cm, CLASSES, args.out_dir, name)
        # if this model beat our best score so far, remember it as the new best
        if test_acc > best_acc:
            best_acc, best_name, best_model = test_acc, name, model

    # draw and save a chart comparing accuracy across all three models
    plot_model_comparison(results, args.out_dir)

    # use the best model to guess labels for the test data one more time
    y_pred_best = best_model.predict(X_test_s)
    # build a detailed report of precision, recall, and F1 for the best model
    report_dict = classification_report(y_test, y_pred_best, labels=CLASSES, output_dict=True, zero_division=0)
    # draw and save a chart of precision, recall, and F1 per gesture class
    p, r, f1 = plot_per_class_metrics(y_test, y_pred_best, CLASSES, args.out_dir)

    # gather everything worth remembering about this run into one dictionary
    summary = {
        "n_total_records": len(records),
        "n_dropped_no_hand": int(dropped),
        "missing_detection_by_class": miss,
        "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "model_results": results,
        "best_model": best_name,
        "best_test_accuracy": best_acc,
        "classification_report_best_model": report_dict,
    }
    # open a new file for writing and save the summary as json text
    with open(os.path.join(args.out_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # print a heading so the final summary is easy to spot in the output
    print("\n=== SUMMARY ===")
    # print the summary, leaving out the very long detailed report part
    print(json.dumps({k: v for k, v in summary.items() if k != "classification_report_best_model"}, indent=2))
    # print one last friendly line naming the winning model and its score
    print(f"\nBest model: {best_name} with test accuracy {best_acc:.4f}")


# this makes sure main() only runs when this file is run directly,
# not when it is imported into another file
if __name__ == "__main__":
    main()
