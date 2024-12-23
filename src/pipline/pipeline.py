# pipeline.py
import cv2
from inference_sdk import InferenceHTTPClient
from alzheimer_detection.src.utils.constants import API_URL, API_KEY, MODEL_ID
from alzheimer_detection.src.utils.helpers import update_image_in_ui, display_predictions_in_table

# Set up the Inference Client
CLIENT = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

def process_image(image_path, result_image_label, predictions_text):
    """
    Process the uploaded image for predictions and draw bounding boxes.
    """
    result = CLIENT.infer(image_path, model_id=MODEL_ID)
    predictions = []

    if 'predictions' in result and result['predictions']:
        image = cv2.imread(image_path)

        # Draw bounding boxes
        for prediction in result['predictions']:
            x1, y1, x2, y2 = map(int, [prediction['x'] - prediction['width'] / 2,
                                        prediction['y'] - prediction['height'] / 2,
                                        prediction['x'] + prediction['width'] / 2,
                                        prediction['y'] + prediction['height'] / 2])

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{prediction['class']} ({prediction['confidence'] * 100:.2f}%)",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            predictions.append({
                "Class": prediction['class'],
                "Confidence (%)": f"{prediction['confidence']*100:.2f}",
                "Coordinates": f"({x1}, {y1}), ({x2}, {y2})"
            })

        # Update UI with the result
        update_image_in_ui(image, result_image_label)
        display_predictions_in_table(predictions, predictions_text)
    else:
        return None  # No predictions foundimport cv2
from inference_sdk import InferenceHTTPClient
from alzheimer_detection.src.utils.constants import API_URL, API_KEY, MODEL_ID
from alzheimer_detection.src.utils.helpers import update_image_in_ui, display_predictions_in_table

# Set up the Inference Client
CLIENT = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

import tempfile
import os
def process_live_feed(result_image_label, predictions_text, result_label):
    """
    Display live webcam feed and perform inference only when an MRI scan is detected.
    """
    cap = cv2.VideoCapture(0)  # Access the webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        result_label.config(text="Failed to access the webcam.", fg="red")
        return

    result_label.config(text="Waiting for MRI scan... Press 'q' to exit.", fg="blue")

    def update_frame():
        ret, frame = cap.read()
        if not ret:
            result_label.config(text="Unable to read frames from the webcam.", fg="red")
            cap.release()
            return

        # Display the live feed frame in the UI
        update_image_in_ui(frame, result_image_label)

        # Run inference every few frames for efficiency
        if hasattr(update_frame, "frame_count"):
            update_frame.frame_count += 1
        else:
            update_frame.frame_count = 0

        if update_frame.frame_count % 10 == 0:  # Perform inference every 10 frames
            temp_image_path = "temp_live_feed.jpg"
            cv2.imwrite(temp_image_path, frame)  # Save current frame temporarily
            result = CLIENT.infer(temp_image_path, model_id=MODEL_ID)
            predictions = []

            if "predictions" in result and result["predictions"]:
                for prediction in result["predictions"]:
                    x1, y1, x2, y2 = map(
                        int,
                        [
                            prediction["x"] - prediction["width"] / 2,
                            prediction["y"] - prediction["height"] / 2,
                            prediction["x"] + prediction["width"] / 2,
                            prediction["y"] + prediction["height"] / 2,
                        ],
                    )

                    # Draw bounding boxes and labels on the live frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{prediction['class']} ({prediction['confidence'] * 100:.2f}%)",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                    )

                    predictions.append(
                        {
                            "Class": prediction["class"],
                            "Confidence (%)": f"{prediction['confidence'] * 100:.2f}",
                            "Coordinates": f"({x1}, {y1}), ({x2}, {y2})",
                        }
                    )

                # Update predictions in the UI
                display_predictions_in_table(predictions, predictions_text)

        # Schedule the next frame update
        result_label.after(10, update_frame)

    update_frame()  # Start the update loop



def is_mri_scan_detected(frame):
    """
    A function that determines whether an MRI scan is detected in the frame.
    Placeholder logic: You should replace this with a better method.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # For example, check if the image is mostly white (like an MRI scan)
    if cv2.countNonZero(thresh) > 5000:
        return True
    return False
