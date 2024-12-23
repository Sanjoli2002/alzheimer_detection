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
        return None  # No predictions found


def process_live_feed(result_image_label, predictions_text):
    """
    Process live webcam feed for predictions and draw bounding boxes.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Call inference on the captured frame
        cv2.imwrite("temp_image.jpg", frame)
        result = CLIENT.infer("temp_image.jpg", model_id=MODEL_ID)
        predictions = []

        if 'predictions' in result and result['predictions']:
            for prediction in result['predictions']:
                x1, y1, x2, y2 = map(int, [prediction['x'] - prediction['width'] / 2,
                                            prediction['y'] - prediction['height'] / 2,
                                            prediction['x'] + prediction['width'] / 2,
                                            prediction['y'] + prediction['height'] / 2])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{prediction['class']} ({prediction['confidence'] * 100:.2f}%)",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                predictions.append({
                    "Class": prediction['class'],
                    "Confidence (%)": f"{prediction['confidence']*100:.2f}",
                    "Coordinates": f"({x1}, {y1}), ({x2}, {y2})"
                })

            # Update UI with the result
            update_image_in_ui(frame, result_image_label)
            display_predictions_in_table(predictions, predictions_text)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
