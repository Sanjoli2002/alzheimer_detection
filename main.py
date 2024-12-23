# main.py
import tkinter as tk
from tkinter import filedialog
from src.pipline.pipeline import process_image, process_live_feed
from src.utils.helpers import handle_invalid_file
from src.utils.constants import MODEL_ID

def upload_image(result_label, result_image_label, predictions_text):
    """
    Handles image upload and starts processing the image.
    """
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.png")])
    if image_path and handle_invalid_file(image_path):
        result_label.config(text="Processing image...", fg="blue")
        process_image(image_path, result_image_label, predictions_text)
    else:
        result_label.config(text="Choose input method", fg="black")


def start_webcam(result_label, result_image_label, predictions_text):
    """
    Starts live webcam feed and processes the frames.
    """
    result_label.config(text="Initializing live webcam feed...", fg="blue")
    process_live_feed(result_image_label, predictions_text, result_label)


# Create main window
root = tk.Tk()
root.title("Alzheimer's Disease Detection")
root.geometry("1000x800")
root.config(bg="#4B86B4")

# Create a frame for the buttons
frame = tk.Frame(root, bg="#4B86B4")
frame.pack(pady=30)

# Create a label to display results
result_label = tk.Label(root, text="Choose input method to start", font=("Arial", 16), fg="white", bg="#4B86B4",
                        wraplength=800)
result_label.pack(pady=10)

# Create a label to display the processed image with bounding boxes
result_image_label = tk.Label(root, bg="#4B86B4")
result_image_label.pack(pady=20)

# Create buttons for uploading an image and starting webcam feed
image_button = tk.Button(frame, text="Select Image", width=20, command=lambda: upload_image(result_label, result_image_label, predictions_text))
image_button.grid(row=0, column=0, padx=20, pady=20)

webcam_button = tk.Button(frame, text="Start Webcam Feed", width=20, command=lambda: start_webcam(result_label, result_image_label, predictions_text))
webcam_button.grid(row=0, column=1, padx=20, pady=20)

# Create a text area to display prediction details
predictions_text = tk.Text(root, width=100, height=15, font=("Arial", 12))
predictions_text.pack(pady=20)
predictions_text.config(state=tk.DISABLED)

# Run the Tkinter event loop
root.mainloop()
