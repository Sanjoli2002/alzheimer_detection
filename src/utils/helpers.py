# helpers.py
import os
import cv2
from PIL import Image, ImageTk
from tkinter import messagebox


def update_image_in_ui(image, result_image_label, image_size=(800, 600)):
    """
    Update the displayed image in the Tkinter label.
    """
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img = img.resize(image_size)  # Resize the image to fit in the window
    img_tk = ImageTk.PhotoImage(img)
    result_image_label.config(image=img_tk)
    result_image_label.image = img_tk  # Keep reference


def handle_invalid_file(image_path):
    """
    Handles invalid image files.
    """
    if not os.path.exists(image_path):
        messagebox.showerror("Error", "Invalid image file.")
        return False
    return True


def display_predictions_in_table(predictions, predictions_text):
    """
    Displays predictions in the Tkinter Text widget.
    """
    if predictions:
        formatted_predictions = "\n".join([f"{p['Class']} ({p['Confidence (%)']}) - Coordinates: {p['Coordinates']}"
                                           for p in predictions])
        predictions_text.config(state='normal')
        predictions_text.delete(1.0, 'end')  # Clear previous content
        predictions_text.insert('end', formatted_predictions)
        predictions_text.config(state='disabled')  # Make it non-editable
    else:
        predictions_text.config(state='normal')
        predictions_text.delete(1.0, 'end')
        predictions_text.insert('end', "No predictions to show.")
        predictions_text.config(state='disabled')
