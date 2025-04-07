import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the trained model
model = load_model('covid19_detector_model.h5')

def predict_covid(image_path):
    # Preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img = np.expand_dims(img, axis=0) / 255.0
    
    # Make prediction
    prediction = model.predict(img)[0][0]
    result = 'COVID-19 Negative' if prediction > 0.5 else 'COVID-19 Positive'
    return result

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((250, 250), Image.LANCZOS)  # Use Image.LANCZOS for resizing
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        result = predict_covid(file_path)
        result_label.config(text=f'Result: {result}')

# Create the main window
root = tk.Tk()
root.title("COVID-19 Checker")
root.geometry('400x500')

# Create a label to display the image
panel = tk.Label(root)
panel.pack()

# Create a button to open the file dialog
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack()

# Create a label to display the result
result_label = tk.Label(root, text='Result: ', font=('Arial', 14))
result_label.pack()

# Run the main loop
root.mainloop()
