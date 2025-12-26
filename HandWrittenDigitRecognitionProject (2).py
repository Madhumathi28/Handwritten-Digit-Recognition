#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tkinter as tk
from tkinter import Canvas, Button, Label
from PIL import Image, ImageDraw
import tensorflow as tf

# Load a pre-trained model for MNIST digit recognition
model = tf.keras.models.load_model('mnist_model.h5')  # Replace with the path to your saved model

# Define accuracy and loss functions
def accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int64)
    y_pred = tf.argmax(y_pred, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))
    return accuracy

def loss(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

# Create a Tkinter GUI for drawing digits
class DigitRecognizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        
        self.canvas = Canvas(root, width=200, height=200, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=4)
        self.button_recognize = Button(root, text="Recognize", command=self.recognize_digit)
        self.button_recognize.grid(row=1, column=0, columnspan=2)
        self.button_clear = Button(root, text="Clear", command=self.clear_canvas)
        self.button_clear.grid(row=1, column=2, columnspan=2)
        self.label_result = Label(root, text="Prediction: ")
        
        self.label_result.grid(row=2, column=0, columnspan=4)
        
        self.image = Image.new("L", (200, 200), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.draw_digit)
        
    def draw_digit(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill="white")
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="Prediction: ")
        
    def recognize_digit(self):
        digit_image = self.image.resize((28, 28))
        digit_array = np.array(digit_image) / 255.0
        digit_array = digit_array[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
        prediction = model.predict(digit_array)
        predicted_digit = np.argmax(prediction)
        self.label_result.config(text=f"Prediction: {predicted_digit}")

# Create the GUI window and start the application
root = tk.Tk()
app = DigitRecognizerGUI(root)
root.mainloop()


# In[ ]:





# In[ ]:




