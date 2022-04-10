# -*- coding: utf-8 -*-
"""
Created on Wed April 06 14:35:33 2022

@author: YASHIM GABRIEL
"""
#from __future__ import division, print_function

#import sys
import os
#import numpy as np
import cv2

# Keras
from keras.models import load_model
# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
model_path = 'model/pancreatic_model_5.h5'

# Load your trained model
model = load_model(model_path)
# model._make_predict_function()          # Necessary
#print('Model loaded. Start serving...')
#print('Model loaded. Check http://127.0.0.1:5000/')
CAT = ["No Tumor", "Tumor"]

def model_predict(filepath, model):
    IMG_SIZE = 128
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    x = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    preds = model.predict(x)
    return preds


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        pred = model_predict(file_path, model)
        result = CAT[int(pred[0][0])]
        return render_template('index.html', prediction_text=result)
    return None


if __name__ == '__main__':
    app.run(debug=True)
