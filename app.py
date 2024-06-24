# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 13:00:37 2024

@author: luisa
"""


from markupsafe import escape
from flask import Flask, render_template, request
from keras.applications import ResNet50
import cv2
import numpy as np 
import pandas as pd

app = Flask (__name__)
resnet = ResNet50(weights="imagenet", input_shape=(224,224,3),pooling='avg')
print("Model is loaded")

labels =pd.read_csv("labels.txt").values

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/prediction", methods=["POST"])
def prediction():
    
    img = request.files['img']
    
    img.save("img.jpg")
    
    image = cv2.imread("img.jpg")
    
    image = cv2.resize(image,(224,224))
    
    image = np.reshape(image, (1,224,224,3))
    
    pred = resnet.predict(image)
    
    pred = np.argmax(pred)
    
    pred = labels[pred]
    
    
    return render_template("prediction.html", data=pred)

if __name__=="__main__":
    app.run(debug=True)
    
#to generate requirements run !python -m pipreqs.pipreqs

