# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 19:07:57 2022

@author: andrwngyn
"""

import numpy as np
from flask import Flask, request, render_template
from keras.models import load_model

from dataProcessing import processData


app = Flask(__name__)


model = load_model('models\model.h5')


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods = ['POST'])
def predict(): 
    mushroom_features = [str(x) for x in request.form.values()]
    data = processData(mushroom_features)
    predictions = model.predict(data)

    return predictions
