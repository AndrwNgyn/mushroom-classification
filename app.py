# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 19:07:57 2022

@author: andrwngyn
"""

from flask import Flask, request, render_template
# from keras.models import load_model
import tensorflow as tf
from tensorflow import keras

from dataProcessing import processData


app = Flask(__name__)


# model = load_model('models\model.h5')
model = tf.keras.models.load_model('.\models\model.h5')


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods = ['POST'])
def predict(): 
    mushroom_features = [str(x) for x in request.form.values()]
    data = processData(mushroom_features)
    predictions = model.predict(data)

    if predictions[0][0] > 0.5:
        result = 'poisonous'
    else:
        result = 'edible'

    print(mushroom_features)
    print(predictions[0][0])
    prediction_result = f'Mushroom is {result}'
    print(prediction_result)
    return render_template('index.html', prediction_text = prediction_result)



if __name__ == "__main__":
    app.run()