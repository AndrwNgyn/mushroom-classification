# Mushroom Classification
Simple classification model to determine whether mushrooms are poisonous or not.

- This model was created via ANNs, including Adam optimizer, dropout layers, and callbacks. 
- Data trained is sourced from Kaggle, and can be found at https://www.kaggle.com/datasets/uciml/mushroom-classification 

# Running the app 
- **This app is hosted on Heroku!** You can locate the application on https://mushroom-classifier-anguyen.herokuapp.com/
- Note that the website may require 30 seconds - 1 minute to load 

## Prerequisites 
1. Python 3.6 
   - This app requires that your machine has python 3.0+ installed on it. you can refer to this url https://www.python.org/downloads/ to download python. Once you have python downloaded and installed, you will need to setup PATH variables (if you want to run python program directly, detail instructions are below in *how to run software section*). To do that check this: https://www.pythoncentral.io/add-python-to-path-python-is-not-recognized-as-an-internal-or-external-command/.  
   - Setting up PATH variable is optional as you can also run program without it and more instruction are given below on this topic. 
2. Second and easier option is to download anaconda and use its anaconda prompt to run the commands. To install anaconda check this url     https://www.anaconda.com/download/
3. You will also need to download and install below 3 packages after you install either python or anaconda from the steps above
   - Sklearn (scikit-learn)
   - numpy
   - tensorflow
   
  - if you have chosen to install python 3.6 then run below commands in command prompt/terminal to install these packages
   ```
   pip install -U scikit-learn
   pip install numpy
   pip install tensorflow
   ```
   - if you have chosen to install anaconda then run below commands in anaconda prompt to install these packages
   ```
   conda install -c scikit-learn
   conda install -c anaconda numpy
   conda install -c anaconda tensorflow
   ```   

## Running the App & Usage (Locally)
- Simply run the ```App.py``` file, and locate http://127.0.0.1:5000 on your desired browser
- Follow the directions on the projected website to determine mushroom edibility 
- When finished, run ```ctrl + C``` on your terminal to terminate the server
