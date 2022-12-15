### This file is dedicated to processing input quantitative mushroom data into scaled quantitative data for processing with ANN model
### HTML Website gathers input data via dropdown menus and sends data to model in a form of an array

import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv("data_csvs/mushrooms.csv")

cap_shape={"b":"bell","c":"conical","x":"convex","f":"flat","k":"knobbed","s":"sunken"}
data["cap-shape"]=data["cap-shape"].replace(cap_shape)

cap_surface={"f": "fibrous", "g": "grooves","y":"scaly","s": "smooth"}
data["cap-surface"]=data["cap-surface"].replace(cap_surface)

cap_color={"n":"brown","b":"buff","c":"cinnamon","g":"gray","r":"green","p":"pink","u":"purple","e":"red","w":"white","y":"yellow"}
data["cap-color"]=data["cap-color"].replace(cap_color)

odor={"a":"almond","l":"anise","c":"creosote","y":"fishy","f":"foul","m":"musty","n":"none","p":"pungent","s":"spicy"}
data["odor"]=data["odor"].replace(odor)

gill_attachment={"a":"attached","f":"free", "d":"descending", "n":"notched"}
data["gill-attachment"]=data["gill-attachment"].replace(gill_attachment)

gill_spacing={"c":"close","w":"crowded", "d":"distant"}
data["gill-spacing"]=data["gill-spacing"].replace(gill_spacing)

gill_size={"b":"broad","n":"narrow"}
data["gill-size"]=data["gill-size"].replace(gill_size)

gill_color={"k":"black","b":"buff","n":"brown","h":"chocolate","g":"gray","r":"green","o":"orange","p":"pink","u":"purple","e":"red","w":"white","y":"yellow"}
data["gill-color"]=data["gill-color"].replace(gill_color)

stalk_shape={"t":"tapering","e":"enlarging"}
data["stalk-shape"]=data["stalk-shape"].replace(stalk_shape)

stalk_root={"b":"bulbous","c":"club","e":"equal","z":"rhizomorphs","r":"rooted","?":"missing"}
data["stalk-root"]=data["stalk-root"].replace(stalk_root)

stalk_surface_above_ring={"s":"smooth","k":"silky","f":"fibrous","y":"scaly"}
data["stalk-surface-above-ring"]=data["stalk-surface-above-ring"].replace(stalk_surface_above_ring)
data["stalk-surface-below-ring"]=data["stalk-surface-below-ring"].replace(stalk_surface_above_ring)

stalk_color_above_ring={"n":"brown","b":"buff","c":"cinnamon","g":"gray","p":"pink","e":"red","w":"white","y":"yellow","o":"orange"}
data["stalk-color-above-ring"]=data["stalk-color-above-ring"].replace(stalk_color_above_ring)
data["stalk-color-below-ring"]=data["stalk-color-below-ring"].replace(stalk_color_above_ring)

veil_type={"p":"partial","u":"universal"} 
data["veil-type"]=data["veil-type"].replace(veil_type)

veil_color={"n":"brown","o":"orange","w":"white","y":"yellow"} 
data["veil-color"]=data["veil-color"].replace(veil_color)

ring_number= {"n":"none","o":"one","t":"two"}
data["ring-number"]=data["ring-number"].replace(ring_number)

ring_type={"c":"cobwebby","e":"evanescent","f":"flaring","l":"large","n":"none","p":"pendant","s":"sheathing","z":"zone"}
data["ring-type"]=data["ring-type"].replace(ring_type)

spore_print_color= {"k":"black","n":"brown","b":"buff","h":"chocolate","r":"green","o":"orange","u":"purple","w":"white","y":"yellow"}
data["spore-print-color"]=data["spore-print-color"].replace(spore_print_color)

population={"a":"abundant","c":"clustered","n":"numerous","s":"scattered","v":"several","y":"solitary"}
data["population"]=data["population"].replace(population)

habitat={"g":"grasses","l":"leaves","m":"meadows","p":"paths","u":"urban","w":"waste","d":"woods"}
data["habitat"]=data["habitat"].replace(habitat)

bruises={"t":"bruises","f":"no"}
data["bruises"]=data["bruises"].replace(bruises)

classification={"e":"edible","p":"poisonous"}
data["class"]=data["class"].replace(classification)

data = data[["class", "cap-color", "odor", "gill-color", "population", "habitat"]]

def convertToDataFrame(input_list):
    mushroom_template = pd.DataFrame(columns = ['cap-color',
                                                'odor',
                                                'gill-color',
                                                'population',
                                                'habitat'])

    mushroom_template.loc[len(mushroom_template)] = input_list
    return mushroom_template

def encodeData(input):
    '''
    function takes array and scales it according to preset 'mushrooms.csv' file via LabelEncoder
    '''
    le = LabelEncoder()

    for col in input.columns:
        le.fit(data[col])
        input[col] = le.transform(input[col])
    return input

def scaleData(input):
    model_location = os.path.join('models', 'scaled.save')
    scaler = joblib.load(model_location)
    input = scaler.transform(input)
    return input

def processData(input):
    data = convertToDataFrame(input)
    data = encodeData(data)
    data = scaleData(data)
    return data

if __name__ == "__main__":
  
    test_input = ['yellow', 'almond',
       'black', 'numerous', 'grasses']

    test_input2 = ['bell', 'fibrous', 'brown', 'bruises', 'almond', 'attached', 'close', 'broad', 'black', 'enlarging', 'bulbous', 'fibrous', 'fibrous', 'brown', 'brown', 'partial', 'brown', 'one', 'cobwebby', 'black', 'abundant', 'grasses']
    
    
    print(convertToDataFrame(test_input))
    print(processData(test_input))
    # print(processData(test_input2))
    
    
    # print(data['habitat'].unique())


    '''
    test_input = pd.DataFrame({'cap-shape': ['convex'],
                                'cap-surface': ['smooth'],
                                'cap-color': ['yellow'],
                                'bruises': ['bruises'],
                                'odor': ['almond'],
                                'gill-attachment': ['free'],
                                'gill-spacing': ['close'],
                                'gill-size': ['broad'],
                                'gill-color': ['black'],
                                'stalk-shape': ['enlarging'],
                                'stalk-root': ['club'],
                                'stalk-surface-above-ring': ['smooth'],
                                'stalk-surface-below-ring': ['smooth'],
                                'stalk-color-above-ring': ['white'],
                                'stalk-color-below-ring': ['white'],
                                'veil-type': ['partial'],
                                'veil-color': ['white'],
                                'ring-number': ['one'],
                                'ring-type': ['pendant'],
                                'spore-print-color': ['brown'],
                                'population': ['numerous'],
                                'habitat': ['grasses']})
    
    processed_data = processData(test_input)                                
    print(processed_data)
   
    mushroom_model = load_model('model.h5')
    
    prediction = mushroom_model.predict(processed_data)
    print(prediction)
    
    
    data = encodeData(data)
    x = data.drop('class', axis = 1)
    y = data["class"].values

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state= 100)
    
    
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
    
    
    mushroom_model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )
    
    model_loss = pd.DataFrame(mushroom_model.history.history)
    model_loss.plot()
    
    predictions = mushroom_model.predict(X_test)
    predictions = (predictions > 0.5)
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test, predictions))    
    '''
