import joblib as jb
import pandas as pd 
import numpy as np 

def predict_wine_quality():
    
    #loading the model and the model data 
    try:
        transformer_path = 'SupervisedLearning/Classification/SupportVectorMachines/wine_transformer.joblib'
        scaler_path = 'SupervisedLearning/Classification/SupportVectorMachines/wine_scaler.joblib' #given the model path instead of the names because of confusion and errors caused
        model_path = 'SupervisedLearning/Classification/SupportVectorMachines/wine_svm_model.joblib'
        
        transformer = jb.load(transformer_path) #fixed the typo that caused the error here
        scaler = jb.load(scaler_path) #fixed an error here
        model = jb.load(model_path)
    
    except FileNotFoundError:
        print("Models not found, double check the path and places")
        return #In exception return stops the function immediately 
    
    #user-input
    feature_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ]
    
    print("Enter wine character input")
    
    user_inputs = {}
    
    for feature in feature_names:
        while True:
            try:
                value = float(input(f"Enter the value of {feature}"))
                user_inputs[feature] = value
                break
            except ValueError:
                print("Invalid input. Please enter a number") #take kd of dictionaries, vectorisation by pandas in python, classes
    
    #creating a dataframe out of the given values for processings
    input_df = pd.DataFrame([user_inputs], columns=feature_names)
    print(user_inputs)
    
    # --- 3. Preprocessing -->yet to be done
    
    # First, apply the PowerTransformer and min_max scaler
    data_transformed = transformer.transform(input_df)

    data_scaled = scaler.transform(data_transformed)
    prediction = model.predict(data_scaled)
    
    print("\n--------------------------")
    print(f"Predicted Wine Quality: {prediction[0]}")
    print("--------------------------")

if __name__ == '__main__':
    predict_wine_quality()
    
    
    
    
    