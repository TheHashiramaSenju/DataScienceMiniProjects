import pandas as pd 
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.models import Dense, Input 

class OutlierDiagnoser:
    '''To learn normal relationships'''
    def __init__(self, features):
        self.features = features
        self.diagnostic_models = {}
        self.scalers = {} #know more about this class stuff
        
        print("Diagnosis Init")
    
    def fit(self, clean_df):
        '''Neural training to make model learn about what actually is normal'''
        for target_values in self.features:
            X = clean_df.drop(columns=[target_values])        
            y = clean_df[target_values]
            
            scaler = StandardScaler()
            scaledx = scaler.fit_transform(X)
            
            self.scalers[target_values] = scaler
            
            #neural network model for training them 
            model = Sequential([
                Input(shape=(scaledx[1, ])),
                Dense(16, activation = 'relu')
                Dense(8, activation='relu')
                Dense(1)
            ])
            
            #compiling the neural network model
            model.compile(optimizer = 'adam', loss = 'mean_squared_error')
            model.fit(scaledx, y, epochs=50, verbose=0)
            
            #saving the model for future use 
            self.diagnostic_models[target_values] = model
            
            print("model is now ready")
            
    def dmlp(self, outlier_row):
        '''
        For a single outlier row, calculates the prediction error for each feature
        and returns the name of the feature with the largest error.
        '''
        errors = {} #for collecting the errors
        
        for target_feature in self.features:
            
            model = self.diagnostic_models[target_feature]
            scaler = self.scalers[target_feature]
            
            actual_value = outlier_row[target_feature] #getting the values from the outlier row
            
            #features for value prediction
            prediction_features = outlier_row.drop(target_feature).values.reshape(1, -1)
            scaled_prediction_features = scaler.transform(prediction_features)
            
            #predict the alternate features 
            predicted_value = model.predict(scaled_prediction_features, verbose=0)[0][0]
            
            error = abs(predicted_value - actual_value)
            errors[target_feature] = error
        
        # Return the feature name with the highest error
        most_problematic_feature = max(errors, key=errors.get)
        return most_problematic_feature
            
def repair_outlier_feature(df_full, normal_df, outlier_row_index, column_to_fix):
    
    #reparing single featire for a specific outlier row
    
    features = [col for col in normal_df.columns if col != column_to_fix]

    X_train = normal_df[features]
    y_train = normal_df[column_to_fix]
    X_predict = df_full.loc[[outlier_row_index]][features]

    scaler = StandardScalerScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_predict_scaled = scaler.transform(X_predict)

    model = Sequential([
        Input(shape=(len(features),)),
        Dense(32, activation = 'relu'),
        Dense(16, activation = 'relu'),
        Dense(8, activation = 'relu')
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train_scaled, y_train, epochs=50, verbose=0)
    
    predicted_value = model.predict(X_predict_scaled, verbose=0)[0][0]
    
    print(f"--> Prediction for replacement: {predicted_value:.2f}")
    return predicted_value
