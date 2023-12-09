import os
import sys
import pandas as pd
import numpy as np
import pickle


from src.exception import CustomException
from src.logger import logging 
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            #train model
            model.fit(x_train,y_train)

            #predict Testing data
            y_test_pred = model.predict(x_test)

            # Get R2 scores for train and test data
            # Train_model_score = r2_score(y_test,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        logging.info('Exception occure during model training')
        raise CustomException(e,sys)



def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occured at load_object function utils')
        raise CustomException(e,sys)    



