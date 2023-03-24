import numpy as np
import pickle
import json
import os
import CONFIG

class Wine_Prediction():
    def load_raw(self):
        with open(CONFIG.MODEL_PATH,"rb") as model_file:
            self.model = pickle.load(model_file)

        with open(CONFIG.COLUMNS_PATH,"r") as col_file:
            self.column = json.load(col_file)
        return "Sucess RAW DAtA Loaded"


    def __init__(self):
        print(os.getcwd())

    def predict_wine_quality(self,data):
        self.load_raw()
        self.data = data

        user_input = np.zeros(len(self.column["Columns"]))
        
        fixed_acidity        = self.data["fixed_acidity"]
        volatile_acidity     = self.data["volatile_acidity"]
        citric_acid          = self.data["citric_acid"]
        residual_sugar       = self.data["residual_sugar"]
        chlorides            = self.data["chlorides"]
        free_sulfur_dioxide  = self.data["free_sulfur_dioxide"]
        total_sulfur_dioxide = self.data["total_sulfur_dioxide"]
        density              = self.data["density"]
        pH                   = self.data["pH"]
        sulphates            = self.data["sulphates"]
        alcohol              = self.data["alcohol"]

        user_input[0] = float(fixed_acidity)
        user_input[1] = float(volatile_acidity)
        user_input[2] = float(citric_acid)
        user_input[3] = float(residual_sugar)
        user_input[4] = float(chlorides)
        user_input[5] = float(free_sulfur_dioxide)
        user_input[6] = float(total_sulfur_dioxide)
        user_input[7] = float(density)
        user_input[8] = float(pH)
        user_input[9] = float(sulphates)
        user_input[10] = float(alcohol)

        print(f"{user_input=}")
        print(len(user_input))

        Quality = self.model.predict([user_input])[0]
        print(f"Quality = {Quality}")

        if Quality == 0:
            print("Bad Wine")
            return "Bad Wine"
        else:
            print("Good Wine")
            return "Good Wine"
        
        
    

if __name__=="__main__":
    pred_obj = Wine_Prediction()
    pred_obj.load_raw()