import dice_ml
from dice_ml.utils import helpers
from networkx import difference
import pandas as pd
import numpy as np
import pickle

from sympy import re

model = pickle.load(open('loan_rf_model.pkl','rb'))
data_scaler = pickle.load(open('ml_data_scaling','rb'))
correct_dict = None
difference = None


def dice_features_extractor(dice_df,user_df,status):
    global difference
    dice_df.drop(columns = 'Loan_Status_Y',inplace = True)
    scaled_Data = data_scaler.transform(dice_df)
    result = model.predict(scaled_Data)[0]

    print(f"Model result:{result}, CounterfactualStatus:{status}")
    # print(dice_df.to_dict(orient = "index"))

    if result == "Y":
        user_prof_dict = user_df.to_dict(orient = 'index')[user_df.index[0]]
        adjusted_dict = dice_df.to_dict(orient = 'index')[dice_df.index[0]]
        
        # print(user_prof_dict)
        print(adjusted_dict)
        columns = list(user_df.columns)
            #Creating a dictionary to hold the difference in the chanegs from one df to the other
        difference = {}
            #iterating throught the user and adjusted dictionary profiles to identify the differences
        for feature in columns:    
            if user_prof_dict[feature] != adjusted_dict[feature]:
                    difference[feature] = adjusted_dict[feature]
        print(difference)
        correct_dict = difference
        return correct_dict
        # else:
        #     return "Invalid"
                 
        print("the diff: ",difference)
        
    else:
         cf_generator(user_df)





#Function to convert counterfactual to a dataframe then checks the prediction with the model to ensure it generates the required data else should rerun the cf_generator
def cf_to_df(counter_fact,user_profile):
    final = counter_fact.cf_examples_list
    b = []
    for i in final:
        b.append(i.final_cfs_df)
        adjusted_df = b[0]
    counter_fact_status = adjusted_df['Loan_Status_Y'].values[0]
    return dice_features_extractor(adjusted_df,user_profile,counter_fact_status)


# Function to generate counterfactual dataframe based on user profile
def cf_generator(user_profile):
    dice_exp = pickle.load(open('loan_dice_exp.pkl','rb'))
    featuretovary = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Credit_History','Dependents_1', 'Dependents_2', 'Dependents_3+','Property_Area_Semiurban', 'Property_Area_Urban']
    reject_explainer = dice_exp.generate_counterfactuals(user_profile,total_CFs=4,
                                                        desired_class="opposite",
                                                        permitted_range={
                                                                'Credit_History':[0,1],
                                                                'Education_Not Graduate':[0,1],
                                                                'Self_Employed_Yes':[0,1]},
                                                        features_to_vary=featuretovary
                                                                                                )
    return cf_to_df(reject_explainer,user_profile)



def diff_return(profile_df):
    cf_generator(profile_df)
    return difference
     