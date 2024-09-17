import dice_ml
from dice_ml.utils import helpers
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open('loan_rf_model.pkl','rb'))
data_scaler = pickle.load(open('ml_data_scaling','rb'))

def dice_features(user_profile,adjusted_df):
    if type(adjusted_df) == "dice_ml.counterfactual_explanations.CounterfactualExplanations":
        final = adjusted_df.cf_examples_list
        b = []
        for i in final:
            b.append(i.final_cfs_df)
            adjusted_df = b[0]

    counter_fact_status = None

    if 'Loan_Status_Y' in adjusted_df.columns:
        counter_fact_status = adjusted_df['Loan_Status_Y'].values[0]


    adjusted_df.drop(columns='Loan_Status_Y',inplace = True,errors = 'ignore')
    scaled_df = data_scaler.transform(adjusted_df)
    result = model.predict(scaled_df)[0]
    print(counter_fact_status)

    if result == "Y":

        # result = model.predict(scaled_df)[0]
            #convert both df to dictionaries
        user_prof_dict = user_profile.to_dict(orient = 'index')[user_profile.index[0]]
        adjusted_dict = adjusted_df.to_dict(orient = 'index')[adjusted_df.index[0]]
        
        print(user_prof_dict)
        print(adjusted_dict)
        columns = list(user_profile.columns)
            #Creating a dictionary to hold the difference in the chanegs from one df to the other
        difference = {}
            #iterating throught the user and adjusted dictionary profiles to identify the differences
        for feature in columns:    
            if user_prof_dict[feature] != adjusted_dict[feature]:
                    difference[feature] = adjusted_dict[feature]

        return (difference,counter_fact_status)
        # else:
        #      return f"Counter_Factual result returned as {counter_fact_status} which is not valid"
    else:
        #  return(result,counter_fact_status)
        return dice_features(user_profile,counter_fact_status)
    

# considerations = cf_profile_extractor(reject_explainer)
#     if type(considerations) != 'pandas.core.frame.DataFrame':
#         return considerations
#     else:
#         final = dice_features(profile_df,considerations)
#     # return "Sorry :(, your loan request has been denied. We will get back to you shortly as to why and how you can improve your chances of recieving a loan."
#         return final