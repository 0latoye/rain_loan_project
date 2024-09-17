from flask import Flask,request,render_template,redirect
import requests
import socket
import csv
# import loan_ml_model
from email.message import EmailMessage
import pandas as pd
import numpy as np
import smtplib
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import dice_ml
from dice_ml.utils import helpers
from lime.lime_tabular import LimeTabularExplainer
from counterfactual_params import dice_features
from counter_factual_generator import cf_generator, diff_return



def send_email(mail):
        
    # email = request.args.get("email")
    # email_list = email.split(",")
    # email_list.append(email.split(","))
    # for user_mail in email_list:
    msg = EmailMessage()
    recipients = mail
    username = "ola3toye@gmail.com"
    user_password = "jbio rtnc hjbz hgyq"

    msg['TO'] = recipients

    msg['From'] = "E-Global Financial Services"
    msg['Subject'] = "Status Of Loan Application"
    msg.set_content ('Congratulations!, your loan request has been approved. Please submit copies of original documents for verification to receive loan.')
            # msg.add_alternative(page1, subtype='html')

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(username, user_password)
            smtp.send_message(msg) 
    # print(f"Email sent successfully to {user_mail}")

    return f"Plese check your mail concerning the status of your application"

app = Flask(__name__)
categorical_data = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
def transformer(feat):
    if feat in categorical_data:
        if feat == 'Gender':
            return 'Gender_Male'
        elif feat == 'Married':
            return 'Married_Yes'
        elif feat == 'Dependents':
            return 'Dependents_1', 'Dependents_2', 'Dependents_3+'
        elif feat == 'Education':
            return 'Education_Not Graduate'
        elif feat == 'Self_Employed':
            return 'Self_Employed_Yes'
        elif feat == 'Property_Area':
            return 'Property_Area_Semiurban', 'Property_Area_Urban'
        else:
            return 'Invalid Data'

@app.route('/')
def home():
    return render_template("Loansphere.html")



@app.route('/index',methods = ["POST","GET"])
def index():
    global age_range, income_amount,loan_amount,repayment_period,occupation,user_email,profile_dict,scaled_df
    profile_dict = {}
    # user_profile = [age_range, income_amount,loan_amount,repayment_period,occupation,email]
    user_profile = ['Gender', 'Married', 'Dependents', 'Education',
                    'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area']

    if request.method == "POST":
        for feature in user_profile:
            profile_dict[feature] = int(request.form.get(feature))
           
            user_email = request.form.get("email")


    categorical_data = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    df_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Gender_Male', 'Married_Yes',
 'Dependents_1', 'Dependents_2', 'Dependents_3+', 'Education_Not Graduate', 'Self_Employed_Yes', 'Property_Area_Semiurban', 'Property_Area_Urban']
    
    print(f"#####################{user_email}############################################################################")
    return redirect("/profile_transformation")


@app.route('/profile_transformation')
def profile_transformation():
    global profile_df,profile_expansion,lime_profile_explained,result,expansion_elements,scaled_df,model
    cat_dict = {
         'Gender': ['Male', 'Female'],
    'Married': ['No', 'Yes'],
    'Dependents': ['0', '1', '2', '3+'],
    'Education': ['Graduate', 'Not Graduate'],
    'Self_Employed': ['No', 'Yes'],
    'Property_Area': ['Urban', 'Rural', 'Semiurban']
    }

    profile_expansion = {}

    expansion_elements = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Gender_Male', 'Married_Yes',
        'Dependents_1', 'Dependents_2', 'Dependents_3+',
        'Education_Not Graduate', 'Self_Employed_Yes',
        'Property_Area_Semiurban', 'Property_Area_Urban']
    
    one_hot_col = ['Gender_Male', 'Married_Yes', 'Dependents_1', 'Dependents_2', 'Dependents_3+', 'Education_Not Graduate', 'Self_Employed_Yes','Property_Area_Semiurban', 'Property_Area_Urban']
    cat_dict_map = {'Gender': {1:'Gender_Male', 0:'Female'},
        'Married': {0:'Single', 1:'Married_Yes'},
        'Dependents': {0:'No Dependents', 1:'Dependents_1',2: 'Dependents_2', 3:'Dependents_3+'},
        'Education': {1:'Graduate', 0:'Education_Not Graduate'},
        'Self_Employed': {0:'Unemployed', 1: 'Self_Employed_Yes'},
        'Property_Area': {0:'Rural',1:'Property_Area_Semiurban', 2:'Property_Area_Urban'}}
    
    for profile_feature in profile_dict:
        if profile_feature in expansion_elements:
            profile_expansion[profile_feature] = profile_dict[profile_feature]

    for cat_profile_feature in profile_dict:
        if cat_profile_feature in categorical_data:
                transformed = transformer(cat_profile_feature)
                if len(transformed) >3:
                        profile_expansion[transformed]=""
                else:
                        for cat_feature in transformed:
                                profile_expansion[cat_feature] = ""
    #Step1 -> Get empty values from profile expansion
    empty_values = [x for x in profile_expansion if profile_expansion[x] =='' ]
    option = []

    #Step2 -> get profile choice from profile
    for profile_key,parent_key in zip(profile_dict,cat_dict):
        #get user selection:
        choice = profile_dict[parent_key]
        map_val = cat_dict_map[parent_key][choice]
        option.append(map_val)
        # profile_expansion.update({:1})

    # update based on selected options in list
    for val in empty_values:
        if val in option:
            profile_expansion.update({val:1})
        else:
            profile_expansion.update({val:0})
    
   

    profile_df = pd.DataFrame(profile_expansion,columns = expansion_elements, index = [0])
    model = pickle.load(open('loan_rf_model.pkl','rb'))
    data_scaler = pickle.load(open('ml_data_scaling','rb'))
    scaled_df = data_scaler.transform(profile_df)
    result = model.predict(scaled_df)[0]
    # return result
    

    return redirect('/profile_page')
    # return profile_df
    

@app.route('/profile_page')
def profile_page():
    # model = pickle.load(open('loan_rf_model.pkl','rb'))
    # data_scaler = pickle.load(open('ml_data_scaling','rb'))
    # scaled_df = data_scaler.fit_transform(profile_df)
    # result = model.predict(scaled_df)

    #Adding lime visualization to profile prediction page result  for further tests
   


    profile_expansion.update({'result':result})
    print(profile_expansion)
    application_result = profile_expansion['result']
    if application_result == 'Y':
        # return 'Congratulations!, your loan request has been approved. Please submit copies of original documents for verification to receive loan.'
        return send_email(user_email)
    else:
        # print('Sorry :(, your loan request has been denied. We will get back to you shortly as to why and how you can improve your chances of recieving a loan.')
        # lime_profile_explained.show_in_notebook()
        # return "Sorry :(, your loan request has been denied. We will get back to you shortly as to why and how you can improve your chances of recieving a loan."
        # return lime_profile_explained.show_in_notebook()
        return redirect('/application_denied_page')
    

# @app.route('/application_denied_page')
# def denial_explainer():
#     dice_exp = pickle.load(open('loan_dice_exp.pkl','rb'))
#     featuretovary = ['Credit_History','Dependents_1', 'Dependents_2', 'Dependents_3+','Property_Area_Semiurban', 'Property_Area_Urban']
#     reject_explainer = dice_exp.generate_counterfactuals(profile_df,total_CFs=1,
#                                                         desired_class="opposite",
#                                                         permitted_range={
#                                                                 'Credit_History':[0,1],
#                                                                         'Education_Not Graduate':[0,1],
#                                                                         'Self_Employed_Yes':[0,1]},
#                                                         features_to_vary=featuretovary
#           counter_f_status = "N"
    # while counter_f_status == "N":
    #     considerations,counter_f_status = dice_features(profile_df,reject_explainer)
    #                                                                                     )

@app.route('/application_denied_page')
def denial_explainer():
    print(profile_dict)
    
    considerations = diff_return(profile_df)
    print(f' mmmmm{considerations}')
    one_hot_col = ['Gender_Male', 'Married_Yes', 'Dependents_1', 'Dependents_2', 'Dependents_3+', 'Education_Not Graduate', 'Self_Employed_Yes','Property_Area_Semiurban', 'Property_Area_Urban']
    cat_dict_map = {'Gender': ['Gender_Male'],
        'Married': ['Married_Yes'],
        'Dependents': ['Dependents_1','Dependents_2','Dependents_3+'],
        'Education': ['Education_Not Graduate'],
        'Self_Employed': ['Self_Employed_Yes'],
        'Property_Area': ['Rural','Property_Area_Semiurban','Property_Area_Urban']}
    select_map = {1:'Select',0:'Deselect'}

    #translate considerations into text
    recommended_response = []
    for value in considerations: 
        if value in one_hot_col:
            for p in cat_dict_map:
                for q in considerations:
                    if q in cat_dict_map[p]:
                        # print(f"Based on the ML recommendation you should {select_map[enw[q]]} the feature {q} instead for better chances")
                        recommended_response.append(f"Based on the ML recommendation you should {select_map[considerations[q]]} the feature {q} instead for better chances")
                    else:
                        continue
        else:
            # print(f"You can adjust the value of {i} with a recommended value of {enw[i]}")
            recommended_response.append(f"You can adjust the value of {value} with a recommended value of {considerations[value]}")
    
    return recommended_response
hostname = socket.gethostname()
ip_addr = socket.gethostbyname(hostname)

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 8080, debug=True)
    # app.run(host = ip_addr, port = 8080, debug=True)