
import pandas as pd
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier,plot_tree,export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,roc_auc_score,classification_report

import shap

import warnings
warnings.filterwarnings('ignore')

#Importing data set
# os.getcwd()

df = pd.read_csv(r"kaggle data files\archive\Training Dataset.csv")



#isolating target column and dropping irrelevant columns
target = df['Loan_Status']
df.drop(columns = ['Loan_ID','Loan_Status'],inplace=True)



#dealing with null values
for column in df.columns:
    if df[column].dtypes == 'O':
        mode_val = df[column].mode().iloc[0]
        df.fillna({column: mode_val},inplace=True)
    else:
        df.fillna({column:df[column].median()},inplace=True)




#getting categorical columns and corresponding values
categorical_columns = {}

for column in df.columns:
    if df[column].dtype == 'O':
        categorical_columns[column] = list(df[column].unique())
# categorical_columns


# ------------------------------------------------------------------------------------------------------------------------------------------
#ML MODELLING

#using one hot encoding to split multi categorical values to binary equivalent then drop other due to redundancy
df = pd.get_dummies(df,columns=categorical_columns.keys(),dtype=int,drop_first=True)


#Scaling data to uniform values
data_scaler = StandardScaler()
data_scaled = data_scaler.fit_transform(df)


# Splitting into test and train set 
X_train, X_test, y_train, y_test = train_test_split(data_scaled,target,train_size=0.2,random_state=1994)

# # Using RandomForsetClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train,y_train)
y_pred = rf_model.predict(X_test)

#final combinination of of both train and test split to be used 

X_train = data_scaled
y_train = target 
rf_model.fit(X_train,y_train)

def prediction(*profile):
    profile = pd.get_dummies(profile,columns=categorical_columns.keys(),dtype=int,drop_first=True)
    scaled_profile = data_scaler.transform(profile)
    prediction_result = rf_model.predict(scaled_profile)[0]
    if prediction_result == 'Y':
              return "Based on your credentails submitted, your loan request will be approved"
    else:
              return "Based on your credentails submitted, your loan request has been denied"
       

#ACCURACY SCORE 
# print(f"The accuracy score is: {accuracy_score(y_test,y_pred)}")
# print(f"The Confusion matrix values are: {confusion_matrix(y_test,y_pred)}")

# # setting positive class label
# print(f"The precision score value is: {precision_score(y_test,y_pred,pos_label='Y')}")
# print(f"The recall score is: {recall_score(y_test,y_pred,pos_label='Y')}")
# print(f"The f1 score is: {f1_score(y_test,y_pred,pos_label='Y')}")


# john = df.sample()



# john


# rf_model.predict(john)


# fi = pd.DataFrame(rf_model.feature_importances_).T
# fi.columns = df.columns
# fi


# plot = sns.barplot(data = fi,orient='h')


# df_summed = fi.sum(axis=0)

# # Sort columns based on summed values
# df_sorted1 = df_summed.sort_values()

# # Create barplot
# plt.figure(figsize=(10, 6))
# sns.barplot(x=df_sorted1.values, y=df_sorted1.index, palette="viridis")

# # Add labels and title
# plt.xlabel('Sum of Values')
# plt.ylabel('Feature')
# plt.title('Sum of Normalized Values')

# # Show plot
# plt.show()


# #top 5
# # Sort columns based on summed values
# df_sorted1 = df_summed.sort_values(ascending=False)

# # Select top 5 columns
# top_5_columns1 = df_sorted1.head(5)
# top_5_columns1


# print(classification_report(y_pred,y_test))


# explainer_rf = shap.Explainer(rf_model)
# shap_val = explainer_rf.shap_values(X_test)


# shap.summary_plot(shap_val,X_test)

# -------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------

# # # Using DecisionTreeClassifier


# dt_model = DecisionTreeClassifier()


# dt_model.fit(X_train,y_train)


# y_pred = dt_model.predict(X_test)


# y_pred


# # print(f"The accuracy score is: {accuracy_score(y_test,y_pred)}")
# # print(f"The Confusion matrix values are: {confusion_matrix(y_test,y_pred)}")

# # # setting positive class label
# # print(f"The precision score value is: {precision_score(y_test,y_pred,pos_label='Y')}")
# # print(f"The recall score is: {recall_score(y_test,y_pred,pos_label='Y')}")
# # print(f"The f1 score is: {f1_score(y_test,y_pred,pos_label='Y')}")


# # roc_auc_score(y_test,y_pred)


# #Test
# john = df.sample()


# john


# john_dict = john.to_dict(orient='index')


# john_dict.values()


# dt_model.predict(john)


# dt_model.predict_proba(john)


# dt_model.feature_importances_


# #".T" used to transpose table
# fi = pd.DataFrame(dt_model.feature_importances_).T
# fi.columns = df.columns
# fi





# sns.set(style='darkgrid')
# plot = sns.barplot(data = fi,orient='h', palette = 'rainbow')
# # fig.bar_label(fig.containers[0], fmt = '%.1f')


# df_summed = fi.sum(axis=0)

# # Sort columns based on summed values
# df_sorted = df_summed.sort_values()

# # Create barplot
# plt.figure(figsize=(10, 6))
# sns.barplot(x=df_sorted.values, y=df_sorted.index, palette="viridis")

# # Add labels and title
# plt.xlabel('Sum of Values')
# plt.ylabel('Feature')
# plt.title('Sum of Normalized Values')

# # Show plot
# plt.show()


# #top 5
# # Sort columns based on summed values
# df_sorted = df_summed.sort_values(ascending=False)

# # Select top 5 columns
# top_5_columns2 = df_sorted.head(5)
# top_5_columns2


# top_5_columns1


# # ApplicantIncome, CoapplicantIncome, LoanAmount,
# #        Loan_Amount_Term, Credit_History, Gender_Male, Married_Yes,
# #        Dependents_1, Dependents_2, Dependents_3,
# #        Education_Not_Graduate, Self_Employed_Yes,
# #        Property_Area_Semiurban, Property_Area_Urban


# def user_profile(data):
#        # rf_model.predict(pd.DataFrame(john_dict).T)
#        prediction = rf_model.predict(data)[0]
#        if prediction == 'Y':
#               return "Based on your credentails submitted, your loan request will be approved"
#        else:
#               return "Based on your credentails submitted, your loan request has been denied"
       



# user_profile(john)


# rf_model.predict(john)[0]


# from random import randint,random


# val = random()


# val


# profile_comp = list(df.columns)


# user_prof = {}
# for x in profile_comp:
#    min_val = int(df[x].min())
#    max_val = int(df[x].max())
#    user_prof[x] = randint(min_val,max_val)

# user_prof


# df.head(5)


# profiles = ['john','jane','max','steve','paul','beatrice','victor','sam','emmanuel','raul']


# complete_profile = {}
# for name in profiles:
#     user_prof = {}
#     for x in profile_comp:
#         min_val = int(df[x].min())
#         max_val = int(df[x].max())
#         user_prof[x] = randint(min_val,max_val)
#     complete_profile[name] = user_prof
# # complete_profile


# for loan_status in complete_profile:
#     profile_df =  pd.DataFrame.from_dict(complete_profile[loan_status],orient='index').T
#     # profile_df
#     print(loan_status,user_profile(profile_df))
#     # test = pd.DataFrame.from_dict(complete_profile[loan_status])


# complete_profile['victor']


# new = df.sample().to_dict(orient = 'index')
# new
# # rf_model.predict(new)


# john


# df.describe()


# df


# import shap


# # using the shap library to explain results 
# shap_values = shap.TreeExplainer(dt_model,df).shap_values(df)


# for i,j in zip(range(len(df.columns)),df.columns):
#     print("Shap value for feature",j,":",shap_values[i])


# explainer_dt = shap.Explainer(dt_model)
# shap_val = explainer_dt.shap_values(X_test)


# print(classification_report(y_pred,y_test))


# shap_val


