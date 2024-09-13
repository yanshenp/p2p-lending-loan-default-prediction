import numpy as np
import pandas as pd
import streamlit as st 
from sklearn import preprocessing
import pickle

# load trained model
model = pickle.load(open('lgbm_finetuned_model.pkl', 'rb')) 

numerical_features = [
    'acc_open_past_24mths', 'annual_inc', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 
    'delinq_2yrs', 'dti', 'inq_last_6mths', 'loan_amnt', 'mort_acc', 'open_acc', 
    'percent_bc_gt_75', 'pub_rec', 'pub_rec_bankruptcies', 'revol_util', 'tax_liens', 
    'total_acc', 'emp_length_int', 'credit_history_years']

# mapping for the names of numerical features
numerical_features_names = {
    'acc_open_past_24mths': 'Accounts Open in Past 24 Months',
    'annual_inc': 'Annual Income',
    'avg_cur_bal': 'Average Current Balance',
    'bc_open_to_buy': 'Credit Card Open to Buy',
    'bc_util': 'Credit Card Utilization',
    'delinq_2yrs': 'Delinquencies in Last 2 Years',
    'dti': 'Debt-to-Income Ratio',
    'inq_last_6mths': 'Credit Inquiries in Last 6 Months',
    'loan_amnt': 'Loan Amount',
    'mort_acc': 'Number of Mortgage Accounts',
    'open_acc': 'Number of Open Accounts',
    'percent_bc_gt_75': 'Percent Credit Card Usage > 75%',
    'pub_rec': 'Public Records',
    'pub_rec_bankruptcies': 'Public Record Bankruptcies',
    'revol_util': 'Revolving Credit Utilization',
    'tax_liens': 'Tax Liens',
    'total_acc': 'Total Accounts',
    'emp_length_int': 'Employment Length (Years)',
    'credit_history_years': 'Credit History Length (Years)'
}

categorical_features = [
    'application_type_Individual', 'application_type_Joint App', 'home_ownership_MORTGAGE',
    'home_ownership_OWN', 'home_ownership_RENT', 'purpose_credit_card', 'purpose_debt_consolidation', 
    'region_West', 'verification_status_Not Verified', 'verification_status_Verified', 
    'fico_score_rating_Fair', 'fico_score_rating_Good', 'fico_score_rating_Very Good', 
    'fico_score_rating_Exceptional', 'term_ 36 months', 'term_ 60 months']

# categorical features with options
categorical_features_options = {
    'application_type': ['Individual','Joint App'],
    'home_ownership': ['MORTGAGE','OWN','RENT'],
    'purpose': ['credit_card','debt_consolidation','Others'],
    'region': ['West','East', 'North', 'South'],
    'verification_status': ['Not Verified','Verified'],
    'fico_score_rating': ['Fair','Good','Very Good','Exceptional'],
    'term': ['36 months','60 months']
    }

# code to accept whole numbers and apply normalization in the background
feature_min_max = {
    'acc_open_past_24mths': (0, 40),  
    'annual_inc': (10000, 1000000),   
    'avg_cur_bal': (0, 200000),
    'bc_open_to_buy': (0, 50000),
    'bc_util': (0, 100),
    'delinq_2yrs': (0, 10),
    'dti': (0, 40),
    'inq_last_6mths': (0, 15),
    'loan_amnt': (500, 40000),
    'mort_acc': (0, 20),
    'open_acc': (0, 50),
    'percent_bc_gt_75': (0, 100),
    'pub_rec': (0, 5),
    'pub_rec_bankruptcies': (0, 2),
    'revol_util': (0, 100),
    'tax_liens': (0, 1),
    'total_acc': (1, 80),
    'emp_length_int': (0, 50),
    'credit_history_years': (0, 50)
}

def normalize(value, min_val, max_val):
    """Normalize the input value using min-max scaling."""
    return (value-min_val) / (max_val-min_val)

def main():
    st.title("P2P Lending Loan Default Predictor")
    st.write("This is a simple app to demonstrate the prediction of the P2P Lending Loan Default predictive model.")
    
    num, cate = st.columns(2)

    with num:
        st.subheader("Numerical Features")
        number_inputs = {}
        for feature in numerical_features:
            display_name = numerical_features_names.get(feature, feature)
            number_inputs[feature] = st.number_input(f"Enter value for {display_name}:")
       
    with cate:
        st.subheader("Categorical Features")
        cate_inputs = {}
        for feature_group, options in categorical_features_options.items():
            selected_option = st.radio(f"Select {feature_group.replace('_', ' ').title()}", options)
            for option in options:
                cate_inputs[f"{feature_group}_{option}"] = (selected_option == option)

    normalized_inputs = []
    for feature in numerical_features:
        min_val, max_val = feature_min_max[feature]
        normalized_value = normalize(number_inputs[feature], min_val, max_val)
        normalized_inputs.append(normalized_value)

    # combine the inputs into a single list for prediction using model
    input_data = normalized_inputs + [int(cate_inputs.get(feature, False)) for feature in categorical_features]
    
    # prediction
    if st.button("Predict"):
        # Prepare input data for prediction
        input_data = np.array([input_data])  # Convert to 2D array for model input

        prediction_probability = model.predict_proba(input_data)[0][1]  # Probability of default (class 1)
        threshold = 0.5

        if prediction_probability > threshold:
            text = "Likely to default"
        else:
            text = "Less Likely to Default"
        
        st.write(f"**The predicted result is: {text} based on ({prediction_probability:.2f} probability)**")

if __name__=='__main__': 
    main()