import streamlit as st
from PIL import Image
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import pickle
import joblib
import os
import sys
import xgboost
from xgboost import XGBClassifier
import shap

sys.path.append("pre_processing.py")
from pre_processing import pre_processing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import category_encoders as ce
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder


# Load the model :
model_folder = os.path.join(os.path.dirname(__file__), '..', 'model')
model_path = os.path.join(model_folder, 'scoring_model.pkl')
with open(model_path, 'rb') as file:
    model = joblib.load(file) # or pickle.load(file)


# import x_train for shapley value :
df_folder = os.path.join(os.path.dirname(__file__), '..', 'DATA')
df_path = os.path.join(df_folder, 'df_train.csv')
with open(df_path, 'r') as file:
    df_train = pd.read_csv(file, sep=";")


# Load the image : 
image_folder = os.path.join(os.path.dirname(__file__), '..', 'PPT_et_rapport')
image_path = os.path.join(image_folder, 'customer_churn3.webp')
image = Image.open(image_path) 


# Predict price :
target = "AFTERGRACE_FLAG"
x_train = df_train.drop([target], axis=1)
y_train = df_train[target]

list_col_to_drop = ['PASS_AFTERGRACE_IND_M1', 'INC_DURATION_MINS_M1', 'INC_PROP_OPE2_MIN_M3', 'OUT_DURATION_MINS_M1', 'OUT_DURATION_MINS_M3',
                    'OUT_INT_DURATION_MINS_M3', 'OUT_VMACC_NO_CALLS_M1', 'INC_OUT_PROP_DUR_MIN_M1']
list_cat_col_OHE = ['CUSTOMER_GENDER']
list_cat_col_TE =  ['marque']

a = pre_processing()
x_train = a.pre_processing(df=x_train, train=True, categorical_var_OHE= list_cat_col_OHE,
                           categorical_var_OrdinalEncoding={}, categorical_var_TE=list_cat_col_TE, target=y_train,
                           continious_var=[], encoding_type_cont=StandardScaler())
 

def predict_churner(feature_dict) :
    df_for_pred = pd.DataFrame(feature_dict)

    # Feature engineering :
    df_for_pred["FLAG_RECHARGE_M1"] = df_for_pred["RECENCY_OF_LAST_RECHARGE"].apply(lambda x : 1 if 0 <= x <= 31 else 0)
    df_for_pred["FLAG_RECHARGE_M2"] = df_for_pred["RECENCY_OF_LAST_RECHARGE"].apply(lambda x : 1 if 32 <= x <= 62 else 0)
    df_for_pred["FLAG_RECHARGE_M3"] = df_for_pred["RECENCY_OF_LAST_RECHARGE"].apply(lambda x : 1 if 63 <= x <= 92 else 0)
    df_for_pred["FLAG_RECHARGE_PLUS_M3"] = df_for_pred["RECENCY_OF_LAST_RECHARGE"].apply(lambda x : 1 if x >= 93 else 0)

    for index, row in df_for_pred.iterrows():
        if row["BALANCE_M1"] > row["BALANCE_M2"] and row["BALANCE_M2"] > row["BALANCE_M3"] :
            df_for_pred.at[index, "AVERAGE_MULTIPLE_RECHARGE_M1_M2_M3"] = 1

        else :
            df_for_pred.at[index, "AVERAGE_MULTIPLE_RECHARGE_M1_M2_M3"] = 0

    for index, row in df_for_pred.iterrows() :
        if row["INC_DURATION_MINS_M1"] + row["INC_PROP_SMS_CALLS_M1"] == 0 :
            df_for_pred.at[index, "FLAG_IN_M1"] = 0
        else :
            df_for_pred.at[index, "FLAG_IN_M1"] = 1
        if row["INC_DURATION_MINS_M2"] + row["INC_PROP_SMS_CALLS_M2"] == 0 :
            df_for_pred.at[index, "FLAG_IN_M2"] = 0
        else :
            df_for_pred.at[index, "FLAG_IN_M2"] = 1
        if row["INC_DURATION_MINS_M3"] + row["INC_PROP_SMS_CALLS_M3"] == 0 :
            df_for_pred.at[index, "FLAG_IN_M3"] = 0
        else :
            df_for_pred.at[index, "FLAG_IN_M3"] = 1

    for index, row in df_for_pred.iterrows() :
        if row["OUT_DURATION_MINS_M1"] + row["OUT_SMS_NO_M1"] + row["OUT_INT_DURATION_MINS_M1"] + row["OUT_888_DURATION_MINS_M1"] + row["OUT_VMACC_NO_CALLS_M1"] == 0 :
            df_for_pred.at[index, "FLAG_OUT_M1"] = 0
        else :
            df_for_pred.at[index, "FLAG_OUT_M1"] = 1
        if row["OUT_DURATION_MINS_M2"] + row["OUT_SMS_NO_M2"] + row["OUT_INT_DURATION_MINS_M2"] + row["OUT_888_DURATION_MINS_M2"] + row["OUT_VMACC_NO_CALLS_M2"] == 0 :
            df_for_pred.at[index, "FLAG_OUT_M2"] = 0
        else :
            df_for_pred.at[index, "FLAG_OUT_M2"] = 1
        if row["OUT_DURATION_MINS_M3"] + row["OUT_SMS_NO_M3"] + row["OUT_INT_DURATION_MINS_M3"] + row["OUT_888_DURATION_MINS_M3"] + row["OUT_VMACC_NO_CALLS_M3"] == 0 :
            df_for_pred.at[index, "FLAG_OUT_M3"] = 0
        else :
            df_for_pred.at[index, "FLAG_OUT_M3"] = 1

    for index, row in df_for_pred.iterrows() :
        if row["CONTRACT_TENURE_DAYS"] > 730 :
            df_for_pred.at[index, "OLD_CONTRACT"] = 1
        else :
            df_for_pred.at[index,"OLD_CONTRACT"] = 0

    # 1st feature selection :
    df_for_pred.drop(list_col_to_drop, axis=1, inplace=True)

    # Encode only cat features :
    df_for_pred = a.pre_processing(df=df_for_pred, train=False, categorical_var_OHE= list_cat_col_OHE,
                                   categorical_var_OrdinalEncoding={}, categorical_var_TE=list_cat_col_TE, target=None,
                                   continious_var=[], encoding_type_cont=StandardScaler())

    # 2nd feature selection RFE :
    #df_for_pred = df_for_pred[list(selector.get_feature_names_out())]
    
    # Reorder the feature for xgboost :
    df_for_pred = df_for_pred[model.get_booster().feature_names]

    return {"Churner" : (model.predict(df_for_pred))[0],
            "Proba 0 (Not churner)": [round(elem, 2) for elem in list(model.predict_proba(df_for_pred)[0])][0],
            "Proba 1 (Churner)": [round(elem, 2) for elem in list(model.predict_proba(df_for_pred)[0])][1]
           } , df_for_pred


# Main function :
def main():
    # Title :
    st.title('Who are the customers churners ? :chart_with_upwards_trend:')
    
    # Image :
    st.image(image, caption='Customer churner')
    

    with st.sidebar :
        
        # Date :
        now = datetime.now()+ timedelta(hours=1)
        time = now.strftime("%H:%M:%S")
        st.write("Heure à Paris : ",time)
        
        
        CUSTOMER_AGE = st.text_input("Customer Age (number)", "30")
        CUSTOMER_GENDER = st.radio("Customer Gender", ('male', 'female'))
        marque = st.text_input("Phone brand", "nokia")
        CONTRACT_TENURE_DAYS = st.slider("CONTRACT TENURE DAYS", 100, 2000) 
        AVERAGE_CHARGE_6M = st.slider("AVERAGE CHARGE 6M", 0, 2000)
        FAILED_RECHARGE_6M = st.slider("FAILED RECHARGE 6M", 0, 50)
        AVERAGE_RECHARGE_TIME_6M = st.slider("AVERAGE RECHARGE TIME 6M", 0, 300)
        BALANCE_M3 = st.slider("BALANCE M3", 0, 2000) 
        BALANCE_M2 = st.slider("BALANCE M2", 0, 2000) 
        BALANCE_M1 = st.slider("BALANCE M1", 0, 2000) 
        FIRST_RECHARGE_VALUE = st.slider("FIRST RECHARGE VALUE", 25, 200)
        LAST_RECHARGE_VALUE = st.slider("LAST RECHARGE VALUE", 25, 200)
        TIME_TO_GRACE = st.slider("TIME TO GRACE", -200, 200) 
        
        TIME_TO_AFTERGRACE = st.slider("TIME TO AFTERGRACE", -200, 200)
        
        RECENCY_OF_LAST_RECHARGE = st.slider("RECENCY OF LAST RECHARGE", 0, 300)
        
        TOTAL_RECHARGE_6M = st.text_input("TOTAL RECHARGE 6M (between 0 and 15000)", "150")
        
        NO_OF_RECHARGES_6M = st.slider("NO OF RECHARGES 6M", 0, 300) 
        
        ZERO_BALANCE_IND_M3 = 1 if st.checkbox("ZERO BALANCE IND M3") else 0
        ZERO_BALANCE_IND_M2 = 1 if st.checkbox("ZERO BALANCE IND M2") else 0
        ZERO_BALANCE_IND_M1 = 1 if st.checkbox("ZERO BALANCE IND M1") else 0
        
        PASS_GRACE_IND_M3 = 1 if st.checkbox("PASS GRACE IND M3") else 0
        PASS_GRACE_IND_M2 = 1 if st.checkbox("PASS GRACE IND M2") else 0
        PASS_GRACE_IND_M1 = 1 if st.checkbox("PASS GRACE IND M1") else 0
        
        PASS_AFTERGRACE_IND_M3 = 1 if st.checkbox("PASS AFTERGRACE IND M3") else 0
        PASS_AFTERGRACE_IND_M2 = 1 if st.checkbox("PASS AFTERGRACE IND M2") else 0
        PASS_AFTERGRACE_IND_M1 = 1 if st.checkbox("PASS AFTERGRACE IND M1") else 0
        
        DATA_FLAG = 1 if st.checkbox("DATA FLAG") else 0
        INT_FLAG = 1 if st.checkbox("INT FLAG") else 0
        ROAM_FLAG = 1 if st.checkbox("ROAM FLAG") else 0
        
        NUM_HANDSET_USED_6M = st.slider("NUM HANDSET USED 6M", 0, 200)
        
        INC_DURATION_MINS_M3 = st.slider("INC DURATION MINS M3", 0, 6500)
        INC_DURATION_MINS_M2 = st.slider("INC DURATION MINS M2", 0, 6500)
        INC_DURATION_MINS_M1 = st.slider("INC DURATION MINS M1", 0, 6500)
        
        INC_PROP_SMS_CALLS_M3 = st.slider("INC PROP SMS CALLS M3", 0, 50) 
        INC_PROP_SMS_CALLS_M2 = st.slider("INC PROP SMS CALLS M2", 0, 50) 
        INC_PROP_SMS_CALLS_M1 = st.slider("INC PROP SMS CALLS M1", 0, 50) 

        
        INC_PROP_OPE1__MIN_M1 = st.slider(label="INC PROP OPE1 MIN M1", min_value=0.00, max_value=1.00, step=0.01)
        INC_PROP_OPE1__MIN_M2 = st.slider(label="INC PROP OPE1 MIN M2", min_value=0.00, max_value=1.00, step=0.01)
        INC_PROP_OPE1__MIN_M3 = st.slider(label="INC PROP OPE1 MIN M3", min_value=0.00, max_value=1.00, step=0.01)
        
        INC_PROP_OPE2_MIN_M1 = st.slider(label="INC PROP OPE2 MIN M1", min_value=0.00, max_value=1.00, step=0.01)
        INC_PROP_OPE2_MIN_M2 = st.slider(label="INC PROP OPE2 MIN M2", min_value=0.00, max_value=1.00, step=0.01)
        INC_PROP_OPE2_MIN_M3 = st.slider(label="INC PROP OPE2 MIN M3", min_value=0.00, max_value=1.00, step=0.01)
        
        INC_PROP_FIXED_MIN_M1 = st.slider(label="INC PROP FIXED MIN M1", min_value=0.00, max_value=1.00, step=0.01)
        INC_PROP_FIXED_MIN_M2 = st.slider(label="INC PROP FIXED MIN M2", min_value=0.00, max_value=1.00, step=0.01)
        INC_PROP_FIXED_MIN_M3 = st.slider(label="INC PROP FIXED MIN M3", min_value=0.00, max_value=1.00, step=0.01)
        
        OUT_DURATION_MINS_M1 = st.slider("OUT DURATION MINS M1", 0, 3000)
        OUT_DURATION_MINS_M2 = st.slider("OUT DURATION MINS M2", 0, 3000)
        OUT_DURATION_MINS_M3 = st.slider("OUT DURATION MINS M3", 0, 3000)
        
        OUT_SMS_NO_M1 = st.slider("OUT SMS NO M1", 0, 1000)
        OUT_SMS_NO_M2 = st.slider("OUT SMS NO M2", 0, 1000)
        OUT_SMS_NO_M3 = st.slider("OUT SMS NO M3", 0, 1000)
        
        OUT_INT_DURATION_MINS_M1 = st.slider("OUT INT DURATION MINS M1", 0, 500)
        OUT_INT_DURATION_MINS_M2 = st.slider("OUT INT DURATION MINS M2", 0, 500)
        OUT_INT_DURATION_MINS_M3 = st.slider("OUT INT DURATION MINS M3", 0, 500)
        
        OUT_888_DURATION_MINS_M1 = st.slider("OUT 888 DURATION MINS M1", 0, 200)
        OUT_888_DURATION_MINS_M2 = st.slider("OUT 888 DURATION MINS M2", 0, 200)
        OUT_888_DURATION_MINS_M3 = st.slider("OUT 888 DURATION MINS M3", 0, 200)
        
        OUT_VMACC_NO_CALLS_M1 = st.slider("OUT_VMACC_NO_CALLS_M1", 0, 2000)
        OUT_VMACC_NO_CALLS_M2 = st.slider("OUT_VMACC_NO_CALLS_M2", 0, 1000)
        OUT_VMACC_NO_CALLS_M3 = st.slider("OUT_VMACC_NO_CALLS_M3", 0, 300)
        
        OUT_PROP_SMS_CALLS_M1 = st.slider("OUT_PROP_SMS_CALLS_M1", 0, 200)
        OUT_PROP_SMS_CALLS_M2 = st.slider("OUT_PROP_SMS_CALLS_M2", 0, 200)
        OUT_PROP_SMS_CALLS_M3 = st.slider("OUT_PROP_SMS_CALLS_M3", 0, 200)
        
        OUT_PROP_OPE1__MIN_M1 = st.slider(label="OUT_PROP_OPE1__MIN_M1", min_value=0.00, max_value=1.00, step=0.01)
        OUT_PROP_OPE1__MIN_M2 = st.slider(label="OUT_PROP_OPE1__MIN_M2", min_value=0.00, max_value=1.00, step=0.01)
        OUT_PROP_OPE1__MIN_M3 = st.slider(label="OUT_PROP_OPE1__MIN_M3", min_value=0.00, max_value=1.00, step=0.01)
        
        OUT_PROP_OPE2_MIN_M1 = st.slider(label="OUT_PROP_OPE2_MIN_M1", min_value=0.00, max_value=1.00, step=0.01)
        OUT_PROP_OPE2_MIN_M2 = st.slider(label="OUT_PROP_OPE2_MIN_M2", min_value=0.00, max_value=1.00, step=0.01)
        OUT_PROP_OPE2_MIN_M3 = st.slider(label="OUT_PROP_OPE2_MIN_M3", min_value=0.00, max_value=1.00, step=0.01)
        
        OUT_PROP_FIXED_MIN_M1 = st.slider(label="OUT_PROP_FIXED_MIN_M1", min_value=0.00, max_value=1.00, step=0.01)
        OUT_PROP_FIXED_MIN_M2 = st.slider(label="OUT_PROP_FIXED_MIN_M2", min_value=0.00, max_value=1.00, step=0.01)
        OUT_PROP_FIXED_MIN_M3 = st.slider(label="OUT_PROP_FIXED_MIN_M3", min_value=0.00, max_value=1.00, step=0.01)
        
        INC_OUT_PROP_DUR_MIN_M1 = st.slider("INC_OUT_PROP_DUR_MIN_M1", 0, 3000)
        INC_OUT_PROP_DUR_MIN_M2 = st.slider("INC_OUT_PROP_DUR_MIN_M2", 0, 3000)
        INC_OUT_PROP_DUR_MIN_M3 = st.slider("INC_OUT_PROP_DUR_MIN_M3", 0, 3000)
                    


    feature_dict = {'CUSTOMER_AGE': [float(CUSTOMER_AGE)], 'CONTRACT_TENURE_DAYS': [CONTRACT_TENURE_DAYS], 'AVERAGE_CHARGE_6M': [AVERAGE_CHARGE_6M],
                    'FAILED_RECHARGE_6M': [FAILED_RECHARGE_6M], 'AVERAGE_RECHARGE_TIME_6M': [AVERAGE_RECHARGE_TIME_6M], 'BALANCE_M3': [BALANCE_M3], 
                    'BALANCE_M2': [BALANCE_M2], 'BALANCE_M1': [BALANCE_M1],'FIRST_RECHARGE_VALUE': [FIRST_RECHARGE_VALUE], 'LAST_RECHARGE_VALUE': [LAST_RECHARGE_VALUE], 
                    'TIME_TO_GRACE': [TIME_TO_GRACE], 'TIME_TO_AFTERGRACE': [TIME_TO_AFTERGRACE], 'RECENCY_OF_LAST_RECHARGE': [RECENCY_OF_LAST_RECHARGE],
                    'TOTAL_RECHARGE_6M': [float(TOTAL_RECHARGE_6M)],'NO_OF_RECHARGES_6M': [NO_OF_RECHARGES_6M], 'ZERO_BALANCE_IND_M3': [ZERO_BALANCE_IND_M3], 
                    'ZERO_BALANCE_IND_M2': [ZERO_BALANCE_IND_M2],'ZERO_BALANCE_IND_M1': [ZERO_BALANCE_IND_M1], 'PASS_GRACE_IND_M3': [PASS_GRACE_IND_M3], 
                    'PASS_GRACE_IND_M2': [PASS_GRACE_IND_M2], 'PASS_GRACE_IND_M1': [PASS_GRACE_IND_M1], 'PASS_AFTERGRACE_IND_M3': [PASS_AFTERGRACE_IND_M3], 
                    'PASS_AFTERGRACE_IND_M2': [PASS_AFTERGRACE_IND_M2], 'PASS_AFTERGRACE_IND_M1' : [PASS_AFTERGRACE_IND_M1], 'DATA_FLAG': [DATA_FLAG], 
                    'INT_FLAG': [INT_FLAG], 'ROAM_FLAG': [ROAM_FLAG], 'NUM_HANDSET_USED_6M': [NUM_HANDSET_USED_6M], 'INC_DURATION_MINS_M1':[INC_DURATION_MINS_M1], 
                    'INC_DURATION_MINS_M2': [INC_DURATION_MINS_M2], 'INC_DURATION_MINS_M3': [INC_DURATION_MINS_M3], 'INC_PROP_SMS_CALLS_M1': [INC_PROP_SMS_CALLS_M1], 
                    'INC_PROP_SMS_CALLS_M2': [INC_PROP_SMS_CALLS_M2], 'INC_PROP_SMS_CALLS_M3': [INC_PROP_SMS_CALLS_M3], 
                    'INC_PROP_OPE1__MIN_M1': [INC_PROP_OPE1__MIN_M1], 'INC_PROP_OPE1__MIN_M2': [INC_PROP_OPE1__MIN_M2], 'INC_PROP_OPE1__MIN_M3': [INC_PROP_OPE1__MIN_M3],
                    'INC_PROP_OPE2_MIN_M1': [INC_PROP_OPE2_MIN_M1], 'INC_PROP_OPE2_MIN_M2': [INC_PROP_OPE2_MIN_M2], 'INC_PROP_OPE2_MIN_M3': [INC_PROP_OPE2_MIN_M3], 
                    'INC_PROP_FIXED_MIN_M1': [INC_PROP_FIXED_MIN_M1], 'INC_PROP_FIXED_MIN_M2': [INC_PROP_FIXED_MIN_M2], 'INC_PROP_FIXED_MIN_M3': [INC_PROP_FIXED_MIN_M3],
                    "OUT_DURATION_MINS_M1":[OUT_DURATION_MINS_M1], 'OUT_DURATION_MINS_M2': [OUT_DURATION_MINS_M2], 'OUT_DURATION_MINS_M3': [OUT_DURATION_MINS_M3],
                    'OUT_SMS_NO_M1': [OUT_SMS_NO_M1],'OUT_SMS_NO_M2': [OUT_SMS_NO_M2],'OUT_SMS_NO_M3': [OUT_SMS_NO_M3],'OUT_INT_DURATION_MINS_M1': [OUT_INT_DURATION_MINS_M1],
                    'OUT_INT_DURATION_MINS_M2': [OUT_INT_DURATION_MINS_M2], 'OUT_INT_DURATION_MINS_M3': [OUT_INT_DURATION_MINS_M3], 
                    'OUT_888_DURATION_MINS_M1': [OUT_888_DURATION_MINS_M1],'OUT_888_DURATION_MINS_M2': [OUT_888_DURATION_MINS_M2],'OUT_888_DURATION_MINS_M3': [OUT_888_DURATION_MINS_M3],
                    'OUT_VMACC_NO_CALLS_M1': [OUT_VMACC_NO_CALLS_M1], 'OUT_VMACC_NO_CALLS_M2': [OUT_VMACC_NO_CALLS_M2],'OUT_VMACC_NO_CALLS_M3': [OUT_VMACC_NO_CALLS_M3],
                    'OUT_PROP_SMS_CALLS_M1': [OUT_PROP_SMS_CALLS_M1],'OUT_PROP_SMS_CALLS_M2': [OUT_PROP_SMS_CALLS_M2], 'OUT_PROP_SMS_CALLS_M3': [OUT_PROP_SMS_CALLS_M3],
                    'OUT_PROP_OPE1__MIN_M1': [OUT_PROP_OPE1__MIN_M1],'OUT_PROP_OPE1__MIN_M2': [OUT_PROP_OPE1__MIN_M2],'OUT_PROP_OPE1__MIN_M3': [OUT_PROP_OPE1__MIN_M3],
                    'OUT_PROP_OPE2_MIN_M1': [OUT_PROP_OPE2_MIN_M1],'OUT_PROP_OPE2_MIN_M2': [OUT_PROP_OPE2_MIN_M2], 'OUT_PROP_OPE2_MIN_M3': [OUT_PROP_OPE2_MIN_M3], 
                    'OUT_PROP_FIXED_MIN_M1': [OUT_PROP_FIXED_MIN_M1], 'OUT_PROP_FIXED_MIN_M2': [OUT_PROP_FIXED_MIN_M2],'OUT_PROP_FIXED_MIN_M3': [OUT_PROP_FIXED_MIN_M3],
                    'INC_OUT_PROP_DUR_MIN_M1': [INC_OUT_PROP_DUR_MIN_M1],'INC_OUT_PROP_DUR_MIN_M2': [INC_OUT_PROP_DUR_MIN_M2],'INC_OUT_PROP_DUR_MIN_M3': [INC_OUT_PROP_DUR_MIN_M3],
                    'CUSTOMER_GENDER': [CUSTOMER_GENDER], "marque": [marque]}


    # Predict button :
    if st.button('Predict score') :
        prediction = predict_churner(feature_dict)
        
        # display the dataframe :
        st.dataframe(prediction[1])
        
        # Print the prediction :
        col1, col2, col3 = st.columns(3)
        col1.metric(label = "Churner", value = prediction[0]["Churner"])
        col2.metric(label = "Proba 0 (Not churner)", value = prediction[0]["Proba 0 (Not churner)"])
        col3.metric(label = "Proba 1 (Churner)", value = prediction[0]["Proba 1 (Churner)"])
        
        # Explain the prediction with shapley method :
        st.subheader('Explanation of the prediction')
        df_p_h = prediction[1]
        explainer = shap.Explainer(model.predict, x_train)
        shap_values = explainer(df_p_h)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        shap.plots.waterfall(shap_values[0], max_display=20)
        st.pyplot(fig)
        
        
    
# __name__ :
if __name__ == '__main__' :
    main()