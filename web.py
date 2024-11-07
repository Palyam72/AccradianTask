import streamlit as st
from pycaret.classification import load_model

# Set up model dictionary
model_list = ['lr', 'dt', 'rf', 'et',  'lightgbm', 'svm', 'nb']
model_dict = {model_name: f"{model_name}.pkl" for model_name in model_list}

# Function to predict transaction using a selected model
def predict_transaction(model_name, input_data):
    if model_name in model_dict:
        model = load_model(model_dict[model_name])  # Load model from PyCaret
        prediction = model.predict(pd.DataFrame([input_data]))
        return prediction[0]
    else:
        st.error("Selected model not found.")
        return None

st.title("Fraud Detection Model")

# Model selection dropdown
selected_model_name = st.selectbox("Please select the model that you want to use", model_list)
st.divider()

st.header("Please Provide the Specific Information as Input")
# Input fields for transaction features
step = st.number_input("Step", min_value=0, format="%d")
transaction_type = st.selectbox("Type", options=["CASH_OUT", "PAYMENT", "DEBIT", "TRANSFER", "CASH_IN"])
amount = st.number_input("Amount", min_value=0.0, format="%.2f")
name_orig = st.text_input("Name Orig")
oldbalance_org = st.number_input("Old Balance Orig", min_value=0.0, format="%.2f")
newbalance_orig = st.number_input("New Balance Orig", min_value=0.0, format="%.2f")
name_dest = st.text_input("Name Dest")
oldbalance_dest = st.number_input("Old Balance Dest", min_value=0.0, format="%.2f")
newbalance_dest = st.number_input("New Balance Dest", min_value=0.0, format="%.2f")

st.divider()

# Prediction checkbox
predict_btn = st.checkbox("Predict")

# Prediction Logic
if predict_btn:
    # Prepare input data as a dictionary
    input_data = {
        'step': step,
        'type': transaction_type,
        'amount': amount,
        'nameOrig': name_orig,
        'oldbalanceOrg': oldbalance_org,
        'newbalanceOrig': newbalance_orig,
        'nameDest': name_dest,
        'oldbalanceDest': oldbalance_dest,
        'newbalanceDest': newbalance_dest,
    }

    # Predict using the selected model
    result = predict_transaction(selected_model_name, input_data)
    if result is not None:
        st.success(f"Prediction: {'Fraudulent' if result == 1 else 'Not Fraudulent'}")
