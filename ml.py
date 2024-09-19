import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from azure.storage.blob import BlobServiceClient
from io import BytesIO

# Load the trained model
try:
    loaded_model = pickle.load(open("D:/Loan prediction/trained_model.sav", 'rb'))
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the DataFrame for reference
df = pd.read_csv(r"D:\Loan prediction\Data\Training Data.csv\Training Data.csv")

# Azure Blob Storage details
connection_string = "DefaultEndpointsProtocol=https;AccountName=loandefaulterdata;AccountKey=n3D079hN+ftpjiTDYPlAeA1/6RsVjYc0ujj/8Zs81LoLlMqducYVb/Q1WFyQWwWT/QeieQbnjm80+AStFnpojQ==;EndpointSuffix=core.windows.net"
container_name = "data"
blob_path = "Training Data.csv"

# Blob Storage Manager class
class BlobStorageManager:
    def __init__(self, connection_string):
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    def upload(self, container_name, dataframe, blob_path):
        try:
            csv_data = dataframe.to_csv(index=False).encode('utf-8')  # Convert to bytes
            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_path)
            blob_client.upload_blob(csv_data, overwrite=True)
            st.success(f"DataFrame uploaded to {blob_path} in container {container_name}.")
        except Exception as e:
            st.error(f"Blob upload failed: {e}")

    def download(self, container_name, blob_path):
        try:
            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_path)
            blob_data = blob_client.download_blob().readall()
            dataframe = pd.read_csv(BytesIO(blob_data)) if blob_path.endswith('.csv') else None
            return dataframe
        except Exception as e:
            st.error(f"Blob download failed: {e}")
        return None

# Encode and Predict Function
def encode_and_predict(input_data):
    binary_mapping = {'married': 1, 'single': 0, 'owned': 1, 'rented': 0, 'yes': 1, 'no': 0}
    binary_columns = ['married_single', 'House_Ownership', 'Car_Ownership']
    multi_cat_columns = ['Profession', 'CITY', 'STATE']
    
    numerical_input = input_data[:3]
    binary_input = input_data[3:6]
    multi_cat_input = input_data[6:9]
    other_numerical_input = input_data[9:]

    numerical_input = [float(val) for val in numerical_input]
    other_numerical_input = [float(val) for val in other_numerical_input]

    encoded_binary = [binary_mapping[val] for val in binary_input]

    professions = df['Profession'].astype(str).unique().tolist() + ['Others']
    cities = df['CITY'].astype(str).unique().tolist() + ['Others']
    states = df['STATE'].astype(str).unique().tolist()

    encoders = {
        'Profession': LabelEncoder().fit(professions),
        'CITY': LabelEncoder().fit(cities),
        'STATE': LabelEncoder().fit(states)
    }

    if multi_cat_input[0] not in professions:
        encoders['Profession'].classes_ = np.append(encoders['Profession'].classes_, multi_cat_input[0])
    
    if multi_cat_input[1] not in cities:
        encoders['CITY'].classes_ = np.append(encoders['CITY'].classes_, multi_cat_input[1])

    encoded_multi_cat = [
        encoders['Profession'].transform([multi_cat_input[0]])[0],
        encoders['CITY'].transform([multi_cat_input[1]])[0],
        encoders['STATE'].transform([multi_cat_input[2]])[0]
    ]

    encoded_input = numerical_input + encoded_binary + encoded_multi_cat + other_numerical_input
    encoded_input_as_numpy_array = np.asarray(encoded_input, dtype=np.float32)
    encoded_input_reshaped = encoded_input_as_numpy_array.reshape(1, -1)
    
    prediction = loaded_model.predict(encoded_input_reshaped)
    
    # Return prediction result for display and storage
    risk_flag = int(prediction[0])  # 0 for not a risk, 1 for risk
    display_text = 'The person is not a risk' if risk_flag == 0 else 'The person is a risk'
    
    return risk_flag, display_text

# Main function for Streamlit app
def main():
    st.set_page_config(page_title="Loan Default Detector", page_icon=":moneybag:", layout="wide")
    st.title('Loan Default Detector')
    col1, col2 = st.columns(2)
    with col1:
        Income = st.text_input('Yearly Income')
        Experience = st.text_input('Enter your total experience in years')
        House_Ownership = st.selectbox('Select your home status:', ['owned', 'rented'])
        profession_variable = sorted(df['Profession'].unique().tolist() + ['Others'])
        Profession = st.selectbox('Enter your Profession', options=profession_variable)
        if Profession == 'Others':
            Profession = st.text_input('Please specify your profession:')
        state_names = sorted(df['STATE'].unique().tolist())
        STATE = st.selectbox('Enter the state of Residence', options=state_names)
        CURRENT_HOUSE_YRS = st.text_input('Enter number of years in current residence')
    
    with col2:
        Age = st.text_input('Enter your Age')
        married_single = st.selectbox('Select your marital status:', ['married', 'single'])
        Car_Ownership = st.radio('Do you own a car:', ['yes', 'no'])
        CITY = st.text_input('Enter Your CITY')
        CURRENT_JOB_YRS = st.text_input('Enter number of years in current job')
    
    if st.button('Get Results'):
        input_data = [Income, Age, Experience, married_single, House_Ownership, Car_Ownership, Profession, CITY, STATE, CURRENT_JOB_YRS, CURRENT_HOUSE_YRS]
        risk_flag, loandef = encode_and_predict(input_data)
        st.success(loandef)
        blob = BlobStorageManager(connection_string)
        df_azure = blob.download(container_name, blob_path)

        # Ensure df_azure is not None and has data
        if df_azure is not None and not df_azure.empty:
            df_id = df_azure['Id'].max() + 1 if 'Id' in df_azure.columns else 1
            input_data_with_id = [df_id] + input_data + [risk_flag]  # Save 0 or 1 in Risk_Flag
            new_data = pd.DataFrame([input_data_with_id], columns=['Id', 'Income', 'Age', 'Experience', 'married_single', 'House_Ownership', 'Car_Ownership', 'Profession', 'CITY', 'STATE', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS', 'Risk_Flag'])
            df_azure = pd.concat([df_azure, new_data], ignore_index=True)
            blob.upload(container_name, df_azure, blob_path)
            st.success("Data uploaded successfully to Azure Blob Storage.")
        else:
            st.error("Failed to load the existing data from Azure Blob Storage.")

if __name__ == '__main__':
    main()