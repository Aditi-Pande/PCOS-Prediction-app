import streamlit as st
import pickle as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def get_clean_data():
    
    data = pd.read_csv("dataset/pcos_data.csv")

    data[data.isnull().any(axis=1)].head()

    return data

def get_scaled_values(input_dict):
    data = get_clean_data()

    x = data.drop(["PCOS_Diagnosis"], axis = 1)

    scaled_dict = {}
  
    for key, value in input_dict.items():
        max_val = x[key].max()
        min_val = x[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    
    return scaled_dict


def add_sidebar():

    st.sidebar.header("Patient Measurements")
  
    data = get_clean_data()

    input_dict = {}

    input_dict["Age"] = st.sidebar.slider(
        "Age",
        min_value=int(data["Age"].min()),
        max_value=int(data["Age"].max()),
        value=int(data["Age"].mean())
    )

    input_dict["BMI"] = st.sidebar.slider(
        "BMI",
        min_value=float(data["BMI"].min()),
        max_value=float(data["BMI"].max()),
        value=float(data["BMI"].mean())
    )

    input_dict["Menstrual_Irregularity"] = st.sidebar.selectbox(
        "Menstrual Irregularity",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    input_dict["Testosterone_Level(ng/dL)"] = st.sidebar.slider(
        "Testosterone Level (ng/dL)",
        min_value=float(data["Testosterone_Level(ng/dL)"].min()),
        max_value=float(data["Testosterone_Level(ng/dL)"].max()),
        value=float(data["Testosterone_Level(ng/dL)"].mean())
    )

    input_dict["Antral_Follicle_Count"] = st.sidebar.slider(
        "Antral Follicle Count",
        min_value=int(data["Antral_Follicle_Count"].min()),
        max_value=int(data["Antral_Follicle_Count"].max()),
        value=int(data["Antral_Follicle_Count"].mean())
    )

    return input_dict

def get_prediction(input_data):
    model = pickle.load(open("model/model.pkl", 'rb'))
    scaler = pickle.load(open("model/scaler.pkl", 'rb'))

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)

    prediction_prob = model.predict_proba(input_array_scaled)[0][1]
    return prediction_prob


def get_gauge_chart(input_data):
    risk_score = get_prediction(input_data) * 100  # assuming output is between 0 and 1

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        title={'text': "PCOS Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))

    fig.update_layout(
        title_text="Predicted Risk of PCOS",
    )

    return fig


def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", 'rb'))
    scaler = pickle.load(open("model/scaler.pkl", 'rb'))

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("PCOS Diagnosis :")

    probability = round(model.predict_proba(input_array_scaled)[0][1] * 100, 2)

    color = "red" if probability > 50 else "green"
    st.markdown(f"<span style='color:{color}; font-weight:bold;'>Probability of being Positive: {probability}%</span>", unsafe_allow_html=True)

    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

def main():
    st.set_page_config(
        page_title="PCOS Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    input_data = add_sidebar()

    with st.container():
        st.title("PCOS Predictor")
        st.write("Please connect this app to your pathology lab to help diagnose PCOS (Polycystic Ovary Syndrome) from patient data. " \
            "This app uses a machine learning model to predict whether a patient is likely to have PCOS based on clinical measurements received " \
            "from your lab. You can also update the measurements manually using the sliders in the sidebar.")
  
    col1, col2 = st.columns([4,1])
  
    with col1:
        gauge_chart = get_gauge_chart(input_data)
        st.plotly_chart(gauge_chart)
    with col2:
        add_predictions(input_data)

if __name__ == "__main__":
    main()