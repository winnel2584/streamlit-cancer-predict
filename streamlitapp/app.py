import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import google.generativeai as genai


api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

def get_clean_data():
    # Load data to get statistics (means/max) for the sliders and scaling
    csv_path = os.path.join(os.path.dirname(__file__), "data.csv")
    df = pd.read_csv(csv_path)
    df.drop(columns=['Unnamed: 32', 'id'], inplace=True, errors='ignore')
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

def add_sidebar():
    st.sidebar.header("🔬 Key Nuclei Measurements")
    st.sidebar.write("Adjust the 6 most critical features:")
    
    df = get_clean_data()
    
    # We only create sliders for the Top 6 Features
    # The keys must match the exact column names in your CSV
    input_dict = {}
    
    # 1. Concave Points (Worst) - #1 Predictor
    input_dict['concave points_worst'] = st.sidebar.slider(
        "Concave Points (Worst)", 
        min_value=float(df['concave points_worst'].min()),
        max_value=float(df['concave points_worst'].max()),
        value=float(df['concave points_worst'].mean())
    )
    
    # 2. Area (Worst)
    input_dict['area_worst'] = st.sidebar.slider(
        "Area (Worst)",
        min_value=float(df['area_worst'].min()),
        max_value=float(df['area_worst'].max()),
        value=float(df['area_worst'].mean())
    )
    
    # 3. Radius (Worst)
    input_dict['radius_worst'] = st.sidebar.slider(
        "Radius (Worst)",
        min_value=float(df['radius_worst'].min()),
        max_value=float(df['radius_worst'].max()),
        value=float(df['radius_worst'].mean())
    )
    
    # 4. Perimeter (Worst)
    input_dict['perimeter_worst'] = st.sidebar.slider(
        "Perimeter (Worst)",
        min_value=float(df['perimeter_worst'].min()),
        max_value=float(df['perimeter_worst'].max()),
        value=float(df['perimeter_worst'].mean())
    )
    
    # 5. Concavity (Mean)
    input_dict['concavity_mean'] = st.sidebar.slider(
        "Concavity (Mean)",
        min_value=float(df['concavity_mean'].min()),
        max_value=float(df['concavity_mean'].max()),
        value=float(df['concavity_mean'].mean())
    )
    
    # 6. Texture (Mean)
    input_dict['texture_mean'] = st.sidebar.slider(
        "Texture (Mean)",
        min_value=float(df['texture_mean'].min()),
        max_value=float(df['texture_mean'].max()),
        value=float(df['texture_mean'].mean())
    )
    
    return input_dict

def get_full_input_array(user_inputs):
    """
    Takes the 6 user inputs and fills the remaining 24 features 
    with the average values from the dataset.
    """
    df = get_clean_data()
    X = df.drop(['diagnosis'], axis=1)
    
    # Get the mean of ALL columns
    mean_values = X.mean().to_dict()
    
    # Update the mean dict with the User's specific inputs
    for key, value in user_inputs.items():
        mean_values[key] = value
        
    # Ensure the order matches the model's expected input (X.columns)
    ordered_input = []
    for col in X.columns:
        ordered_input.append(mean_values[col])
        
    return np.array(ordered_input).reshape(1, -1), mean_values

def get_radar_chart(input_data_dict):
    # Simplified Radar Chart for just the 6 features
    
    # Normalize data for the chart (0 to 1 scale)
    df = get_clean_data()
    scaled_data = {}
    for key, value in input_data_dict.items():
        max_val = df[key].max()
        min_val = df[key].min()
        scaled_data[key] = (value - min_val) / (max_val - min_val)
        
    categories = list(scaled_data.keys())
    values = list(scaled_data.values())

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Selected Sample'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        title="Feature Impact Radar"
    )
    return fig

def generate_gemini_report(prediction_label, probability, input_data):
    """
    Uses Google Gemini to generate a medical report.
    """
    # DELETED THE "IF" TRAP HERE
    
    model = genai.GenerativeModel('gemini-flash-latest')
    
    # Construct the prompt
    prompt = f"""
    Act as a senior oncologist. You are analyzing a breast mass biopsy.
    
    The Machine Learning model has predicted the mass is: {prediction_label}
    Confidence Level: {probability:.2%}
    
    Here are the key measurements from the microscope:
    - Radius (Worst): {input_data['radius_worst']} (Normal avg is ~16.2)
    - Concave Points: {input_data['concave points_worst']} (Normal avg is ~0.11)
    - Area: {input_data['area_worst']} (Normal avg is ~880)
    - Texture: {input_data['texture_mean']}
    
    Write a short, professional 3-sentence medical summary explaining why this result is {prediction_label} based on the numbers provided. 
    Do not mention "AI" or "Model". Write it as a doctor's note.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Generation Error: {str(e)}"

def add_predictions(input_data):
    # Load Model
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
    
    with open(model_path, "rb") as f: model = pickle.load(f)
    with open(scaler_path, "rb") as f: scaler = pickle.load(f)

    # Prepare data (fill missing 24 features with means)
    input_array, full_dict = get_full_input_array(input_data)
    
    # Scale and Predict
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)
    prob = model.predict_proba(input_array_scaled)
    
    # Display Results
    st.subheader("Diagnostic Prediction")
    
    malignant_prob = prob[0][1]
    benign_prob = prob[0][0]
    
    col1, col2 = st.columns(2)
    
    if prediction[0] == 1:
        prediction_label = "MALIGNANT"
        final_prob = malignant_prob
        with col1:
            st.error(f"### {prediction_label}")
        with col2:
             st.metric("Confidence", f"{final_prob*100:.1f}%")
    else:
        prediction_label = "BENIGN"
        final_prob = benign_prob
        with col1:
            st.success(f"### {prediction_label}")
        with col2:
             st.metric("Confidence", f"{final_prob*100:.1f}%")

    st.write("---")
    
    # --- GEMINI AI SECTION ---
    st.subheader("📝 Generative AI Pathology Report")
    with st.spinner('Consulting AI Doctor...'):
        report = generate_gemini_report(prediction_label, final_prob, input_data)
        st.info(report)

def main():
    st.set_page_config(
        page_title="SWIES Breast Cancer AI",
        page_icon="🧬",   
        layout="wide",
        initial_sidebar_state="expanded" 
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main { background-color: #f5f5f5; }
        .stButton>button { width: 100%; }
        </style>
        """, unsafe_allow_html=True)

    input_data = add_sidebar()
    
    with st.container():
        st.title("🏥 Intelligent Breast Cancer Diagnosis System")
        st.write("Advanced AI diagnostics for SWIES Project. Adjust the biopsy parameters in the sidebar.")

    col1, col2 = st.columns([1, 1])

    with col1:
        add_predictions(input_data)
        
    with col2:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart, use_container_width=True)

if __name__ == '__main__':
    main()