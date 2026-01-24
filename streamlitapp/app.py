import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import google.generativeai as genai

# 1. Configuration
st.set_page_config(
    page_title="SWIES Breast Cancer AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup Gemini AI
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
except KeyError:
    st.error("Error: GOOGLE_API_KEY not found in Streamlit secrets.")

def get_clean_data():
    # Load data to get statistics (min/max/mean) for the sliders
    csv_path = os.path.join(os.path.dirname(__file__), "data.csv")
    try:
        df = pd.read_csv(csv_path)
        df.drop(columns=['Unnamed: 32', 'id'], inplace=True, errors='ignore')
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
        return df
    except FileNotFoundError:
        st.error("Error: 'data.csv' not found. Please ensure it is in the same directory.")
        return None

def add_sidebar():
    st.sidebar.header("🔬 Tumor Measurements")
    st.sidebar.write("Adjust the 10 key features below:")
    
    df = get_clean_data()
    if df is None: return {}
    
    input_dict = {}
    
    # Group 1: 'Worst' Features (Most Critical)
    st.sidebar.subheader("Worst-Case Features")
    
    # 1. Area (Worst)
    input_dict['area_worst'] = st.sidebar.slider(
        "Area (Worst)",
        min_value=float(df['area_worst'].min()),
        max_value=float(df['area_worst'].max()),
        value=float(df['area_worst'].mean())
    )
    
    # 2. Concave Points (Worst)
    input_dict['concave points_worst'] = st.sidebar.slider(
        "Concave Points (Worst)", 
        min_value=float(df['concave points_worst'].min()),
        max_value=float(df['concave points_worst'].max()),
        value=float(df['concave points_worst'].mean())
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

    # 5. Concavity (Worst)
    input_dict['concavity_worst'] = st.sidebar.slider(
        "Concavity (Worst)",
        min_value=float(df['concavity_worst'].min()),
        max_value=float(df['concavity_worst'].max()),
        value=float(df['concavity_worst'].mean())
    )
    
    # Group 2: 'Mean' Features
    st.sidebar.subheader("Average Features")
    
    # 6. Concave Points (Mean)
    input_dict['concave points_mean'] = st.sidebar.slider(
        "Concave Points (Mean)",
        min_value=float(df['concave points_mean'].min()),
        max_value=float(df['concave points_mean'].max()),
        value=float(df['concave points_mean'].mean())
    )
    
    # 7. Perimeter (Mean)
    input_dict['perimeter_mean'] = st.sidebar.slider(
        "Perimeter (Mean)",
        min_value=float(df['perimeter_mean'].min()),
        max_value=float(df['perimeter_mean'].max()),
        value=float(df['perimeter_mean'].mean())
    )

    # 8. Concavity (Mean)
    input_dict['concavity_mean'] = st.sidebar.slider(
        "Concavity (Mean)",
        min_value=float(df['concavity_mean'].min()),
        max_value=float(df['concavity_mean'].max()),
        value=float(df['concavity_mean'].mean())
    )
    
    # 9. Area (Mean)
    input_dict['area_mean'] = st.sidebar.slider(
        "Area (Mean)",
        min_value=float(df['area_mean'].min()),
        max_value=float(df['area_mean'].max()),
        value=float(df['area_mean'].mean())
    )
    
    # 10. Radius (Mean)
    input_dict['radius_mean'] = st.sidebar.slider(
        "Radius (Mean)",
        min_value=float(df['radius_mean'].min()),
        max_value=float(df['radius_mean'].max()),
        value=float(df['radius_mean'].mean())
    )
    
    return input_dict

def get_radar_chart(input_data_dict):
    df = get_clean_data()
    
    # Scale data 0-1 for plotting
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
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title="Tumor Profile (Normalized)"
    )
    return fig

def generate_gemini_report(prediction_label, probability, input_data):
    model = genai.GenerativeModel('gemini-flash-latest')
    
    prompt = f"""
    Act as a senior oncologist. Analyze this breast mass biopsy result.
    
    **AI Prediction:** {prediction_label}
    **Confidence:** {probability:.2%}
    
    **Key Measurements:**
    - Area (Worst): {input_data['area_worst']}
    - Concave Points (Worst): {input_data['concave points_worst']}
    - Radius (Worst): {input_data['radius_worst']}
    - Concavity (Mean): {input_data['concavity_mean']}
    
    Write a professional, empathetic 3-sentence medical summary explaining this result. 
    Focus on what the "Worst" values imply about cell irregularity.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Generation Error: {str(e)}"

def add_predictions(input_data):
    # Load Model and Scaler using JOBLIB
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
    except FileNotFoundError:
        st.error("Error: 'model.pkl' or 'scaler.pkl' not found.")
        return

    # CRITICAL: Arrange inputs in the EXACT order used during training
    feature_order = [
        'area_worst', 'concave points_worst', 'concave points_mean', 'radius_worst', 
        'perimeter_worst', 'perimeter_mean', 'concavity_mean', 'area_mean', 
        'concavity_worst', 'radius_mean'
    ]
    
    # Convert dict to ordered list
    ordered_values = [input_data[feature] for feature in feature_order]
    
    # Convert to numpy array and reshape
    input_array = np.array(ordered_values).reshape(1, -1)
    
    # Scale inputs
    input_scaled = scaler.transform(input_array)
    
    # Predict
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)
    
    malignant_prob = prob[0][1]
    benign_prob = prob[0][0]
    
    st.subheader("Diagnostic Prediction")
    col1, col2 = st.columns(2)
    
    if prediction[0] == 1:
        prediction_label = "MALIGNANT"
        final_prob = malignant_prob
        with col1:
            st.error(f"### {prediction_label}")
        with col2:
            st.metric("Confidence", f"{final_prob:.1%}")
    else:
        prediction_label = "BENIGN"
        final_prob = benign_prob
        with col1:
            st.success(f"### {prediction_label}")
        with col2:
            st.metric("Confidence", f"{final_prob:.1%}")

    st.write("---")
    
    # Gemini Report
    st.subheader("📝 Generative AI Pathology Report")
    with st.spinner('Consulting AI Doctor...'):
        report = generate_gemini_report(prediction_label, final_prob, input_data)
        st.info(report)

def main():
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

    if input_data:
        with col1:
            add_predictions(input_data)
        with col2:
            radar_chart = get_radar_chart(input_data)
            st.plotly_chart(radar_chart, use_container_width=True)

if __name__ == '__main__':
    main()