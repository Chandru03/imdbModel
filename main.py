import streamlit as st
import pandas as pd
import subprocess
import sys
import joblib
import requests
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from io import BytesIO

# Define GitHub raw file URLs
GITHUB_BASE_URL = "https://raw.githubusercontent.com/Chandru03/imdbModel/master/"
DATASET_URL = GITHUB_BASE_URL + "imdb_top_1000.csv"
MODEL_URL = GITHUB_BASE_URL + "tabnet_imdb.zip"
SCALER_URL = GITHUB_BASE_URL + "scaler.pkl"


# Function to download files
def download_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        st.error(f"Failed to download {url}")
        return None


# Load dataset
df = pd.read_csv(DATASET_URL)

# Load trained model
model = TabNetRegressor()
model_path = download_file(MODEL_URL)
if model_path:
    model.load_model(model_path)

# Load scaler
scaler_path = download_file(SCALER_URL)
if scaler_path:
    scaler = joblib.load(scaler_path)

# Configure page
st.set_page_config(page_title="🍿 Movie Rating Predictor", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .big-font { font-size:50px !important; }
    .result-box { padding: 20px; border-radius: 10px; margin: 10px 0; }
    .st-emotion-cache-1v0mbdj { width: 100%; }
</style>
""", unsafe_allow_html=True)

# Split layout into two columns
col1, col2 = st.columns([1, 2], gap="large")

# Left Column - Inputs
with col1:
    st.header("🎬 Movie Details")

    with st.form("prediction_form"):
        movie_title = st.text_input("Movie Title", placeholder="Enter movie title...")
        director = st.selectbox("Director", options=df["Director"].dropna().unique(), index=0)
        cast = st.multiselect(
            "Cast Members",
            options=pd.concat([df["Star1"], df["Star2"], df["Star3"], df["Star4"]]).unique(),
            placeholder="Select actors..."
        )
        runtime = st.slider("Runtime (minutes)", 60, 240, 120)
        genre = st.selectbox("Genre", options=df["Genre"].dropna().unique(), index=0)
        metascore = st.slider("Metascore", 0, 100, 70)
        votes = st.number_input("Number of Votes", 1000, 2000000, 10000)

        submitted = st.form_submit_button("🚀 Predict Rating!")

# Right Column - Results
with col2:
    st.header("🌟 Prediction Results")

    if submitted:
        try:
            # Make prediction
            input_features = [runtime, metascore, votes]
            scaled_features = scaler.transform([input_features])
            input_tensor = torch.tensor(scaled_features, dtype=torch.float32)

            with torch.no_grad():
                prediction = model.predict(input_tensor)

            predicted_rating = round(prediction[0][0], 1)

            # Display rating with style
            st.markdown(f"""
            <div class="result-box" style="background-color: {'#4CAF50' if predicted_rating >= 7 else '#FF9800'};">
                <h2 style="color: white;">{movie_title or 'Your Movie'}</h2>
                <p class="big-font" style="color: white; margin: 0;">{predicted_rating:.1f}/10</p>
            </div>
            """, unsafe_allow_html=True)

            # Rating feedback
            if predicted_rating >= 8:
                st.balloons()
                st.success("⭐ Blockbuster Alert! This could be the next big hit!")
            elif 6 <= predicted_rating < 8:
                st.info("🎥 Solid Performer! Worth a watch!")
            else:
                st.warning("💤 Might need some improvements...")

            # Progress bar visualization
            st.markdown(f"""
            <div style="margin: 20px 0;">
                <div style="background: #eee; border-radius: 5px; padding: 3px;">
                    <div style="background: {'#4CAF50' if predicted_rating >= 7 else '#FF9800'}; 
                        width: {predicted_rating * 10}%; 
                        height: 20px; 
                        border-radius: 3px;"></div>
                </div>
                <p style="text-align: center; margin: 5px 0;">IMDb Score Meter</p>
            </div>
            """, unsafe_allow_html=True)

            # Movie details card
            with st.expander("📄 Movie Details"):
                cols = st.columns(2)
                with cols[0]:
                    st.write(f"**Director:** {director}")
                    st.write(f"**Runtime:** {runtime} minutes")
                    st.write(f"**Metascore:** {metascore}")
                with cols[1]:
                    st.write(f"**Genre:** {genre}")
                    st.write(f"**Votes:** {votes:,}")
                    st.write(f"**Cast:** {', '.join(cast) if cast else 'Not specified'}")

        except Exception as e:
            st.error(f"🚨 Prediction failed: {str(e)}")
    else:
        st.markdown("""
        <div style="text-align: center; padding: 50px 20px; border: 2px dashed #4CAF50; border-radius: 10px;">
            <h3>👈 Fill in the details and click Predict!</h3>
            <p>Your prediction will appear here</p>
        </div>
        """, unsafe_allow_html=True)
