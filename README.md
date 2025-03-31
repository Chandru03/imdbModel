# 🎬 IMDb Rating Predictor

## 📌 About the Project
This project is an **IMDb Rating Predictor** that uses a **TabNet Regressor** model to predict the IMDb rating of a movie based on its characteristics. The prediction is powered by a **Streamlit-based web application**, where users can input movie details like runtime, metascore, and number of votes to get an estimated IMDb rating.

## 🚀 Features
- 📊 **Machine Learning Model:** Uses a trained **TabNet Regressor** for predictions.
- 🎥 **Movie Data Input:** Users can enter movie runtime, metascore, votes, and select genre, director, and cast.
- ⚡ **Fast Predictions:** The app instantly provides an estimated IMDb rating based on user input.
- 🎭 **Interactive UI:** Built with **Streamlit** for a seamless user experience.
- 📂 **Pre-trained Model:** The model is pre-trained on IMDb’s **Top 1000 Movies dataset**.

## 📁 Dataset
The model is trained on the **IMDb Top 1000 Movies dataset**, which includes details such as:
- 🎬 Movie Title
- ⏳ Runtime
- 🏆 Meta Score (Critic reviews)
- 🗳️ Number of Votes
- 🎭 Genre
- 🎬 Director & Cast
- ⭐ IMDb Rating (Target variable)

## 🧠 Model Details
- **Algorithm:** TabNet Regressor (PyTorch-based Deep Learning Model)
- **Training Framework:** PyTorch + TabNet
- **Input Features:**
  - `Runtime`
  - `Meta Score`
  - `Number of Votes`
- **Target Variable:** IMDb Rating
- **Evaluation Metric:** Mean Absolute Error (MAE)

## 🛠️ Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/imdb-rating-predictor.git
cd imdb-rating-predictor
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

## 🏗️ How It Works
1. **User Inputs Movie Details** 🎬
2. **The Model Processes Inputs** 🧠
3. **Features are Scaled using StandardScaler** ⚖️
4. **TabNet Model Predicts IMDb Rating** ⭐
5. **The Web App Displays Results** 📊

## 🔍 Example Usage
- **Movie:** "Inception"
- **Runtime:** 148 minutes
- **Metascore:** 74
- **Number of Votes:** 2,000,000+
- **Predicted IMDb Rating:** ~8.8⭐

## 📜 License
This project is open-source and available under the **MIT License**.

---

### 👨‍💻 Author
Developed by **Chandru S** 👨‍💻✨ | AI & Web Developer | Passionate about ML & AI

📩 Feel free to reach out for collaborations!

