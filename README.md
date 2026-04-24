# ✈️ Flight Fare Prediction

A machine learning web application that predicts Indian domestic flight ticket prices based on user inputs like airline, source, destination, date, and travel details.

---

## 🔗 Live Demo
[Click here to view the app](https://your-app-name.streamlit.app)

---

## 📌 Project Overview

This project uses a **Random Forest Regressor** model trained on Indian domestic flight data to predict ticket prices. The model was tuned using **RandomizedSearchCV** for optimal performance. The web application is built using **Streamlit** and deployed on **Streamlit Cloud**.

---

## 🗂️ Dataset

The dataset used for training contains the following features:

| Feature | Description |
|---|---|
| Airline | Name of the airline |
| Source | Departure city |
| Destination | Arrival city |
| Date of Journey | Date of travel |
| Dep Time | Departure time |
| Arrival Time | Arrival time |
| Duration | Total flight duration |
| Total Stops | Number of stops |
| Price | Ticket price (Target) |

---

## ⚙️ How It Works

1. User fills in flight details in the web form
2. Inputs are preprocessed exactly as done during model training
   - Date split into day and month
   - Time split into hours and minutes
   - Duration split into hours and minutes
   - Categorical features encoded using One Hot Encoding
   - Stops encoded using Label Encoding
3. Preprocessed inputs are fed into the trained Random Forest model
4. Predicted fare is displayed on screen

---

## 🧠 Model Details

| Detail | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Hyperparameter Tuning | RandomizedSearchCV |
| Training Data | Indian domestic flights |
| Model File | Stored on Google Drive, downloaded at runtime |

---

## 🛠️ Tech Stack

- **Python**
- **Pandas & NumPy** — data preprocessing
- **Scikit-learn** — model training
- **Streamlit** — web application
- **gdown** — model download from Google Drive
- **Pickle** — model serialization

---

## 📁 Project Structure
```
Flight-Fare-Prediction/
├── app.py              ← Streamlit web application
├── requirements.txt    ← Required Python packages
├── .gitignore          ← Ignores large model file
└── README.md           ← Project documentation

```
---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Meghasyam-Gorla/Flight-Fare-Prediction.git
cd Flight-Fare-Prediction
```

**2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Google Drive File ID in `.streamlit/secrets.toml`**
```toml
GDRIVE_FILE_ID = "your_file_id_here"
```

**5. Run the app**
```bash
streamlit run app.py
```

---

## ☁️ Deployment
- The trained model (`flight_rf.pkl`) is stored on **Google Drive**
- The Google Drive File ID is stored securely in **Streamlit Secrets**
- On first launch the model is automatically downloaded using `gdown`

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| R² Score | ~0.80 |
| MAE | ~1200 |
| RMSE | ~1900 |

> Note: Values are approximate based on training results.

---

## 👤 Author

**Meghasyam Gorla**
- GitHub: [@Meghasyam-Gorla](https://github.com/Meghasyam-Gorla)
 ---
