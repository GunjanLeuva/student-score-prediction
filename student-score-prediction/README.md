# 📊 Student Score Prediction using Linear Regression

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Linear%20Regression-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![Project Status](https://img.shields.io/badge/Status-Completed-brightgreen)

A beginner-friendly **Machine Learning project** that predicts student **exam scores based on study hours** using a Linear Regression model.
The project now includes an **interactive web application** built with Streamlit for real-time predictions and data visualization.

---

# 🚀 Project Overview

This project demonstrates a **complete end-to-end Machine Learning workflow**:

* Load and explore dataset (Study Hours vs Exam Score)
* Visualize the relationship using scatter plots
* Preprocess data and split into training and testing sets
* Train a Linear Regression model using scikit-learn
* Evaluate model performance using MAE, MSE, and R²
* Display regression line visualization
* Build an **interactive Streamlit web application**

---

# 🖥️ Interactive Web Application

The project includes a modern **Streamlit dashboard** where users can interact with the ML model.

### Key Features

✔ Interactive study hours slider
✔ Real-time exam score prediction
✔ Data visualization with regression line
✔ User prediction point highlighted on graph
✔ Contextual feedback based on predicted score
✔ Clean and professional UI suitable for portfolio

---

# 📷 Application Preview

Add your dashboard screenshot here after uploading it to the repository.

```
student-score-prediction/
└── images/
    └── dashboard.png
```

Then display it like this:

![Student Score Prediction Dashboard](images/dashboard.png)

---

# 🧠 Machine Learning Model

This project uses **Linear Regression** to model the relationship between:

```
Input  (X) → Study Hours
Output (Y) → Exam Score
```

The model learns the relationship between study time and expected exam performance.

Example equation:

```
Score = m × Hours + b
```

Where:

* **Hours** → Study hours input
* **Score** → Predicted exam score
* **m** → Slope learned from training data
* **b** → Intercept

---

# 📈 Model Performance

Example evaluation metrics:

| Metric                    | Value  |
| ------------------------- | ------ |
| Mean Absolute Error (MAE) | ~3.34  |
| Mean Squared Error (MSE)  | ~10.48 |
| R² Score                  | ~0.966 |

### Interpretation

* **MAE ≈ 3.34** → Model predictions are on average about 3 marks away from actual values.
* **R² ≈ 0.966** → The model explains about **96.6% of the variance** in the dataset.

---

# 🧰 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit
* Plotly (for interactive graphs)

---

# 📁 Project Structure

```
student-score-prediction/
│
├── data/
│   └── student_scores.csv
│
├── models/
│   └── linear_regression_model.joblib
│
├── notebooks/
│   └── student_score_prediction.ipynb
│
├── src/
│   ├── data_loader.py
│   └── model.py
│
├── app.py
├── main.py
├── test_app.py
├── requirements.txt
└── README.md
```

---

# ▶️ How to Run the Project

## 1️⃣ Clone the repository

```
git clone https://github.com/YOUR_USERNAME/student-score-prediction.git
```

## 2️⃣ Navigate to the project folder

```
cd student-score-prediction
```

## 3️⃣ Create a virtual environment

```
python -m venv .venv
```

Activate environment

**Windows**

```
.venv\Scripts\activate
```

**Mac/Linux**

```
source .venv/bin/activate
```

---

## 4️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## 5️⃣ Run the Streamlit Web App

```
streamlit run app.py
```

Then open your browser and go to:

```
http://localhost:8501
```

---

# 🧪 Testing

You can test the core ML functionality before launching the app.

```
python test_app.py
```

Expected output:

```
Testing core ML functionality...
Data loaded successfully
Model trained successfully
Predictions working correctly
All tests passed
```

---

# 📊 Example Usage

**Web App**

1. Move the slider to select study hours (0-15)
2. Click **Predict Score**
3. See the predicted exam score
4. View the graph with regression line and your prediction highlighted

Example:

```
Study Hours: 7.5
Predicted Score: 78.5
```

---

# 🚀 Future Improvements

* Deploy the application online
* Add multiple machine learning models
* Improve UI with advanced Streamlit components
* Add larger datasets for better accuracy

---

# 👨‍💻 Author

**Gunjan Leuva**

MCA Student | Aspiring AI & Full-Stack Developer
Machine Learning Enthusiast

📧 Email: [leuvagunjan18@gmail.com](mailto:leuvagunjan18@gmail.com)

---

⭐ If you found this project helpful, consider **starring the repository on GitHub**.
