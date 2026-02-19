from flask import Flask, render_template, request
import pandas as pd
import pickle
import joblib

app = Flask(__name__)

# Load trained pipeline model
model = joblib.load("model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Create dataframe from form input
        input_df = pd.DataFrame([request.form.to_dict()])

        # Convert numeric columns safely
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

        for col in numeric_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        # Check for invalid numeric inputs
        if input_df[numeric_cols].isnull().any().any():
            return render_template(
                "index.html",
                prediction_text="Invalid numeric input. Please enter valid numbers."
            )

        # Make prediction
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            result = "Customer is likely to Churn ❌"
        else:
            result = "Customer is likely to Stay ✅"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error occurred: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)