import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the Logistic Regression model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract 17 features matching feature_names_in_
        age = float(request.form['age'])
        gender = request.form['gender']
        city = request.form['city']
        bmi = float(request.form['bmi'])
        family_history = request.form['family_history_diabetes']
        activity_level = request.form['physical_activity_level']
        diet = request.form['diet_type']
        smoking = request.form['smoking_status']
        alcohol = request.form['alcohol_consumption']
        sleep = float(request.form['hours_sleep_per_night'])
        stress = request.form['stress_level']
        fasting_bs = float(request.form['fasting_blood_sugar'])
        hba1c = float(request.form['hba1c_level'])
        bp_systolic = float(request.form['blood_pressure_systolic'])
        bp_diastolic = float(request.form['blood_pressure_diastolic'])
        waist = float(request.form['waist_circumference_cm'])
        income = request.form['income_bracket']

        # NOTE: Ensure string/categorical values match any encoding used during training
        features = np.array([[
            age, gender, city, bmi, family_history, activity_level,
            diet, smoking, alcohol, sleep, stress, fasting_bs,
            hba1c, bp_systolic, bp_diastolic, waist, income
        ]], dtype=object)

        prediction = model.predict(features)[0]
        
        # Get probability confidence score
        probabilities = model.predict_proba(features)[0]
        confidence = round(np.max(probabilities) * 100, 2)

        return render_template(
            'index.html', 
            prediction=prediction, 
            confidence=confidence
        )
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
