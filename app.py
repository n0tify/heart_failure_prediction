from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# CSV logging path (to log input and prediction)
LOG_FILE = 'submitted_data_log.csv'

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None, form_data={})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting form data
        form_data = {
            'age': float(request.form['age']),
            'anaemia': int(request.form['anaemia']),
            'creatinine_phosphokinase': float(request.form['creatinine_phosphokinase']),
            'diabetes': int(request.form['diabetes']),
            'ejection_fraction': float(request.form['ejection_fraction']),
            'high_blood_pressure': int(request.form['high_blood_pressure']),
            'platelets': float(request.form['platelets']),
            'serum_creatinine': float(request.form['serum_creatinine']),
            'serum_sodium': float(request.form['serum_sodium']),
            'sex': int(request.form['sex']),
            'smoking': int(request.form['smoking']),
            'time': int(request.form['time'])
        }

        # Prepare features for prediction
        features = np.array(list(form_data.values())).reshape(1, -1)
        prediction = model.predict(features)
        
        # Create result text
        result = '💔 High Risk of Heart Failure' if prediction[0] == 1 else '❤️ Low Risk of Heart Failure'

        # Log to CSV (optional)
        log_data = form_data.copy()
        log_data['prediction'] = prediction[0]
        df = pd.DataFrame([log_data])
        if os.path.exists(LOG_FILE):
            df.to_csv(LOG_FILE, mode='a', header=False, index=False)
        else:
            df.to_csv(LOG_FILE, index=False)

        # Render with prediction and pre-filled form
        return render_template('index.html', prediction_text=result, form_data=form_data)

    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {e}", form_data=request.form)

if __name__ == '__main__':
    app.run(debug=True, port=5050)
