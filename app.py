from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['age']),
            int(request.form['anaemia']),
            float(request.form['creatinine_phosphokinase']),
            int(request.form['diabetes']),
            float(request.form['ejection_fraction']),
            int(request.form['high_blood_pressure']),
            float(request.form['platelets']),
            float(request.form['serum_creatinine']),
            float(request.form['serum_sodium']),
            int(request.form['sex']),
            int(request.form['smoking']),
            int(request.form['time'])
        ]
        
        prediction = model.predict([np.array(features)])
        result = '💔 High Risk of Heart Failure' if prediction[0] == 1 else '❤️ Low Risk of Heart Failure'
        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {e}")

if __name__ == '__main__':
     app.run(debug=True, port=5050)

