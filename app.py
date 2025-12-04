import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

# Column order exactly as in your notebook (all lowercase)
FEATURES = ['crim', 'zn', 'indus', 'chas', 'nox',
            'rm', 'age', 'dis', 'rad', 'tax',
            'ptratio', 'b', 'lstat']


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = []
        missing = []

        # Safely read all fields
        for f in FEATURES:
            v = request.form.get(f)   # returns None instead of raising error
            if v is None or v.strip() == "":
                missing.append(f)
            else:
                values.append(float(v))

        # If any field missing, show message instead of 400 error
        if missing:
            return render_template(
                'home.html',
                prediction_text=f"Please fill all fields. Missing: {', '.join(missing)}"
            )

        # Convert to numpy, scale, predict
        arr = np.array(values).reshape(1, -1)
        transformed = scaler.transform(arr)
        prediction = regmodel.predict(transformed)[0]

        return render_template(
            'home.html',
            prediction_text=f'The House price prediction is {prediction:.2f}'
        )

    except Exception as e:
        return render_template(
            'home.html',
            prediction_text=f'Error: {str(e)}'
        )


# Optional JSON API
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'No JSON received'}), 400

    try:
        values = [float(data[f]) for f in FEATURES]
        arr = np.array(values).reshape(1, -1)
        transformed = scaler.transform(arr)
        prediction = regmodel.predict(transformed)[0]
        return jsonify({'prediction': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
