import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained pipeline (Scaler + Model inside)
# Make sure regmodel.pkl is in the same folder as this app.py
regmodel = pickle.load(open('regmodel.pkl', 'rb'))


@app.route('/')
def home():
    # home.html should have a form that posts to /predict (for browser use)
    return render_template('home.html')


# ---------- API route: for Postman / JS / external use ----------
@app.route('/predict_api', methods=['POST'])
def predict_api():
    """
    Expects JSON like:
    {
      "data": {
        "feature1": 0.1,
        "feature2": 3.5,
        ...
      }
    }
    """
    content = request.get_json()

    # Support both {"data": {...}} and just {...}
    if isinstance(content, dict) and 'data' in content:
        data = content['data']
    else:
        data = content

    # Convert to numpy array
    try:
        values = np.array(list(data.values()), dtype=float).reshape(1, -1)
    except Exception as e:
        return jsonify({"error": f"Invalid input format: {str(e)}"}), 400

    # Pipeline handles scaling + prediction inside
    prediction = regmodel.predict(values)[0]

    return jsonify({"prediction": float(prediction)})


# ---------- Form route: for browser form submission ----------
@app.route('/predict', methods=['POST'])
def predict():
    """
    For HTML <form> that sends form-data (from home.html)
    """
    try:
        # request.form.values() -> all inputs from the form in order
        values = [float(x) for x in request.form.values()]
        final_input = np.array(values).reshape(1, -1)

        prediction = regmodel.predict(final_input)[0]

        return render_template(
            'home.html',
            prediction_text=f'Predicted value: {prediction:.2f}'
        )
    except Exception as e:
        return render_template(
            'home.html',
            prediction_text=f'Error: {str(e)}'
        )


if __name__ == "__main__":
    app.run(debug=True)
