from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('credit_card_fraud_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input features from the form
        features_str = request.form['features']
        
        # Split the input string by commas, strip any extra spaces, and convert to float
        features = [float(feature.strip()) for feature in features_str.split(',')]
        
        # Ensure the number of features is correct
        if len(features) != 30:
            return render_template('index.html', prediction_text='Error: Please enter exactly 30 features.')
        
        # Convert to numpy array and reshape for the model
        data = np.array([features])
        
        # Make prediction
        prediction = model.predict(data)
        
        # Output the result
        output = prediction[0]
        result_text = 'Fraud' if output == 1 else 'Not Fraud'
        
        return render_template('index.html', prediction_text=f'Fraud Prediction: {result_text}')
    
    except ValueError:
        # Handle value errors, such as when conversion to float fails
        return render_template('index.html', prediction_text='Error: Please enter valid numerical values separated by commas.')

if __name__ == "__main__":
    app.run(debug=True)
