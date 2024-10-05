from flask import Flask, request, jsonify
from flask_cors import CORS  # for CORS
import pickle

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Load models with error handling
try:
    with open("C:/Users/Admin/Desktop/depolearn/Iris_classifications/svm.pkl", "rb") as f:
        svm_model = pickle.load(f)
    with open("C:/Users/Admin/Desktop/depolearn/Iris_classifications/log_model.pkl", "rb") as f:
        log_model = pickle.load(f)
    with open("C:/Users/Admin/Desktop/depolearn/Iris_classifications/lin_model.pkl", "rb") as f:
        lin_model = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    exit(1)  # Exit if models cannot be loaded

def classify(input_data):
    """
    Classifies the input data using different machine learning models.

    Args:
        input_data (list): A list of input features for prediction.

    Returns:
        dict: A dictionary containing predictions from Linear Regression,
              Logistic Regression, and SVM models.
    """
    return {
        "Linear Regression": lin_model.predict(input_data).tolist(),
        "Logistic Regression": log_model.predict(input_data).tolist(),
        "SVM": svm_model.predict(input_data).tolist()
    }

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to make predictions based on input features.

    Expected JSON input:
    {
        "sepal_length": float,
        "sepal_width": float,
        "petal_length": float,
        "petal_width": float
    }

    Returns:
        json: A JSON response containing the predictions or an error message
              if required fields are missing.
    """
    data = request.json  # Get JSON data from request
    
    # Check for required fields
    if not all(key in data for key in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']):
        return jsonify({"error": "Missing fields"}), 400  # Bad Request

    inputs = [[data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]]

    # Predictions
    predictions = classify(inputs)
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app
