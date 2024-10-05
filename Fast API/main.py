import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

# Serve static files for frontend (if any)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load models with error handling
try:
    with open("C:/Users/Admin/Desktop/depolearn/Iris_classifications/svm.pkl", "rb") as f:
        svm_model = pickle.load(f)
except Exception as e:
    print(f"Error loading SVM model: {e}")
    svm_model = None

try:
    with open("C:/Users/Admin/Desktop/depolearn/Iris_classifications/log_model.pkl", "rb") as f:
        log_model = pickle.load(f)
except Exception as e:
    print(f"Error loading Logistic model: {e}")
    log_model = None

try:
    with open("C:/Users/Admin/Desktop/depolearn/Iris_classifications/lin_model.pkl", "rb") as f:
        lin_model = pickle.load(f)
except Exception as e:
    print(f"Error loading Linear Regression model: {e}")
    lin_model = None

# Define the input model (input fields for Iris features)
class IrisFeatures(BaseModel):
    """
    Model for representing the features of an Iris flower.

    Attributes:
        sepal_length (float): Length of the sepal.
        sepal_width (float): Width of the sepal.
        petal_length (float): Length of the petal.
        petal_width (float): Width of the petal.
    """
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Helper function to classify results
def classify(num):
    """
    Classifies the iris flower species based on the prediction output.

    Args:
        num (float): The prediction output to classify.

    Returns:
        str: The predicted species name.
    """
    if num < 0.5:
        return 'Setosa'
    elif num < 1.5:
        return 'Versicolor'
    else:
        return 'Virginica'

# Define a single route that uses all models
@app.post("/predict_all_models")
async def predict_all_models(features: IrisFeatures):
    """
    Endpoint to predict the species of an Iris flower using multiple models.

    Args:
        features (IrisFeatures): The input features for the Iris flower.

    Returns:
        dict: A dictionary containing predictions from all available models.
    """
    input_data = np.array([[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]])

    # Initialize response dictionary
    response = {}

    # Linear Regression prediction
    if lin_model is not None:
        lin_prediction = lin_model.predict(input_data)
        response['Linear Regression'] = classify(lin_prediction)

    # Logistic Regression prediction
    if log_model is not None:
        log_prediction = log_model.predict(input_data)
        response['Logistic Regression'] = classify(log_prediction)

    # SVM prediction
    if svm_model is not None:
        svm_prediction = svm_model.predict(input_data)
        response['SVM'] = classify(svm_prediction)

    # Return the response
    return response

@app.get("/")
async def read_index():
    """
    Serves the static index.html file.

    Returns:
        FileResponse: The response containing the index.html file.
    """
    return FileResponse("static/index.html")
