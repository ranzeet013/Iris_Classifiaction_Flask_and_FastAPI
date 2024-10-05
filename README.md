# Iris Flower Classification Project

## Project Overview

This project is designed to classify Iris flower species using machine learning models. It uses three different algorithms Linear Regression, Logistic Regression, and Support Vector Machine (SVM) to predict which species of Iris a given flower belongs to, based on its physical characteristics: sepal length, sepal width, petal length, and petal width.

The project consists of:

1. **Machine Learning Models**:
   - Trained models for Iris flower classification.
   - Models are trained using the famous Iris dataset from the `sklearn` library.
   - The models used are:
     - **Linear Regression**
     - **Logistic Regression**
     - **Support Vector Machine (SVM)**

2. **API Implementation**:
   - Two versions of APIs are provided: one using **FastAPI** and another using **Flask**.
   - These APIs allow users to send flower measurements via POST requests and get predictions from the trained models.

3. **Frontend Support (Optional)**:
   - Static files can be served through FastAPI to create a simple frontend for interaction.

## What's Included

### 1. **Pre-Trained Models**
   - The project includes three pre-trained models saved as `.pkl` files:
     - `lin_model.pkl` (Linear Regression)
     - `log_model.pkl` (Logistic Regression)
     - `svc_model.pkl` (SVM)
   - These models are loaded at runtime and used for predictions.

### 2. **API with FastAPI**:
   - The main functionality of the project is exposed through an API built using FastAPI.
   - The API provides an endpoint (`/predict_all_models`) that accepts a JSON input containing the flower's measurements and returns predictions from each model.
   
### 3. **API with Flask**:
   - An alternative version of the API built with Flask is included.
   - The Flask API provides a similar endpoint (`/predict`) that also accepts flower measurements and returns predictions from the models.

### 4. **Model Training Script**:
   - The project includes a Python script that trains the machine learning models.
   - This script uses the Iris dataset and the `train_test_split` method to train Linear Regression, Logistic Regression, and SVM models.
   - After training, the models are saved as `.pkl` files using Python's `pickle` library.

### 5. **Request Handling and Response**:
   - The API accepts requests with four features: sepal length, sepal width, petal length, and petal width.
   - Based on these inputs, the API uses each model to predict the species of the flower and returns the results in a structured JSON format.

## How the Project Works

1. **Input**: The user provides the flower's features (sepal length, sepal width, petal length, and petal width) through a POST request to the API.
   
2. **Model Predictions**: The API uses the pre-trained models to classify the flower species into one of the three categories: Setosa, Versicolor, or Virginica.

3. **Response**: The API returns a JSON response with the predictions from each model, allowing users to see how each model classifies the flower based on the provided input.

## Installation and Setup

1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the FastAPI app using `uvicorn` or the Flask app directly from the terminal.
4. Use tools like Postman or `requests` in Python to test the API by sending POST requests.

## Features

- **Multiple Models**: Offers predictions from three different models.
- **API Support**: Two API implementations using FastAPI and Flask for flexibility.
- **Error Handling**: Graceful handling of errors, such as missing model files.
- **Static File Serving**: FastAPI setup includes an option to serve static files for frontend use.

## How to Use

1. **Train the Models**: Use the included training script to train and save models if needed.
2. **Run the API**: Start the FastAPI or Flask server to serve predictions.
3. **Make Predictions**: Send a POST request to the API with flower measurements, and receive predictions from each model.


