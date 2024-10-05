import requests

def send_prediction_request(url, features):
    """
    Sends a POST request to the specified URL with the given flower features.

    Args:
        url (str): The URL of the prediction endpoint.
        features (dict): A dictionary containing the features of the Iris flower.

    Returns:
        tuple: A tuple containing the status code and the response JSON.
    """
    response = requests.post(url, json=features)
    return response.status_code, response.json()

# Define the URL for the prediction endpoint
url = "http://127.0.0.1:5000/predict"

# Prepare the data to be sent in the POST request
data = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

# Call the function and get the response
status_code, response_json = send_prediction_request(url, data)

# Print the results
print("Status Code:", status_code)
print("Response JSON:", response_json)
