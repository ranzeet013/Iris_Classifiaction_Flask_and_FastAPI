from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
import pickle

def load_data():
    """
    Loads the Iris dataset and returns features and target variables.

    Returns:
        tuple: A tuple containing features (X) and target (y) arrays.
    """
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y

def train_models(X, y):
    """
    Splits the data into training and testing sets, trains three models,
    and returns the trained models.

    Args:
        X (array-like): Features for training.
        y (array-like): Target variable for training.

    Returns:
        tuple: A tuple containing trained Linear Regression, 
               Logistic Regression, and SVC models.
    """
    x_train, x_test, y_train, y_test = train_test_split(X, y)

    lin_reg = LinearRegression().fit(x_train, y_train)
    log_reg = LogisticRegression().fit(x_train, y_train)
    svc_model = SVC().fit(x_train, y_train)

    return lin_reg, log_reg, svc_model

def save_models(models):
    """
    Saves the trained models to disk.

    Args:
        models (tuple): A tuple containing trained models to be saved.
    """
    pickle.dump(models[0], open('lin_model.pkl', 'wb'))
    pickle.dump(models[1], open('log_model.pkl', 'wb'))
    pickle.dump(models[2], open('svc_model.pkl', 'wb'))

# Main script execution
if __name__ == "__main__":
    X, y = load_data()               # Load data
    models = train_models(X, y)      # Train models
    save_models(models)              # Save trained models
