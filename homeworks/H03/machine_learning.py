from typing import Tuple
import numpy as np
from typing import Callable
from sklearn.model_selection import train_test_split

def binarize(labels: list[str]) -> np.array:
    """Binarize the labels.

    Binarize the labels such that "Chinstrap" is 1 and "Adelie" is 0.

    Args:
        labels (list[str]): The labels to binarize.
    
    Returns:
        np.array: The binarized labels.
    """

    binary_labels = np.array(list(map(lambda x : 1 if x == 'Chinstrap' else 0, labels)))
    return binary_labels

    raise NotImplementedError("Please implement the binarize function.")


def split_data(X: np.array, y: np.array, test_size: float=0.2, 
                random_state: float = 42, shuffle: bool = True) -> Tuple[np.array, np.array, np.array, np.array]:
    """Split the data into training and testing sets.

    IMPORTANT:
        Please use the train_test_split function from sklearn to split the data.
        Ensure your test_size is set to 0.2.
        Ensure your random_state is set to 42.
        Ensure shuffle is set to True.

    Args:
        X (np.array): The independent variables.
        y (np.array): The dependent variables.
        test_size (float): The proportion of the data to use for testing.
        random_state (int): The random seed to use for the split.
        shuffle (bool): Whether or not to shuffle the data before splitting.

    """

    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)

    raise NotImplementedError("Please implement the split_data function.")

def standardize(X_train: np.array, X_test: np.array) -> Tuple[np.array, np.array]:
    """Standardize the training and testing data.

    Standardize the training and testing data using the mean and standard deviation of
    the training set.

    Recall that your samples are rows and your features are columns. Your goal is to
    standardize along the columns (features). Ensure you use the mean and standard deviation
    of the training set for standardization of both training and testing sets.

    Args:
        X_train (np.array): The training data.
        X_test (np.array): The testing data.

    Returns:
        Tuple[np.array, np.array]: The standardized training and testing data.
    """

    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)

    X_train_standardized = (X_train - X_train_mean)/X_train_std
    X_test_standardized = (X_test - X_train_mean)/X_train_std

    return X_train_standardized, X_test_standardized

    raise NotImplementedError("Please implement the standardize function.")


def euclidean_distance(x1: np.array, x2: np.array) -> float:
    """Calculate the Euclidean distance between two points x1 and x2.

    Args:
        x1 (np.array): The first point.
        x2 (np.array): The second point.
    
    Returns:
        float: The Euclidean distance between the two points.
    """
    
    squared_diff = (x1 - x2) ** 2
    sum_squared_diff = np.sum(squared_diff)
    distance = np.sqrt(sum_squared_diff)

    return distance

    raise NotImplementedError("Please implement the euclidean_distance function.")


def cosine_distance(x1: np.array, x2: np.array) -> float:
    """Calculate the cosine distance between two points x1 and x2.

    Args:
        x1 (np.array): The first point.
        x2 (np.array): The second point.

    Returns:
        float: The cosine distance between the two points.
    """

    dot_product = np.dot(x1,x2)
    norm1 = np.linalg.norm(x1)
    norm2 = np.linalg.norm(x2)
    cos_sim = dot_product/(norm1*norm2)
    distance = 1 - cos_sim

    return distance

    raise NotImplementedError("Please implement the cosine_distance function.")
    
def knn(x: np.array, y: np.array, 
        sample: np.array, distance_method: Callable, k: int) -> int:
    """Perform k-nearest neighbors classification.

    Args:
        X (np.array): The training data.
        y (np.array): The training labels.
        sample (np.array): The point you want to classify.
        distance_method (Callable): The distance metric to use. This MUST 
            accept two np.arrays and return a float.
        k (int): The number of neighbors to consider as equal votes.
    
    Returns:
        int: The label of the sample.
    """

    # (distance, label) between the test sample and all the training samples.
    distances = []

    for x_i, y_i in zip(x, y):

        # 1. Calculate the distance between the test sample and the training sample.
        if distance_method == 'euclidean_distance':
            dist = euclidean_distance(sample, x_i)
        else:
            dist = cosine_distance(sample, x_i)

        # 2. Append the (distance, label) tuple to the distances list.
            distances.append((dist, y_i))

        # raise NotImplementedError("Please implement the knn function distance loop.")

    # 3. Sort the tuples by distance (the first element of each tuple in distances).
    distances.sort(key=lambda d: d[0])

    # 4. Get the unique labels and their counts. HINT: np.unique has a return_counts parameter.
    k_nearest_labels = [label for _, label in distances[:k]]
    unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)

    # 5. Return the label with the most counts.
    most_common_label = unique_labels[np.argmax(counts)]

    return most_common_label
    
    raise NotImplementedError("Please implement the knn function.")

def linear_regression(X: np.array, y: np.array) -> np.array:
    """Perform linear regression using the normal equation.

    Args:
        X (np.array): The independent variables.
        y (np.array): The dependent variables.
    
    Returns:
        np.array: The weights for the linear regression model
                (including the bias term)
    """

    # Add the bias term using hstack.
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # 1. Calculate the weights using the normal equation.
    X_transpose = X.T
    weights = (np.linalg.inv(X_transpose @ X)) @ X_transpose @ y

    return weights
    
    raise NotImplementedError("Please implement the linear_regression function.")

def linear_regression_predict(X: np.array, weights: np.array) -> np.array:
    """Predict the dependent variables using the weights and independent variables.

    Args:
        X (np.array): The independent variables.
        weights (np.array): The weights of the linear regression model.
    
    Returns:
        np.array: The predicted dependent variables.
    """
    # Add the bias term using hstack.
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # 1. Calculate the predictions.
    y_pred = X @ weights
    return y_pred

    raise NotImplementedError("Please implement the linear_regression_predict function.")
    

def mean_squared_error(y_true: np.array, y_pred: np.array) -> float:
    """Calculate the mean squared error.

    You should use only numpy for this calculation.

    Args:
        y_true (np.array): The true values.
        y_pred (np.array): The predicted values.
    
    Returns:
        float: The mean squared error.
    """

    sum_of_squared_error = (y_pred - y_true)**2
    mse = np.mean(sum_of_squared_error)

    return mse

    raise NotImplementedError("Please implement the mean_squared_error function.")

def logistic_regression_gradient_descent(X: np.array, y: np.array, 
                                         learning_rate: float = 0.01, 
                                         num_iterations: int = 5000) -> np.array:
    
    """Perform logistic regression using gradient descent.

    NOTE: It is important that you add a column of ones to the independent
    variables before performing gradient descent. This will effectively add
    a bias term to the model. The hstack function from numpy will be useful.

    NOTE: The weights should be initialized to zeros. np.zeros will be useful.

    NOTE: Please follow the formula provided in the lecture to update the weights.
    Other algorithms will work, but the tests are expecting the weights to be
    calculated in the way described in our lecture.

    NOTE: The tests expect a learning rate of 0.01 and 5000 iterations. Do
    not change these values prior to submission.

    Args:
        X (np.array): The independent variables.
        y (np.array): The dependent variables.
        learning_rate (float): The learning rate.
        num_iterations (int): The number of iterations to perform.
    
    Returns:
        np.array: The weights for the logistic regression model.
    """
    # Add the bias term using np.hstack.
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # 1. Initialize the weights with zeros. np.zeros is your friend here! 
    weights = np.zeros(X.shape[1])

    # For each iteration, update the weights.
    for _ in range(num_iterations):

        # 2. Calculate the predictions.
        z = np.dot(X, weights)
        Y_pred = 1 / (1 + np.exp(-z))

        # 3. Calculate the gradient.
        gradient = np.dot(X.T, (Y_pred - y)) / y.size

        # 4. Update the weights -- make sure to use the learning rate!
        weights -= learning_rate * gradient

    return weights

    raise NotImplementedError("Please implement the logistic_regression_gradient_descent function.")
    

def logistic_regression_predict(X: np.array, weights: np.array) -> np.array:
    """Predict the labels for the logistic regression model.

    Args:
        X (np.array): The independent variables.
        weights (np.array): The weights of the logistic regression model. This
            should include the bias term.
    
    Returns:
        np.array: The predicted labels.
    """
    # Add the bias term using np.hstack.
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # 1. Calculate the predictions using the provided weights.
    z = np.dot(X, weights)
    probabilities = 1 / (1 + np.exp(-z))

    return probabilities
    
    raise NotImplementedError("Please implement the logistic_regression_predict function.")