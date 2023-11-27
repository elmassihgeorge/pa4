import numpy as np
import matplotlib.pyplot as plt
import math
import util

class Model:
    """
    Abstract class for a machine learning model.
    """
    
    def get_features(self, x_input):
        pass

    def get_weights(self):
        pass

    def hypothesis(self, x):
        pass

    def predict(self, x):
        pass

    def loss(self, x, y):
        pass

    def gradient(self, x, y):
        pass

    def train(self, dataset):
        pass

from typing import List
import numpy as np
# PA4 Q1
class PolynomialRegressionModel(Model):
    """
    Linear regression model with polynomial features (powers of x up to specified degree).
    x and y are real numbers. The goal is to fit y = hypothesis(x).
    """
    def __init__(self, degree : int = 1, learning_rate : float = 1e-4):
        self.degree = degree
        self.learning_rate = learning_rate
        self.weights = [1 for _ in range(self.degree + 1)]
 
    """
    Returns a list of features for input x
    """
    def get_features(self, x : float) -> List[float]:
        li = [x ** i for i in range(1, self.degree + 1)]
        li.insert(0, 1)
        return li

    """
    Returns the current parameter values
    """
    def get_weights(self) -> List[float]:
        return self.weights

    """
    Returns h_theta(x)
    """
    def hypothesis(self, x) -> float:
        return np.dot(self.get_features(x), self.get_weights())

    """
    Return the prediction y for input x
    In regression, y = h_theta(x)
    """
    def predict(self, x):
        return self.hypothesis(x)

    """
    Returns the loss on a single sample L(h_theta(x), y)
    In linear regression, this is the squared-error loss
    """
    def loss(self, x, y):
        return (self.hypothesis(x) - y) ** 2

    """
    Returns a list of partial derivatives of the loss function with respect to each weight,
    evaluated at the sample (x, y) and the current weights theta.
    """
    def gradient(self, x, y):
        features = self.get_features(x)
        gradient = []
        for feature in features:
            gradient.append(2*(self.hypothesis(x)-y)*feature)
        return gradient

    """
    Performs the training loop using the supplied dataset.
    (Ignore evalset for now)
    Use stochastic gradient descent (one weight update per sample)
    Choose an appropriate number of iterations for the training loop
    """
    def train(self, dataset, epochs : int = 10000, interval : int = 10, evalset = None):
        eval_iters = []
        losses = []

        for epoch in range(epochs):
            for x, y in zip(dataset.xs, dataset.ys):
                gradient = self.gradient(x, y)
                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * gradient[i]
                if epoch % interval == 0:
                    eval_iters.append(epoch)
                    losses.append(dataset.compute_average_loss(self, interval))
        return (eval_iters, losses)
                    


# PA4 Q2
def linear_regression():
    "*** YOUR CODE HERE ***"
    # Examples
    # Part (a)
    sine_train = util.get_dataset("sine_train")
    sine_model = PolynomialRegressionModel()
    eval_iters, losses = sine_model.train(sine_train)
    sine_train.plot_data(sine_model)

    # Part (b)
    sine_train.plot_loss_curve(eval_iters, losses)

    # Part (c)
    test_hyperparameters = [
        (1, 1e-4),
        (1, 1e-2),
        (2, 1e-4),
        (2, 1e-6),
        (3, 1e-6),
        (4, 1e-8),
        (4, 1e-12),
        (5, 1e-12),
        (5, 1e-15),
        (6, 1e-18)
    ]
        
# PA4 Q3
class BinaryLogisticRegressionModel(Model):
    """
    Binary logistic regression model with image-pixel features (num_features = image size, e.g., 28x28 = 784 for MNIST).
    x is a 2-D image, represented as a list of lists (28x28 for MNIST). y is either 0 or 1.
    The goal is to fit P(y = 1 | x) = hypothesis(x), and to make a 0/1 prediction using the hypothesis.
    """

    def __init__(self, num_features = 784, learning_rate = 1e-2):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.weights = np.zeros(num_features + 1)  # Additional weight for bias term

    def get_features(self, x):
        return np.array(x).flatten()

    def get_weights(self):
        return self.weights

    def hypothesis(self, x):
        z = np.dot(self.weights, np.insert(self.get_features(x), 0, 1))
        return 1 / (1 + np.exp(-z))

    def predict(self, x):
        return 1 if self.hypothesis(x) >= 0.5 else 0

    def loss(self, x, y):
        h = self.hypothesis(x)
        return -y * np.log(h) - (1 - y) * np.log(1 - h)
    
    def gradient(self, x, y):
        h = self.hypothesis(x)
        error = h - y
        gradient = error * np.insert(self.get_features(x), 0, 1)
        return gradient

    
    def train(self, dataset, evalset=None, num_epochs=100):
        eval_iters = []
        accuracies = []
        interval = 10

        for epoch in range(num_epochs):
            total_loss = 0.0
            for x, y in zip(dataset.xs, dataset.ys):
                grad = self.gradient(x, y)
                self.weights -= self.learning_rate * grad
                total_loss += self.loss(x, y)
            
            # Optionally, evaluate the model on the evaluation set
            if evalset is not None and epoch % interval == 0:
                accuracy = self.evaluate(evalset)
                accuracies.append(accuracy)
                eval_iters.append(epoch + 1)
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}, Accuracy: {accuracy}")

        return (eval_iters, accuracies)

    def evaluate(self, dataset):
        correct_predictions = 0
        total_samples = dataset.get_size()

        for x, y in zip(dataset.xs, dataset.ys):
            prediction = self.predict(x)
            if prediction == y:
                correct_predictions += 1

        accuracy = correct_predictions / total_samples
        return accuracy

# PA4 Q4
def binary_classification():
    mnist_binary_train = util.get_dataset("mnist_binary_train")
    binary_logistic_regresion_model = BinaryLogisticRegressionModel()
    mnist_binary_test = util.get_dataset("mnist_binary_test")
    eval_iters, accuracies = binary_logistic_regresion_model.train(mnist_binary_train, evalset=mnist_binary_test)
    mnist_binary_train.plot_accuracy_curve(eval_iters, accuracies)


# PA4 Q5
class MultiLogisticRegressionModel(Model):
    """
    Multinomial logistic regression model with image-pixel features (num_features = image size, e.g., 28x28 = 784 for MNIST).
    x is a 2-D image, represented as a list of lists (28x28 for MNIST). y is an integer between 1 and num_classes.
    The goal is to fit P(y = k | x) = hypothesis(x)[k], where hypothesis is a discrete distribution (list of probabilities)
    over the K classes, and to make a class prediction using the hypothesis.
    """

    def __init__(self, num_features, num_classes, learning_rate = 1e-2):
        "*** YOUR CODE HERE ***"

    def get_features(self, x):
        "*** YOUR CODE HERE ***"

    def get_weights(self):
        "*** YOUR CODE HERE ***"

    def hypothesis(self, x):
        "*** YOUR CODE HERE ***"

    def predict(self, x):
        "*** YOUR CODE HERE ***"

    def loss(self, x, y):
        "*** YOUR CODE HERE ***"

    def gradient(self, x, y):
        "*** YOUR CODE HERE ***"

    def train(self, dataset, evalset = None):
        "*** YOUR CODE HERE ***"


# PA4 Q6
def multi_classification():
    "*** YOUR CODE HERE ***"


def main():
    #linear_regression()
    binary_classification()
    multi_classification()

if __name__ == "__main__":
    main()