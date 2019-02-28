import mnist_loader
import numpy as np
import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    sg = sigmoid(x)
    return sg * (1 - sg)


def partition(iterable, n):
    """Partitions the input into batches of size n."""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

alpha = 3.0
num_iter = 10
batch_size = 10

# https://stackoverflow.com/a/47241066/5107319
w1 = np.random.randn(30, 784)
w2 = np.random.randn(10, 30)
b1 = np.random.randn(30, 1)
b2 = np.random.randn(10, 1)

for i in xrange(num_iter):
    # Apply stochastic gradient descent.
    random.shuffle(training_data)
    for batch in partition(training_data, batch_size):
        grad_w1, grad_w2 = np.zeros((30, 784)), np.zeros((10, 30))
        grad_b1, grad_b2 = np.zeros((30, 1)), np.zeros((10, 1))
        for input in batch:
            # Feedforward
            l1_output = np.dot(w1, input[0]) + b1
            l1_activation = sigmoid(l1_output)
            l2_output = np.dot(w2, l1_activation) + b2
            l2_activation = sigmoid(l2_output)

            # --- [ Layer 2 ] -----------------------------------------------
            l2_error = l2_activation - input[1]

            # Backpropagation
            l2_delta = l2_error * sigmoid_prime(l2_output)
            grad_w2 += (alpha / batch_size) * np.dot(l2_delta, l1_activation.T)
            grad_b2 += (alpha / batch_size) * l2_delta

            # --- [ Layer 1 ] -----------------------------------------------
            l1_error = np.dot(w2.T, l2_delta)

            # Backpropagation
            l1_delta = l1_error * sigmoid_prime(l1_output)
            grad_w1 += (alpha / batch_size) * np.dot(l1_delta, input[0].T)
            grad_b1 += (alpha / batch_size) * l1_delta

        w1 -= grad_w1
        w2 -= grad_w2
        b1 -= grad_b1
        b2 -= grad_b2

    correct = 0
    for input in test_data:
        l1_activation = sigmoid(np.dot(w1, input[0]) + b1)
        l2_activation = sigmoid(np.dot(w2, l1_activation) + b2)
        if (np.argmax(l2_activation) == input[1]):
            correct += 1
    print("accuracy: %f" % ((100.0 * correct) / len(test_data)))
