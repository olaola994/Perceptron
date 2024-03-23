import random
import numpy as np
import tkinter as tk
from tkinter import filedialog

data = []
train_set = []
test_set = []

X_train = []
Y_train = []

X_test = []
Y_test = []

print('Please choose option for your KNN classifier\nType:\n-"file" if you want to choose a .txt file with your data set\n-"custom" if you want to type your own vectors')
option = input()
if option == 'file':
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'r') as file:
            for line in file:
                elements = line.strip().split(',')
                data.append(elements)

    random.shuffle(data)

    train_set = data[:len(data) // 2]
    test_set = data[len(data) // 2:]

    for line in train_set:
        X_train.append([float(val) for val in line[:len(line) - 1]])
        Y_train.append(int(line[len(line) - 1]))


    for line in test_set:
        X_test.append([float(val) for val in line[:len(line) - 1]])
        Y_test.append(int(line[len(line) - 1]))

elif option == 'custom':
    vectors = []
    with open('file.txt', 'r') as file:
        for line in file:
            elements = line.strip().split(',')
            data.append(elements)

    random.shuffle(data)

    train_set = data[:len(data) // 2]

    X_train = []
    Y_train = []
    for line in train_set:
        X_train.append([float(val) for val in line[:len(line) - 1]])
        Y_train.append(int(line[len(line) - 1]))

    X_test = []
    print("Please enter your vectors. Each vector should be in the format 'x1,x2,x3,x4', e.g., '6.3,3.3,4.7,1.6. Press Enter twice to finish.")
    while True:
        line = input()
        if not line:
            break
        elements = line.strip().split(',')
        if (len(elements) < 2):
            print("Each vector should contain more than 2 elements separated by commas.")
            continue
        vectors.append([float(val) for val in elements])
    X_test.extend(vectors)



class Perceptron:
    def __init__(self, input_size, alpha=0.02,  epochs=10):
        self.input_size = input_size
        self.epochs = epochs
        self.weights = np.zeros(input_size + 1)
        self.weights[-1] = 1
        self.alpha = alpha

    def activation(self, sum, W):
        return 1 if sum >= W[-1] else 0

    def weighted_sum(self, X, W):
        w_sum = 0
        for weight, inp in zip(W[:-1], X):
            w_sum += weight * inp
        return w_sum

    def predict(self, inp):
        w_sum = self.weighted_sum(inp, self.weights)
        return self.activation(w_sum, self.weights)

    def train(self, x_training_inp, labels):
        for _ in range(self.epochs):
            for i, x in enumerate(x_training_inp):
                d = labels[i]
                y = self.predict(x)
                err = d - y
                for j, (weight, inp) in enumerate(zip(self.weights, x)):
                    self.weights[j] += err * self.alpha * inp

    def test(self, x_test_inp):
        predictions = []
        for i, x in enumerate(x_test_inp):
            prediction = self.predict(x)
            predictions.append(prediction)
            print(str(x) + " " + str(predictions[i]))


perceptron = Perceptron(4)
perceptron.train(X_train, Y_train)
perceptron.test(X_test)
