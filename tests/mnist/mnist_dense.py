from gradnet import Input, Model
from gradnet.layers import Dense
from gradnet.activations import get_activation
from gradnet.optimizers import get_optimizer
from gradnet.losses import Loss
from gradnet.metrics import get_metric

import numpy as np

class Callback(object):
    
    def __init__(self, print_every=5000, alpha=0.1):
        self.RunningLoss = self.RunningAccuracy = None
        self.NextPrint = self.PrintEvery = print_every
        
    def train_batch_end(self, nsamples, loss_values, metrics):
        if nsamples >= self.NextPrint:
            print("nsamples:", nsamples, "   cce loss:", loss_values["cce"], "   accuracy:", metrics[0])
            self.NextPrint += self.PrintEvery

class Callback(object):
    
    def __init__(self, print_every=5000, alpha=0.1):
        self.RunningLoss = self.RunningAccuracy = None
        self.NextPrint = self.PrintEvery = print_every
        
    def train_batch_end(self, nsamples, loss_values, metrics):
        if nsamples >= self.NextPrint:
            print("nsamples:", nsamples, "   cce loss:", loss_values["cce"], "   accuracy:", metrics[0])
            self.NextPrint += self.PrintEvery

    
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def one_hot(labels, n):
    out = np.zeros((len(labels), n))
    for i in range(n):
        out[labels==i, i] = 1.0
    return out
    
x_train = (x_train/256.0).reshape((-1, 28*28))
x_test = (x_test/256.0).reshape((-1, 28*28))
n_train = len(x_train)
y_train = one_hot(y_train, 10)
y_test = one_hot(y_test, 10)

np.set_printoptions(precision=4, suppress=True)

sgd = get_optimizer("SGD", learning_rate=0.01, momentum=0.5)
adam = get_optimizer("adam")
#ad = get_optimizer("ad", learning_rate=0.1)
accuracy = get_metric("accuracy")

def create_model():
    inp = Input((28*28,))
    dense1 = Dense(1024, name="dense1", activation="relu")(inp)
    probs = Dense(10, activation="softmax", name="top")(dense1)
    model = Model([inp], [probs])
    
    model.add_loss(Loss("cce", probs), name="cce")
    return model

model = create_model()
model.compile(adam, metrics=[accuracy])

mbsize = 100

for epoch in range(10):
    #print("main: x:", x_train[:3], "   y:", y_train[:3])
    model.fit(x_train, y_train, batch_size=mbsize, metrics=[accuracy], callbacks=[Callback()])
    y = model.compute(x_test)
    y_ = y_test
    acc = accuracy(y_test, y[0])
    print("test accuracy:", acc, "   losses:", model.LossValues)
    













    

