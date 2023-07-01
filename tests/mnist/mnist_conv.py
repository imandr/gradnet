from gradnet import Input, Model, ModelClient
from gradnet.layers import Dense, Conv2D, Pool, Flatten
from gradnet.activations import get_activation
from gradnet.optimizers import get_optimizer
from gradnet.losses import get_loss, Loss
from gradnet.metrics import get_metric

import numpy as np, getopt, sys, math
    
from tensorflow.keras.datasets import mnist

accuracy = get_metric("accuracy")

opts, args = getopt.getopt(sys.argv[1:], "m:")
opts = dict(opts)

model_server_url = opts.get("-m")

def create_model():
    relu = get_activation("relu")
    #cce = get_loss("cce")
    #mse = get_loss("mse")

    inp = Input((28,28,1))
    conv1 = Conv2D(3,3,32, activation="relu")(inp)
    pool1 = Pool(2,2, "max")(conv1)
    conv2 = Conv2D(3,3,64, activation="relu")(pool1)
    pool2 = Pool(2,2, "max")(conv2)
    flat = Flatten()(pool2)
    top = Dense(10, name="top")(flat)
    probs = get_activation("softmax", name="softmax")(top)
    model = Model([inp], [probs])
    model.add_loss(Loss("cce", probs), name="cce")
    #sgd = get_optimizer("SGD", learning_rate=0.01, momentum=0.5)
    model.compile(get_optimizer("adam"), metrics=[accuracy])
    
    return model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def one_hot(labels, n):
    out = np.zeros((len(labels), n))
    for i in range(n):
        out[labels==i, i] = 1.0
    return out
    
x_train = (x_train/256.0).reshape((-1,28,28,1))
x_test = (x_test/256.0).reshape((-1,28,28,1))
n_train = len(x_train)
y_train = one_hot(y_train, 10)
y_test = one_hot(y_test, 10)

np.set_printoptions(precision=4, suppress=True)

model = create_model()
mbsize = 100

class Callback(object):
    
    def __init__(self):
        self.NextPrint = self.PrintEvery = 100
        self.LossMA = None
        self.Alpha = 0.1
    
    def train_batch_end(self, samples, losses, metrics):
        loss = losses["cce"]
        if self.LossMA is None:
            self.LossMA = loss
        self.LossMA += self.Alpha * (loss - self.LossMA)
        if samples >= self.NextPrint:
            #print(f"Samples: {samples}, loss:{loss}, MA:{self.LossMA}, metrics:{metrics}")
            self.NextPrint += self.PrintEvery
            
class SyncModelCallback(object):

    def __init__(self, model, model_client):
        self.NextSync = self.SyncEvery = 1000
        self.LossMA = None
        self.AccMA = None
        self.Alpha = 0.1
        self.ModelClient = model_client
        self.Model = model
    
    def train_batch_end(self, samples, losses, metrics):
        loss = losses["cce"]
        accuracy = metrics[0]
        if self.LossMA is None:
            self.LossMA = loss
        if self.AccMA is None:
            self.AccMA = accuracy
        self.LossMA += self.Alpha * (loss - self.LossMA)
        self.AccMA += self.Alpha * (accuracy - self.AccMA)
        if samples > self.NextSync and self.ModelClient is not None:
            weights = self.Model.get_weights()
            #r = -math.log(1.0-self.AccMA)
            r = -self.LossMA
            weights = self.ModelClient.update_weights(weights, r)
            self.Model.set_weights(weights)
            self.NextSync += self.SyncEvery
            print("Weights synced. Samples:", samples, "   Accuracy MA:", self.AccMA)

callbacks = [Callback()]
if model_server_url:
    client = ModelClient("mnist_conv", model_server_url)
    callbacks.append(SyncModelCallback(model, client))
    weights = client.get_weights()
    if weights:
        model.set_weights(weights)
        print("Model weights initialized from the server")

for epoch in range(10):

    losses, metrics = model.fit(x_train, y_train, batch_size=30, metrics=[accuracy], callbacks=callbacks, shuffle=True)
        
    y = model.compute(x_test)
    y_ = y_test
    acc = accuracy(y_test, y[0])
    print("test accuracy:", acc)
    













    

