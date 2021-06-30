from gradnet import Input, Model
from gradnet.layers import Dense, Conv2D, Pool, Flatten
from gradnet.activations import get_activation
from gradnet.optimizers import get_optimizer
from gradnet.losses import get_loss
from gradnet.metrics import get_metric

import numpy as np
    
from tensorflow.keras.datasets import mnist

accuracy = get_metric("accuracy")


def create_model():
    relu = get_activation("relu")
    sgd = get_optimizer("SGD", learning_rate=0.01, momentum=0.5)
    cce = get_loss("cce")
    mse = get_loss("mse")

    inp = Input((28,28,1))
    conv1 = Conv2D(3,3,32)(inp)
    pool1 = Pool(2,2, "max")(conv1)
    conv2 = Conv2D(3,3,64)(pool1)
    pool2 = Pool(2,2, "max")(conv2)
    flat = Flatten()(pool2)
    top = Dense(10, name="top")(flat)
    probs = get_activation("softmax", name="softmax")(top)
    l = cce(probs)

    model = Model([inp], [probs])
    model.compile(sgd, output_losses=[l])
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

for epoch in range(10):
    for i in range(0, n_train, mbsize):
        batch = x_train[i:i+mbsize]
        labels = y_train[i:i+mbsize]
        p, losses, metrics = model.train(batch, labels, [accuracy])
        print(np.mean(losses[0]), metrics[0])
        
        
    y = model.call(x_test)
    y_ = y_test
    acc = accuracy(y_test, y[0])
    print("test accuracy:", acc)
    













    

