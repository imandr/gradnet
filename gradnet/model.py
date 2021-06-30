from .util import make_list
    
class Model(object):
    
    def __init__(self, inputs, outputs):
        self.Inputs = make_list(inputs)        # inputs
        self.Outputs = make_list(outputs)      # paths
        self.Y = None
        self.Losses = []
        self.OutLosses = []
        self.Optimizer = None
        self.Metrics = []
        self.LossValues = []
        self.OutLossValues = []
        self.AllLayers = []
        seen = set()
        for out in self.Outputs:
            self.AllLayers += list(self._layers_rec(out, seen))

    def compile(self, optimizer=None, output_losses = [], losses=[], metrics=[]):
        self.Losses = losses
        self.OutLosses = output_losses
        self.Metrics = metrics
        if optimizer is not None:
            for layer in self.AllLayers:
                layer.Optimizer = optimizer()
        
    def _layers_rec(self, link, seen):
        layer = link.Layer
        if layer is not None and not id(layer) in seen:
            seen.add(id(layer))
            yield layer
        for lnk in link.Inputs:
            yield from self._layers_rec(lnk, seen)

    @property
    def layers(self):
        return self.AllLayers
        
    def call(self, inputs):
        inputs = make_list(inputs)
        assert len(inputs) == len(self.Inputs)
        for o in self.Outputs:
            o.reset()
        for i, x in zip(self.Inputs, inputs):
            i.set(x)
        return [o.compute() for o in self.Outputs]
        
    def fit(self, y_):
        y_ = make_list(y_)
        self.LossValues = [l.compute() for l in self.Losses]
        self.OutLossValues = [l.compute(yi_) for l, yi_ in zip(self.OutLosses, y_)]
        for l in self.Losses + self.OutLosses:
            l.backprop()
            
        for layer in self.AllLayers:
            layer.apply_deltas()
            
    def train(self, x, y_, metrics=[]):
        y_ = make_list(y_)
        x = make_list(x)
        y = self.call(x)
        self.fit(y_)
        return y, self.OutLossValues + self.LossValues, [m(y_, y[0]) for m in metrics] 
        
if __name__ == "__main__":
    from graphs import Input
    from layers import Dense
    from activations import get_activation
    from optimizers import get_optimizer
    from losses import get_loss
    from metrics import get_metric
    import numpy as np
    
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
    
    relu = get_activation("relu")
    sgd = get_optimizer("SGD", learning_rate=0.005, momentum=0.5)
    cce = get_loss("cce")
    mse = get_loss("mse")
    accuracy = get_metric("accuracy")
    
    inp = Input((28*28,))
    dense1 = Dense(1024, name="dense1", activation="relu")(inp)
    top = Dense(10, name="top")(dense1)
    probs = get_activation("softmax", name="softmax")(top)
    l = cce(probs)
    
    model = Model([inp], [probs])
    model.compile(sgd, output_losses=[l])

    mbsize = 30

    for epoch in range(10):
        for i in range(0, n_train, mbsize):
            batch = x_train[i:i+mbsize]
            labels = y_train[i:i+mbsize]
            p, losses, _ = model.train(batch, labels, [])
            
            
        y = model.call(x_test)
        y_ = y_test
        acc = accuracy(y_test, y[0])
        print("test accuracy:", acc)
        
    
    
    
    
    
    
    
    
    
    
    
    

        

    