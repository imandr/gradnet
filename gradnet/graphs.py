import numpy as np
from .util import make_list

class Computer(object):
    
    def compute(self):
        raise NotImplementedError()
        return y
        
    def reset_cached(self):
        pass
        
    def reset_state(self):
        pass
        
    def add_grads(self, y_grads):
        raise NotImplementedError()
    
    def apply_grads(self):
        raise NotImplementedError()

    def grads(self, y_grads):
        raise NotImplementedError()
        return x_grads, p_grads

    def call(self, x):
        raise NotImplementedError()
        return y
        
    def apply_deltas(self, deltas):
        raise NotImplementedError()
        
class Link(object):
    
    def __init__(self, layer, shape, inputs):
        self.Inputs = inputs
        self.Layer = layer
        self.Shape = shape
        self.Xs = self.Y = self.StateGrads = self.InState = self.OutState = self.Context = None
        if self.Layer is not None:
            self.Layer.reset_grads()
        
    def reset(self):
        self.Xs = self.Y = self.StateGrads = self.InState = self.OutState = self.Context = None
        if self.Layer is not None:
            self.Layer.reset_grads()
        for i in self.Inputs:
            i.reset()
        
    def compute(self):
        if self.Xs is None:
            self.InState = self.OutState
            self.Xs = [i.compute() for i in self.Inputs]
            self.Y, self.OutState, self.Context = self.Layer.compute(self.Xs, self.InState)
        return self.Y

    def backprop(self, y_grads):
        xgrads, self.StateGrads = self.Layer.backprop(y_grads, self.StateGrads, self.Xs, self.Y, self.Context)
        for xg, i in zip(xgrads, self.Inputs):
            i.backprop(xg)
            
class Input(Link):
    def __init__(self, shape):
        self.Shape = shape      # tensor shape without the minibatch dimension
        self.Values = None
        self.XGradSum = None
        self.Inputs = []
        self.Layer = None
        
    def set(self, values):
        assert values.shape[1:] == self.Shape, "Incompatible shape for input layer. Expected %s, got %s" % (('*',)+self.Shape, values.shape)
        self.Values = values
        self.XGradSum = np.zeros(values.shape)
        
    def compute(self):
        return self.Values
        
    def backprop(self, grads):
        self.XGradSum[...] += grads
        
    def reset_grads(self):
        pass
        
