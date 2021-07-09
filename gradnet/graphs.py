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
            self.Layer.reset_gradients()
        
    def reset(self):
        #
        # get ready for next compute(), but do not reset accumulated grads
        #
        self.Xs = self.Y = self.InState = self.OutState = self.Context = None
        for i in self.Inputs:
            i.reset()
            
    def reset_gradients(self):
        self.StateGrads = None
        if self.Layer is not None:
            self.Layer.reset_gradients()
        for i in self.Inputs:
            i.reset_gradients()
        
    def compute(self):
        if self.Xs is None:
            self.InState = self.OutState
            self.Xs = [i.compute() for i in self.Inputs]
            self.Y, self.OutState, self.Context = self.Layer.compute(self.Xs, self.InState)
        return self.Y

    def backprop(self, y_grads):
        #assert isinstance(y_grads, np.ndarray) and y_grads.shape[1:] == self.Shape
        xgrads, self.StateGrads = self.Layer.backprop(y_grads, self.StateGrads, self.Xs, self.Y, self.Context)
        assert isinstance(xgrads, list) and len(xgrads) == len(self.Inputs)
        #print(f"    x_grads from layer {self.Layer}:", [g.shape if g is not None else "" for g in xgrads])
        for xg, i in zip(xgrads, self.Inputs):
            if xg is not None:
                i.backprop(xg)
            
class Input(Link):
    def __init__(self, shape, name=None):
        self.Shape = shape      # tensor shape without the minibatch dimension
        self.Values = None
        self.XGradSum = None
        self.Inputs = []
        self.Layer = None
        self.Name = name
        
    def __str__(self):
        name = self.Name or "(unnamed)"
        return f"[Input {name} {self.Shape}]"
        
    def set(self, values):
        assert values.shape[1:] == self.Shape, "Incompatible shape for input layer. Expected %s, got %s" % (('*',)+self.Shape, values.shape)
        self.Values = values
        self.XGradSum = np.zeros(values.shape)
        
    def compute(self):
        return self.Values
        
    def backprop(self, grads):
        self.XGradSum[...] += grads
        
    def reset_gradsients(self):
        pass
        
