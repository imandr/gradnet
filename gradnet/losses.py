import numpy as np

class Loss(object):
    
    def __init__(self, input):
        self.Input = input
        self.Grads = None
        self.Value = None
        
    def backprop(self):
        self.Input.backprop(self.Grads)

    def compute(self, y_=None):
        raise NotImplementedError()
        
class MSE(Loss):
    
    def compute(self, y_):
        diff = self.Input.Y - y_
        self.Value = np.mean(diff**2, axis=-1)/len(y_)
        self.Grads = diff*2/y_.shape[-1]          
        return self.Value
        
class CategoricalCrossEntropy(Loss):
    
    def compute(self, p_):
        inp = self.Input
        assert len(inp.Xs) == 1
        x = inp.Xs[0]
        p = inp.Y
        self.Value = -np.sum(p_*np.log(p), axis=-1)/len(p_)
        self.Grads = p - p_
        return self.Value
        
    def backprop(self):
        self.Input.Inputs[0].backprop(self.Grads)
                
def get_loss(name):
    return {
        "mse":  MSE,
        "cce":  CategoricalCrossEntropy
    }[name]