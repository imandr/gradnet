import numpy as np
from .util import make_list
from .activations import SoftMaxActivation

class Loss(object):
    
    def __init__(self, *inputs, name=None):
        self.Name = name
        if isinstance(inputs[0], list):
            self.Inputs = inputs[0]
        else:
            self.Inputs = list(inputs)
        self.Grads = None
        self.Values = None
        self.Context = None
        
    def __str__(self):
        name = self.Name or self.__class__.__name__
        return f"[Loss {name}]"
        
    @property
    def value(self):
        return np.mean(self.Values)

    def backprop(self, weight=1.0):
        assert isinstance(self.Grads, list) and len(self.Grads) == len(self.Inputs),\
            f"{self}: self.Grads:%s, len=%s" % (type(self.Grads), len(self.Grads))
        #print(f"{self}.backprop: self.Grads:", [g.shape if g is not None else "" for g in self.Grads])
        for i, g in zip(self.Inputs, self.Grads):
            #print(self,".backprop: i,g:", i, g)
            if g is not None:
                i.backprop(g*weight)

    def compute(self, data={}):
        # data is a dictionary { "item name" -> ndarray }
        raise NotImplementedError()
        
class MSE(Loss):
    
    def compute(self, data):
        y_ = data["y_"]
        diff = self.Inputs[0].Y - y_
        #print("MSE.compute: input y:", self.Inputs[0].Y.shape, "   y_:", y_.shape)
        self.Values = np.mean(diff**2, axis=-1)          
        self.Grads = [diff*2]          
        return self.value
        
class CategoricalCrossEntropy(Loss):
    
    def compute(self, data):
        p_ = data["y_"]
        inp = self.Inputs[0]
        p = inp.Y
        
        #print("cce: p:", p)
        self.Values = -np.sum(p_*np.log(np.clip(p, 1e-6, None)), axis=-1)             
        #print("CCE.values:", self.Values.shape, "  p:",p.shape, "  p_:", p_.shape)
        
        inp = self.Inputs[0]
        if isinstance(inp.Layer, SoftMaxActivation):
            # if the input layer is SoftMax activation, bypass it and send simplified grads to its input
            #print("CategoricalCrossEntropy: sending simplified grads")
            self.Grads = p - p_
        else:
            self.Grads = -p_/np.clip(p, 1.0e-2, None)
        #print("CCE.compute(): p:", p)
        #print("              p_:", p_)
        return self.value
        
    def backprop(self, weight=1.0):
        inp = self.Inputs[0]
        if isinstance(inp.Layer, SoftMaxActivation):
            # if the input layer is SoftMax activation, bypass it and send simplified grads to its input
            inp.Inputs[0].backprop(self.Grads*weight)
        else:
            #print("CategoricalCrossEntropy: sending grads to:", inp, inp.Layer)
            inp.backprop(self.Grads*weight)
                
def get_loss(name):
    return {
        "mse":  MSE,
        "cce":  CategoricalCrossEntropy
    }[name]