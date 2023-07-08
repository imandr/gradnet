import numpy as np
from .util import make_list
from .activations import SoftMaxActivation
from types import FunctionType

class LossBase(object):
    
    def __init__(self, *inputs, name=None, callable=None):
        if name is None:
            if callable is not None:
                if hasattr(callable, "Name"):
                    name = callable.Name
                elif hasattr(callable, "__name__"):
                    name = callable.__name__
                elif hasattr(callable, "__class__"):
                    name = callable.__class__.__name__
            else:
                name = self.__class__.__name__
                
        #print("LossBase create: name:", name)

        self.Name = name
        self.Inputs = inputs
        self.Grads = None
        self.Values = None
        self.Callable = callable
        
    def __str__(self):
        if self.Name:   
            name = self.Name
        else:
            name = self.__class__.__name__
        return f"[Loss {name}]"
        
    def backprop(self, weight=1.0):
        assert isinstance(self.Grads, list) and len(self.Grads) == len(self.Inputs),\
            f"{self}: self.Grads:%s, len=%s" % (type(self.Grads), len(self.Grads))
        #print(f"{self}.backprop: self.Grads:", [g.shape if g is not None else "" for g in self.Grads])
        #print(self, "inputs:", self.Inputs, "   grads:", self.Grads)
        for i, g in zip(self.Inputs, self.Grads):
            #print(self,".backprop: i,g:", i, g)
            if g is not None:
                i.backprop(g*weight)
                
    @property
    def value(self):
        return np.sum(self.Values)

    def compute(self, data={}):
        f = self if self.Callable is None else self.Callable
        #print(f"Loss {self.Name}: calling f with:", *[i.Y for i in self.Inputs], " and data{%s}" % (list(data.keys()),))
        values, grads = f(data.get("y_"), *[i.Y for i in self.Inputs], data)
        grads = make_list(grads)
        if grads is None:
            grads = [None]
        self.Values, self.Grads = values, grads
        return self.value
        
    @staticmethod
    def from_function(f, *inputs, name=None):
        if name is None:
            if f is not None:
                if hasattr(f, "Name"):
                    name = callable.Name
                elif hasattr(f, "__name__"):
                    name = f.__name__
                elif hasattr(f, "__class__"):
                    name = f.__class__.__name__
        return LossBase(*inputs, name=name, callable=f)
        
class CategoricalCrossEntropy(LossBase):
    
    def __call__(self, y_, y, data):
        p_ = y_
        p = y
        
        #print("cce: p:", p)
        values = -np.sum(p_*np.log(np.clip(p, 1e-6, None)), axis=-1)             
        #print("CCE.values:", self.Values.shape, "  p:",p.shape, "  p_:", p_.shape)
        
        inp = self.Inputs[0]
        if isinstance(inp.Layer, SoftMaxActivation):
            # if the input layer is SoftMax activation, bypass it and send simplified grads to its input
            #print("CategoricalCrossEntropy: sending simplified grads")
            grads = p - p_
        else:
            grads = -p_/np.clip(p, 1.0e-2, None)
        #print("CCE.compute(): p:", p)
        #print("              p_:", p_)
        return values, grads
        
    def backprop(self, weight=1.0):
        inp = self.Inputs[0]
        if isinstance(inp.Layer, SoftMaxActivation):
            # if the input layer is SoftMax activation, bypass it and send simplified grads to its input
            inp.Inputs[0].backprop(self.Grads[0]*weight)
        else:
            #print("CategoricalCrossEntropy: sending grads to:", inp, inp.Layer)
            inp.backprop(self.Grads[0]*weight)
            
class LossStubForFunction(object):
    
    def __init__(self, f):
        self.F = f
        #print("stub: function.__name__:", f.__name__)
        
    def __call__(self, *inputs, **args):
        #print("LossStubForFunction.__call__: inputs:", inputs)
        return LossBase.from_function(self.F, *inputs, **args)
        
def mse(y_, y, data):
    diff = y - y_
    return np.sum(diff**2, axis=-1), diff*2
        
def zero(y_, y, data):
    z = np.zeros_like(y)
    return z, None
                
loss_stubs = {
        "mse":  LossStubForFunction(mse),
        "zero": LossStubForFunction(zero),
        "cce":  CategoricalCrossEntropy
}
        
def Loss(*params, **args):
    if isinstance(params[0], str):
        name = params[0]
        if not name in loss_stubs:
            raise ValueError(f"Loss {name} not found")
        return loss_stubs[name](*params[1:], **args)
    elif isinstance(params[0], FunctionType):
        return LossBase.from_function(*params, **args)
    else:
        raise ValueError(f"Can not create Loss object for {params}")
            
def get_loss(name):
    print("get_loss(name) is deprecated. Please use Loss(name)")
    return Loss(name)