from .layer import Layer
import numpy as np
from .util import make_list

class Activation(Layer):
    
    params = []
    
    def configure(self, inputs):
        assert len(inputs) == 1
        return inputs[0].Shape
        #print(self,".configure(): shape->", self.Shape)

    check_configuration = configure
    
    def compute(self, inputs, in_state=None):
        raise NotImplementedError()
        return y, None, None
        
    def grads(self, y_grads, out_state_grads, xs, y, context):
        raise NotImplementedError()
        return [x_grads], None, None
        
class LinearActivation(Activation):


    def compute(self, inputs, in_state=None):
        inputs = make_list(inputs)
        assert len(inputs) == 1
        return inputs[0], None, None
                
    def grads(self, y_grads, out_state_grads, xs, y, context):
        return [y_grads], None, None
    

class TanhActivation(Activation):
    
    def compute(self, xs, in_state=None):
        #print("tanh.call: x:", xs)
        xs = make_list(xs)
        assert len(xs) == 1
        x = xs[0]
        return np.tanh(x), None, None
        
    def grads(self, y_grads, out_state_grads, xs, y, context):
        #print("tanh.grads: xs:", xs, "  y:", y, "   y_grads:", y_grads)
        return [(1.0-y**2)*y_grads], None, None
        

class ReLUActivation(Activation):
    
    def compute(self, xs, in_state=None):
        xs = make_list(xs)
        assert len(xs) == 1
        x = xs[0]
        return (x + np.abs(x))/2, None, None
        
    def grads(self, y_grads, out_state_grads, xs, y, context):
        return [(y>0.0)*y_grads], None, None
        
class SoftMaxActivation(Activation):
    
    def compute(self, xs, in_state=None):
        xs = make_list(xs)
        assert len(xs) == 1
        x = xs[0]
        exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp/(np.sum(exp, axis=-1, keepdims=True)), None, None
        
    def grads(self, y_grads, out_state_grads, xs, y, context):
        n = y.shape[-1]
        eye = np.eye(n)[None,:,:]
        z = eye - y[:,:,None]
        jac = y[:,None,:]*z
        x_grads = np.einsum("mij,mi->mj", jac, y_grads)
        #print("SoftMaxActivation.grads: x:", xs[0], "  y_grads:", y_grads, "  x_grads:", x_grads)
        return [x_grads], None, None
    
def get_activation(kind, *params, **args):
    a = {
        "linear":   LinearActivation,
        "tanh":     TanhActivation,
        "softmax":  SoftMaxActivation,
        "relu":     ReLUActivation
    }[kind]
    return a(*params, **args)
