from .. import Layer
import numpy as np
import math

class Dense(Layer):

    def __init__(self, n, **args):
        Layer.__init__(self, **args)
        self.NOut = n
        self.NIn = None
        self.B = self.W = None
        
    def __str__(self):
        name = self.Name or "(unnamed)"
        return f"[Dense {name} {self.NIn}->{self.NOut}]"
        
    def configure(self, inputs):
        assert len(inputs) == 1
        inp = inputs[0]
        self.NIn = inp.Shape[-1]
        #print(self,".configure(): shape -> ", self.Shape)
        self.W = np.random.random((self.NIn, self.NOut)) - 0.5
        self.B = np.zeros((self.NOut,))
        return inp.Shape[:-1] + (self.NOut,)
        
    def check_configuration(self, inputs):
        assert len(inputs) == 1
        inp = inputs[0]
        assert inp.Shape[-1] == self.NIn
        return inp.Shape[:-1] + (self.NOut,)

    def _set_weights(self, weights):
        w, b = weights
        #print("Dense._set_weights: w,b:", w.shape, b.shape)
        assert w.shape == self.W.shape
        assert b.shape == self.B.shape
        self.W = w
        self.B = b 
        
    def get_weights(self):
        return [self.W, self.B]
        
    def call(self, xs, in_state=None):
        assert isinstance(xs, list) and len(xs) == 1, f"Dense layer accepts single input. Received: {xs}"
        x = xs[0]
        assert x.shape[-1] == self.NIn
        return np.dot(x, self.W) + self.B, None, None
        
    def grads(self, y_grads, s_out_grads, xs, y, context):
        #print(self, ".grads: y_grads:", y_grads.shape)
        assert isinstance(xs, list) and len(xs) == 1, "%s.grads(): invalid type of xs: %s %s" % (self, type(xs), xs)
        assert y_grads.shape[-1] == self.NOut
        x = xs[0]
        m = len(y_grads)        # batch size
        inp_flat = x.reshape((-1, self.NIn))
        g_flat = y_grads.reshape((-1, self.NOut))
        #print("Dense.grads: inp_flat:", inp_flat.shape, "  g_flat:", g_flat.shape)
        gW = np.dot(inp_flat.T, g_flat)    # [...,n_in],[...,n_out]
        #print("             gW:", gW.shape)
        gb = np.sum(g_flat, axis=0)
        gx = np.dot(y_grads, self.W.T)
        return [gx], [gW, gb], None

    @property
    def params(self):
        return self.W, self.B
        
class Flatten(Layer):
    
    params = []
    
    def configure(self, inputs):
        assert len(inputs) == 1
        inp = inputs[0]
        l = math.prod(inp.Shape)
        return (l,)
        
    check_config = configure
    
    def call(self, xs, in_state):
        x = xs[0]
        n = len(x)
        return x.reshape((n, -1)), None, x.shape
        
    def grads(self, y_grads, s_out_grads, xs, y, context):
        return [y_grads.reshape(context)], None, None
        
