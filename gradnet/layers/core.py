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
        limit = math.sqrt(6.0/(self.NIn + self.NOut))
        self.W = np.random.random((self.NIn, self.NOut))*2*limit - limit
        shape_out = inp.Shape[:-1] + (self.NOut,)
        self.B = np.zeros((self.NOut,))
        return shape_out
        
    def check_configuration(self, inputs):
        assert len(inputs) == 1
        inp = inputs[0]
        assert inp.Shape[-1] == self.NIn
        shape_out = inp.Shape[:-1] + (self.NOut,)
        return shape_out

    def _set_weights(self, weights):
        w, b = weights
        #print("Dense._set_weights: w,b:", w.shape, b.shape)
        assert w.shape == self.W.shape
        assert b.shape == self.B.shape
        self.W = w
        self.B = b 
        
    def get_weights(self):
        return [self.W, self.B]
        
    def compute(self, xs, in_state=None):
        assert isinstance(xs, list) and len(xs) == 1, f"Dense layer accepts single input. Received: {xs}"
        x = xs[0]
        assert x.shape[-1] == self.NIn
        dims = len(x.shape)
        b = self.B
        if dims > 1:
            b = b.reshape((1,)*(dims-1) + (-1,))
        return np.dot(x, self.W) + b, None, None
        
    def grads(self, y_grads, s_out_grads, xs, y, context):
        #print(self, ".grads: y_grads:", y_grads)
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
        #print(self, ".grads: gb:", gb)
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
    
    def compute(self, xs, in_state):
        x = xs[0]
        n = len(x)
        return x.reshape((n, -1)), None, x.shape
        
    def grads(self, y_grads, s_out_grads, xs, y, context):
        return [y_grads.reshape(context)], None, None
        
class Transpose(Layer):
    
    params = []
    def __init__(self, *map, name=None):
        Layer.__init__(self, name=name)
        assert all(x in map for x in range(len(map)))   # make sure each index appears once and only once
        self.MapWithoutBatch = map
        map = [0] + [m+1 for m in map]          # add th eminibatch dimension
        self.Map = tuple(map)
        rev = [0]*len(map)
        for i, m in enumerate(map):
            rev[m] = i
        self.Reversed = tuple(rev)
    
    def configure(self, inputs):
        assert len(inputs) == 1
        inp = inputs[0]
        assert len(inp.Shape) == len(self.MapWithoutBatch), \
                f"Transpose: input shape {inp.Shape} does not match the transpose index map {self.Map}"
        return tuple(inp.Shape[self.MapWithoutBatch[i]] for i in range(len(inp.Shape)))

    check_config = configure
    
    def compute(self, xs, in_state):
        assert len(xs) == 1
        return xs.transpose(self.Map)
        
    def grads(self, y_grads, s_out_grads, xs, y, context):
        return [y_grads.transpose(self.Reversed)], None, None
        
class Concatenate(Layer):
    
    params = []
    
    def __init__(self, axis = -1, name=None):
        Layer.__init__(self, name=name)
        self.Axis = axis

    def configure(self, inputs):
        assert len(inputs) >= 2
        shapes = [inp.Shape for inp in inputs]
        inp0 = inputs[0]
        assert all(len(inp.Shape) == len(inp0.Shape) for inp in inputs)
        ndims = len(inp0.Shape)
        axis = self.Axis if self.Axis >= 0 else self.Axis + ndims
        
        reduced = inp0.Shape[:axis] + inp0.Shape[axis+1:]
        assert all(inp.Shape[:axis] + inp.Shape[axis+1:] == reduced for inp in inputs)
        newdim = sum(inp.Shape[axis] for inp in inputs)
        out_shape = reduced[:axis] + (newdim,) + reduced[axis:]
        return out_shape
        
    check_config = configure
        
    def compute(self, xs, in_state):
        return np.concatenate(xs, axis=self.Axis), None, tuple(x.shape for x in xs)
        
    def grads(self, y_grads, s_out_grads, xs, y, context):
        x_shapes = context
        ndims = len(x_shapes[0])
        axis = self.Axis if self.Axis >= 0 else self.Axis + ndims
        selector = [slice(None)]*ndims
        i = 0
        x_grads = []
        for shape in x_shapes:
            n = shape[axis]
            selector[axis] = slice(i,i+n)
            i += n
            x_grads.append(y_grads[tuple(selector)])
        #print("concatente: ygrads:", y_grads," ---> ", x_grads)
        return x_grads, None, None
            
    
        
