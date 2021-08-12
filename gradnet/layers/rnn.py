from .layer import Layer
from ..util import make_list
import numpy as np
from numpy.random import default_rng

rng = default_rng()


class RNNLayer(Layer):
    
    def __init__(self, return_sequences=True, reversed=False, **args):
        Layer.__init__(self, **args)
        self.Reversed = reversed
        self.ReturnSequences = return_sequences
        #print("RNNLayer.__init__: reversed=", self.Reversed)
        
    def init_state(self, mb):
        raise NotImplementedError()
        return initial_state
    
    def forward(self, t, x, s, context):
        # return y, c
        raise NotImplementedError()
        return y, c, context
        
    def backward(self, t, gy_t, gstate_t, gw, context):
        # given dL/dc = gc and dL/dy = gy and accumulated dL/dw = gw return dL/dx, dL/ds and updated dL/dw
        # initial gw is None
        raise NotImplementedError()
        return gx, gw, gs       # gx is ndarray, not list !
        
    def init_context(self, x, state_in):
        return None
        
    def compute(self, xs, state_in=None):
        xs = make_list(xs)
        assert len(xs) == 1
        x = xs[0]
        assert len(x.shape) == 3, f"Invalid input shape for RNN: {x.shape}. Expected 3 dimensions"       # at [mb, t, w]
        x = x.transpose((1,0,2))
        n, mb, xw = x.shape
        assert n > 0
        if state_in is None:
            state_in = self.init_state(mb)            
        context = self.init_context(x, state_in)
        s = state_in
        y = []
        for i in range(n):
            t = n-1-i if self.Reversed else i
            xt = x[t]
            yt, c, context = self.forward(t, xt, s, context)
            s = c
            if self.ReturnSequences:
                y.append(yt)
        if self.ReturnSequences:
            y = np.array(y).transpose((1,0,2))
        else:
            y = yt
        return y, c, context
        
    def grads(self, y_grads, s_out_grads, xs, y, context):
        assert len(xs) == 1
        x = xs[0]
        assert len(x.shape) == 3       # at [mb, t, w]
        x = x.transpose((1,0,2))
        n, mb, xw = x.shape
        assert n > 0
        
        if s_out_grads is None:
            s_out_grads = np.zeros_like(self.init_state(mb))
        
        if not self.ReturnSequences:
            yg = np.zeros((n,)+y_grads.shape)
            yg[-1] = y_grads
            y_grads = yg
        else:
            y_grads = y_grads.transpose((1,0,2))
            
        x_grads = np.zeros_like(x)
        #print("rnn.grads: x_grads:", x_grads.shape)
        gw = [np.zeros_like(w) for w in self.params]
        gc = s_out_grads
        for i in range(n):
            t = i if self.Reversed else n-1-i
            #print("RNNLayer.grads: t=", t)
            gyt = y_grads[t]
            gxt, gw, gs = self.backward(t, gyt, gc, gw, context)
            #print("rnn.grads: x_grads[t,:,:]", x_grads[t,:,:].shape, "   gxt:", gxt.shape)
            x_grads[t,:,:] = gxt
            gc = gs
        
        return [x_grads.transpose((1,0,2))], gw, gc
        

class RNN(RNNLayer):
    
    def __init__(self, out_size, recurrent_size=None, return_sequences=True, **args):
        
        RNNLayer.__init__(self, return_sequences=return_sequences, reversed=reversed, **args)
        
        self.Nout = out_size
        self.NC = recurrent_size or out_size
        self.ReturnSequences = return_sequences
        self.Nin = None
        
    def configure(self, inputs):
        assert len(inputs) == 1
        inp = inputs[0]
        shape = inp.Shape
        #print("lstm.configure: inp.Shape:", inp.Shape)
        assert len(shape) == 2, f"Expected shape with 3 dimensions, got {shape}"
        self.Nin = shape[-1]

        #print("lstm.configure: Nin:", self.Nin, "  NC:", self.NC)

        self.W = rng.normal(size=(1 + self.Nin + self.NC, self.Nout + self.NC),
                scale= 1.0 / np.sqrt(self.Nin + 2*self.NC + self.Nout))
                
        #if self.Nin == self.NC:
        #    eye = np.eye(self.NC)
        #    #print("eye=", eye)
        #    for i in range(1,1 + self.Nin + self.NC,self.NC):
        #        for j in range(0, 4 * self.NC, self.NC):
        #            self.WLSTM[i:i+self.Nin, j:j+self.NC] = eye
        
        #self.WLSTM[...] = 1.0
        
        
        #print "WLSTM shape=", self.WLSTM.shape
        self.W[0,:] = 0 # initialize biases to zero
        
        return (shape[0], self.Nout) if self.ReturnSequences else (self.Nout,)
        
    @property
    def params(self):
        return [self.W]
        
    def set_weights(self, w):
        self.W = w[0]

    def init_state(self, mb):
        return np.zeros((mb, self.NC))
        
    def init_context(self, x, state_in):
        #print("init_context: x:", x.shape)
        n, b = x.shape[:2]
        d = self.NC
        
        U = np.empty((n, b, 1+d+self.Nin))
        U[:,:,0] = 1.0      # for bias
        U[:,:,1:1+self.Nin] = x
        #print("U:", U.shape)
        
        return U
        
    def forward(self, t, x, s, context):
        # return y, c
        #print("forward: x:", x.shape)
        b = x.shape[0]
        d = self.NC
        if s is None:
            s = np.zeros((b, d))
            #print("s initialized:", s.shape)

        U = context
        U[t,:,-d:] = s
        Vt = U[t].dot(self.W)
        #print("U:", U.shape, "  Vt:", Vt.shape)
        
        Yt = Vt[:,:self.Nout]
        Ct = Vt[:,self.Nout:]
        #print(Yt, Ct)
        return Yt, Ct, context
        
    def backward(self, t, gy_t, gstate_t, gw, context):
        # given dL/dc = gc and dL/dy = gy and accumulated dL/dw = gw return dL/dx, dL/ds and updated dL/dw
        # initial gw is None
        
        U = context
        d = self.NC

        dVt = np.concatenate([gy_t, gstate_t], axis=-1)
        #print("gy:", gy_t.shape, "  gstate:", gstate_t.shape, "  dVt:", dVt.shape)
        dUt = dVt.dot(self.W.T)

        gw = gw[0]
        gw += U[t].T.dot(dVt)        
        gx = dUt[:,1:1+self.Nin]
        gs = dUt[:,-d:]

        return gx, [gw], gs       # gx is ndarray, not list !
        
        
        
        
    