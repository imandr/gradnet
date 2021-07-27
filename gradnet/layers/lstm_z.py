import numpy as np
import math, time
from ..util import make_list
from numpy.random import default_rng
from .rnn import RNNLayer

rng = default_rng()

class LSTM_Z(RNNLayer):
    
    def __init__(self, out_size, recurrent_size=None, **args):        
        RNNLayer.__init__(self, **args)
        
        recurrent_size = recurrent_size or out_size
        
        self.Nout = out_size
        self.NC = recurrent_size
        self.Nin = None
        
    def configure(self, inputs):
        assert len(inputs) == 1
        inp = inputs[0]
        shape = inp.Shape
        #print("lstm.configure: inp.Shape:", inp.Shape)
        assert len(shape) == 2, f"Expected shape with 3 dimensions, got {shape}"
        self.Nin = shape[-1]

        #print("lstm.configure: Nin:", self.Nin, "  NC:", self.NC)

        self.WIFN = rng.normal(size=(1 + self.Nin + self.NC, 3 * self.NC),
                scale= 1.0 / np.sqrt(self.Nin + self.NC))
        self.WZ = rng.normal(size=(1 + self.Nin + self.NC*2, self.Nout),
                scale= 1.0 / np.sqrt(2*self.NC + self.Nin + self.Nout))
                
        #self.WIFN[...] = 1
        #self.WZ[...] = 1
                
        self.WIFN[0,:] = 0 # initialize biases to zero
        self.WZ[0,:] = 0 # initialize biases to zero
        # forget gates get little bit negative bias initially to encourage them to be turned off
        # remember that due to Xavier initialization above, the raw output activations from gates before
        # nonlinearity are zero mean and on order of standard deviation ~1
        
        self.WIFN[0,self.NC:2*self.NC] = 3.0

        return (shape[0], self.Nout) if self.ReturnSequences else (self.Nout,)

    @property
    def params(self):
        return [self.WIFN, self.WZ]
        
    def set_weights(self, weights):
        self.WIFN, self.WZ = weights
        
    def check_config(self, inputs):
        assert len(inputs) == 1
        inp = inputs[0]
        shape = inp.Shape
        assert len(shape) == 2
        assert shape[-1] == self.Nin
        return self.Shape
        
    def init_state(self, mb):
        return np.zeros((mb, self.NC))
        
    def init_context(self, x, state_in):
        n, b, input_size = x.shape
        d = self.NC
        s0 = state_in if state_in is not None else np.zeros((b,d))
        
        w = self.WIFN.shape[0] # x plus h plus bias, lol
        
        V = np.zeros((n, b, w+d))           # <bias, x, s, c>
        V[:,:,1:input_size+1] = x
        V[:,:,0] = 1
        
        C = V[:,:,-d:]          # view 
        U = V[:,:,:-d]          # view, <bias, x, s>
        S = U[:,:,-d:]          # view, <s>
                

        IFN = np.zeros((n, b, d * 3))
        IFNf = np.zeros((n, b, d * 3))
        
        context = {
            "IFNf":     IFNf,
            "IFN":      IFN,
            "C":        C,
            "U":        U,
            "V":        V,
            "S":        S,
            "S0":       s0
        }
        return context
        
        
    def forward(self, t, xt, s, context):

        d = self.NC
        b, input_size = xt.shape

        IFNf = context['IFNf']
        IFN = context['IFN']
        V = context['V']
        C = context['C']        # view into V
        U = context['U']        # view into V
        S = context['S']        # view into V

        # views
        I = IFN[t,:,:d]
        F = IFN[t,:,d:d*2]
        IF = IFN[t,:,:-d]
        N = IFN[t,:,-d:]
        If = IFNf[t,:,:d]           
        Ff = IFNf[t,:,d:d*2]        
        IFf = IFNf[t,:,:-d]
        Nf = IFNf[t,:,-d:]      
        
        #print("WIFN:", self.WIFN)

        #print("WZ:", self.WZ)

        S[t] = s
        U[t]
        IFN[...] = U[t].dot(self.WIFN)
        
        #print("U[t]:", U[t])
        #print("IFN:", IFN)
        
        IFf[...] = 1.0/(1.0+np.exp(-IF)) # sigmoids; these are the gates
        Nf[...] = np.tanh(N) # tanh
        #print("IFNf:", IFNf)

        C[t] = If * Nf + Ff * S[t]
        #print("C[t]:", C[t])
        #print("IFNf:", IFNf)        
        #print("V[t]:", V[t])
        y = V[t].dot(self.WZ)
        #print("y=", y)
        
        return y, C[t], context
        
    def backward(self, t, gy_t, gstate_t, gw, context):
        
        #print("--- backward ---")
        
        #print("gY:", gy_t)

        d = self.NC
        b = gy_t.shape[0]
        if gstate_t is None:
            gstate_t = np.zeros((b, self.NC))

        input_size = self.Nin

        dWIFN, dWZ = gw
        
        #print("context:", context)

        IFNf = context['IFNf']
        IFN = context['IFN']
        V = context['V']
        C = context['C']        # view into V
        U = context['U']        # view into V
        S = context['S']        # view into V
        
        # views
        I = IFN[t,:,:d]
        F = IFN[t,:,d:d*2]
        IF = IFN[t,:,:-d]
        N = IFN[t,:,-d:]
        If = IFNf[t,:,:d]           
        Ff = IFNf[t,:,d:d*2]        
        IFf = IFNf[t,:,:-d]
        Nf = IFNf[t,:,-d:]      

        dIFNf = np.zeros((b, d*3))
        dIFN = np.zeros((b, d*3))
        
        dIf = dIFNf[:,:d]                  # views
        dIFf = dIFNf[:,:-d]
        dFf = dIFNf[:,d:2*d]
        dNf = dIFNf[:,2*d:]        

        dI = dIFN[:,:d]                  # views
        dIF = dIFN[:,:-d]
        dF = dIFN[:,d:2*d]
        dN = dIFN[:,2*d:]        

        dV = np.dot(gy_t, self.WZ.T)
        dC = gstate_t + dV[:,-d:]
        
        #print("dV:", dV)
        
        dU = dV[:,:-d]                  # view
        
        dS = Ff * dC
        dIf[...] = Nf * dC
        dNf[...] = If * dC
        dFf[...] = S[t] * dC
        
        # now we have all components of dIFNf
        #print("dIFNf:", dIFNf)
        
        # backpropagate through non-linearity
        dIF[...] = IFf*(1-IFf) * dIFf
        dN[...] = (1-Nf**2) * dNf

        #print("dIFN:", dIFN)

        dWIFN += np.dot(U[t].T, dIFN)
        dWZ += np.dot(V[t].T, gy_t)

        dU += dIFN.dot(self.WIFN.T)

        dx = dU[:,1:input_size+1]
        dS += dU[:,-d:] 

        return dx, [dWIFN, dWZ], dS
