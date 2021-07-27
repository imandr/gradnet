import numpy as np
import math, time
from .layer import Layer
from ..util import make_list
from numpy.random import default_rng

rng = default_rng()

class GRU(Layer):

    def __init__(self, out_size, return_sequences=True, reversed=False, **args):
        Layer.__init__(self, **args)
        self.Nout = self.NC = out_size
        self.ReturnSequences = return_sequences
        self.Nin = None
        self.Reversed = reversed
        
    def configure(self, inputs):
        assert len(inputs) == 1
        inp = inputs[0]
        shape = inp.Shape
        #print("lstm.configure: inp.Shape:", inp.Shape)
        assert len(shape) == 2, f"Expected shape with 3 dimensions, got {shape}"
        self.Nin = shape[-1]

        self.W = rng.normal(size=(self.Nin + 1 + self.NC, 3 * self.NC),         # f then N
                scale= 1.0 / np.sqrt(self.Nin + self.NC))
        self.W[0,:] = 0 # initialize biases to zero
        # forget gates get little bit negative bias initially to encourage them to be turned off
        # remember that due to Xavier initialization above, the raw output activations from gates before
        # nonlinearity are zero mean and on order of standard deviation ~1
        self.W[0,self.NC:-self.NC] = 3.0
        return (shape[0], self.Nout) if self.ReturnSequences else (self.Nout,)
        
    def check_config(self, inputs):
        assert len(inputs) == 1
        inp = inputs[0]
        shape = inp.Shape
        assert len(shape) == 2, f"Expected shape with 3 dimensions, got {shape}"
        assert self.Nin == shape[-1], f"Incompatible input. Expected input width {self.Nin}. Got {shape[-1]} instead"
        return self.Shape

    @property
    def params(self):
        return [self.W]
        
    def compute(self, xs, state_in):
        
        assert len(xs) == 1
        x = xs[0]
        
        # assume x in [batch, time, width]
        # transpose so that x is [time, batch, width]
        X = np.transpose(x, (1, 0, 2))

        n,b,input_size = X.shape
        d = self.NC # hidden size
        c0 = state_in if state_in is not None else np.zeros((b,d))
        if self.Reversed:
            X = X[::-1,:,:]

        xpcpb = input_size + 1 + d
        
        RF = np.zeros((n, b, d*2))          # forget and N
        RFf = np.zeros((n, b, d*2))         # after non-linearity
        
        N = np.empty((n,b,d))
        Nf = np.empty((n,b,d))
        
        U = np.empty((n, b, xpcpb))        # input + C + bias
        U[:,:,0] = 1
        U[:,:,1:input_size+1] = X
        
        V = np.empty((n, b, xpcpb))        # input + C + bias
        V[:,:,0] = 1
        V[:,:,1:input_size+1] = X
        
        Wrf = self.W[:,:-d]
        Wn = self.W[:,-d:]
        
        S=np.empty((n, b, d))
        Y = np.empty((n, b, self.NC))
        prevc = c0 if not c0 is None else np.zeros((b, d))
        C = np.empty((n, b, d))
        for t in range(n):
            S[t] = prevc
            U[t,:,-d:] = prevc
            print("U[t]:", U[t].shape, "   Wrf:", Wrf.shape)
            RF[t] = U[t].dot(Wrf)
            RFf[t] = 1.0/(1.0+np.exp(-RF[t]))        # f: sigmoid
            
            print("RF[t]:", RF[t].shape)
            print("S[t]:", S[t].shape)
            
            V[t,:,-d:] = RFf[t,:,:d]*S[t]
            N[t] = V[t].dot(Wn)
            Nf[t] = np.tanh(N[t])
            C[t] = (1-RFf[t,:,d:])*S[t] + RFf[t,:,d:]*Nf[t]
            
            prevc = C[t]
            
        self.Y = C.transpose((1,0,2)).copy()
        context = {
            "C":    C,
            "S":    S,
            "c0":   c0,
            "U":    U,
            "V":    V,
            "RF":   RF,
            "RFf":  RFf,
            "N":    N,
            "Nf":   Nf
        }
        return self.Y, prevc, context
        
    def grads(self, dY, dcn, xs, y, context):

    #def backward(self, dY, dcn = None, cache=None): 

        """
        dY_in should be of shape (n,b,output_size), where n = length of sequence, b = batch size
        """
    
        # transpose dY (=dL/dy) so that it is [t, mb, y]
        
        dY = dY.transpose((1,0,2))
    
        if self.Reversed:
            dY = dY[::-1,:,:]

        C = context['C']
        S = context['S']
        c0 = context['c0']
        U = context['U']
        V = context['V']
        RF = context['RF']
        RFf = context['RFf']
        N = context['N']
        Nf = context['Nf']
        
        n, b, _ = dY.shape
        d = self.NC # hidden size
        input_size = U.shape[2]-1-d        # - C - bias

        # views
        X = U[:,:,1:input_size+1]
        Ff = RFf[:,:,d:]
        F = RF[:,:,d:]
        
        Wf = self.W[1:,:d]
        Wn = self.W[1:,d:]

        Wx = self.W[1:1+input_size,:]
        Ws = self.W[1+input_size:,:]
        
        dW = np.zeros_like(self.W)
        
        dX = np.empty_like(X)
        dS = np.empty_like(S)
        
        NmS = Nf - S
        dFprime = Ff*(1.0-Ff)
        dNprime = 1.0-Nf**2
        
        #gF = SmN*dFprime
        #gN = (1.0-Ff)*dNprime

        gFN = np.empty((n, b, d*2))
        gFN[:,:,:d] = NmS*dFprime
        gFN[:,:,d:] = Ff*dNprime
        gF = gFN[:,:,:d]
        gN = gFN[:,:,d:]
        
        gCgFNt = np.empty((b, d*2))
        
        
        #print "S:", S
        #print "N:", N

        dC = dcn if not dcn is None else np.zeros((b, d))
        
        #if dcn is not None: dC[n-1] += dcn.copy() # carry over gradients from later
        for t in reversed(range(n)):
            gC = dC + dY[t]
            
            gCgFNt[:,:d] = gC*gF[t]
            gCgFNt[:,d:] = gC*gN[t]
                        
            dX[t] = np.dot(gCgFNt, Wx.T)
            dS[t] = np.dot(gCgFNt, Ws.T) + gC*(1-Ff[t])

            dW += np.dot(U[t].T, gCgFNt)
                        
            dC = dS[t]
        
        # transpse dX backwards
        
        dX = dX.transpose((1,0,2))
        
        return [dX], [dW], dC
    
