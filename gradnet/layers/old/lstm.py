import numpy as np
import math, time
from .. import Layer
from ..util import make_list
from numpy.random import default_rng

rng = default_rng()

class LSTM(Layer):
    
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

        #print("lstm.configure: Nin:", self.Nin, "  NC:", self.NC)

        self.WLSTM = rng.normal(size=(1 + self.Nin + self.NC, 4 * self.NC),
                scale= 1.0 / np.sqrt(self.Nin + self.NC))
                
        #if self.Nin == self.NC:
        #    eye = np.eye(self.NC)
        #    #print("eye=", eye)
        #    for i in range(1,1 + self.Nin + self.NC,self.NC):
        #        for j in range(0, 4 * self.NC, self.NC):
        #            self.WLSTM[i:i+self.Nin, j:j+self.NC] = eye
                
        #print "WLSTM shape=", self.WLSTM.shape
        self.WLSTM[0,:] = 0 # initialize biases to zero
        # forget gates get little bit negative bias initially to encourage them to be turned off
        # remember that due to Xavier initialization above, the raw output activations from gates before
        # nonlinearity are zero mean and on order of standard deviation ~1
        self.WLSTM[0,self.NC:2*self.NC] = 3.0

        return (shape[0], self.Nout) if self.ReturnSequences else (self.Nout,)

    @property
    def weights(self):
        return [self.WLSTM]
        
    def set_weights(self, weights):
        self.WLSTM = weights[0]
        
    def check_config(self, inputs):
        assert len(inputs) == 1
        inp = inputs[0]
        shape = inp.Shape
        assert len(shape) == 2
        assert shape[-1] == self.Nin
        return self.Shape
        
    def compute(self, xs, state_in):
        c0, h0 = state_in if not state_in is None else (None, None)
        
        """
        X should be of shape (n,b,input_size), where n = length of sequence, b = batch size
        """
        X = xs[0].transpose((1,0,2))
        if self.Reversed:
            X = X[::-1,:,:]

        n,b,input_size = X.shape
        d = self.NC # hidden size
        if c0 is None: c0 = np.zeros((b,d))
        if h0 is None: h0 = np.zeros((b,d))
        
        # Perform the LSTM forward pass with X as the input
        xphpb = self.WLSTM.shape[0] # x plus h plus bias, lol
        Hout = np.zeros((n, b, d)) # hidden representation of the LSTM (gated cell content)
        IFOG = np.zeros((n, b, d * 4)) # input, forget, output, gate (IFOG)
        IFOGf = np.zeros((n, b, d * 4)) # after nonlinearity
        C = np.zeros((n, b, d)) # cell content
        Ct = np.zeros((n, b, d)) # tanh of cell content
        Hin = np.empty((n, b, xphpb)) # input [1, xt, ht-1] to each tick of the LSTM
        Hin[:,:,1:input_size+1] = X
        Hin[:,:,0] = 1 # bias
        prevh = h0
        for t in range(n):
            # concat [x,h] as input to the LSTM
            #print "Hin shape:", Hin.shape, "   X shape=",X[t].shape, "   input_size=", input_size
            #Hin[t,:,1:input_size+1] = X[t]
            Hin[t,:,input_size+1:] = prevh
            # compute all gate activations. dots: (most work is this line)
            IFOG[t] = Hin[t].dot(self.WLSTM)
            #print("IFOG:", IFOG[t])
            # non-linearities
            IFOGf[t,:,:3*d] = 1.0/(1.0+np.exp(-IFOG[t,:,:3*d])) # sigmoids; these are the gates
            IFOGf[t,:,3*d:] = np.tanh(IFOG[t,:,3*d:]) # tanh
            
            #print("IFOGf:", IFOGf[t])
            # compute the cell activation
            prevc = C[t-1] if t > 0 else c0
            #print("prevc:", prevc)
            C[t] = IFOGf[t,:,:d] * IFOGf[t,:,3*d:] + IFOGf[t,:,d:2*d] * prevc
            #print("C:",C[t])
            Ct[t] = np.tanh(C[t])
            #print("Ct:", Ct[t])
            Hout[t] = IFOGf[t,:,2*d:3*d] * Ct[t]
            #print("Hout:", Hout[t])
            prevh = Hout[t]
        
        cache = {}
        cache['Hout'] = Hout
        cache['IFOGf'] = IFOGf
        cache['IFOG'] = IFOG
        cache['C'] = C
        cache['Ct'] = Ct
        cache['Hin'] = Hin
        cache['c0'] = c0
        cache['h0'] = h0
        
        # return C[t], as well so we can continue LSTM with prev state init if needed
        y = Hout
        cache['y'] = y
        
        return y.transpose((1,0,2)), np.array([C[t], Hout[t]]), cache
        
    def grads(self, y_grads, s_out_grads, xs, y, context):

        """
        dY_in should be of shape (n,b,output_size), where n = length of sequence, b = batch size
        """
        #print("--- grads ---")
        dY_in = y_grads.transpose((1,0,2))
        #print("dY_in:", dY_in.shape, dY_in)
        if self.Reversed:
            dY_in = dY_in[::-1,:,:]
        n,b,d = dY_in.shape
        #print("s_out_grads:", s_out_grads.shape)
        dcn, dhn = s_out_grads if s_out_grads is not None else (np.zeros((b, self.NC)), np.zeros((b, self.NC)))
        #print("dcn:", dcn.shape)

        #print("dcn, dhn:", dcn, dhn)

        cache = context
        y = cache['y']
        Hout = cache['Hout']
        IFOGf = cache['IFOGf']
        IFOG = cache['IFOG']
        C = cache['C']
        Ct = cache['Ct']
        Hin = cache['Hin']
        c0 = cache['c0']
        h0 = cache['h0']
        
        #print("IFOG:", IFOG)

        dHout = dY_in.copy()

        input_size = self.WLSTM.shape[0] - d - 1 # -1 due to bias

        # backprop the LSTM
        dIFOG = np.zeros(IFOG.shape)
        dIFOGf = np.zeros(IFOGf.shape)
        dWLSTM = np.zeros(self.WLSTM.shape)
        dHin = np.zeros(Hin.shape)
        dC = np.zeros(C.shape)
        dX = np.zeros((n,b,input_size))
        dh0 = np.zeros((b, d))
        dc0 = np.zeros((b, d))
        # IM: already a copy dHout = dHout_in.copy() # make a copy so we don't have any funny side effects
        dC[n-1] += dcn.copy() # carry over gradients from later
        dHout[n-1] += dhn.copy()
        
        #print("n-1=", n-1, "  dC[n-1]=", dC[n-1], "   dHout[n-1]=", dHout[n-1])
        
        for t in reversed(range(n)):
            #print("t=", t)
            tanhCt = Ct[t]
            dIFOGf[t,:,2*d:3*d] = tanhCt * dHout[t]     # dGf     
            # backprop tanh non-linearity first then continue backprop
            dC[t] += (1-tanhCt**2) * (IFOGf[t,:,2*d:3*d] * dHout[t])
            #print("dC[t]:", dC[t])
        
            if t > 0:
                dIFOGf[t,:,d:2*d] = C[t-1] * dC[t]
                dC[t-1] += IFOGf[t,:,d:2*d] * dC[t]
            else:
                dIFOGf[t,:,d:2*d] = c0 * dC[t]
                dc0 = IFOGf[t,:,d:2*d] * dC[t]
            dIFOGf[t,:,:d] = IFOGf[t,:,3*d:] * dC[t]
            dIFOGf[t,:,3*d:] = IFOGf[t,:,:d] * dC[t]

            #print("dIFOGf:", dIFOGf)
        
            # backprop activation functions
            dIFOG[t,:,3*d:] = (1 - IFOGf[t,:,3*d:] ** 2) * dIFOGf[t,:,3*d:]
            y = IFOGf[t,:,:3*d]
            dIFOG[t,:,:3*d] = (y*(1.0-y)) * dIFOGf[t,:,:3*d]
            
            #print("dIFOG:", dIFOG)
        
            # backprop matrix multiply
            dWLSTM += np.dot(Hin[t].transpose(), dIFOG[t])
            dHin[t] = dIFOG[t].dot(self.WLSTM.T)
        
            # backprop the identity transforms into Hin
            dX[t] = dHin[t,:,1:input_size+1]
            if t > 0:
                dHout[t-1,:] += dHin[t,:,input_size+1:]
            else:
                dh0 += dHin[t,:,input_size+1:]

        return [dX.transpose((1,0,2))], [dWLSTM], np.array([dc0, dh0])
