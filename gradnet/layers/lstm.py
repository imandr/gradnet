import numpy as np
import math, time
from ..util import make_list
from numpy.random import default_rng
from .rnn import RNNLayer

rng = default_rng()

class LSTM(RNNLayer):
    
    def __init__(self, out_size, return_sequences=True, **args):
        
        RNNLayer.__init__(self, return_sequences=return_sequences, **args)
        
        self.Nout = self.NC = out_size
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

        self.WLSTM = rng.normal(size=(1 + self.Nin + self.NC, 4 * self.NC),
                scale= 1.0 / np.sqrt(self.Nin + self.NC))
                
        #if self.Nin == self.NC:
        #    eye = np.eye(self.NC)
        #    #print("eye=", eye)
        #    for i in range(1,1 + self.Nin + self.NC,self.NC):
        #        for j in range(0, 4 * self.NC, self.NC):
        #            self.WLSTM[i:i+self.Nin, j:j+self.NC] = eye
        
        #self.WLSTM[...] = 1.0
        
        
        #print "WLSTM shape=", self.WLSTM.shape
        self.WLSTM[0,:] = 0 # initialize biases to zero
        # forget gates get little bit negative bias initially to encourage them to be turned off
        # remember that due to Xavier initialization above, the raw output activations from gates before
        # nonlinearity are zero mean and on order of standard deviation ~1
        self.WLSTM[0,self.NC:2*self.NC] = 3.0

        return (shape[0], self.Nout) if self.ReturnSequences else (self.Nout,)

    @property
    def params(self):
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
        
    def init_context(self, x, state_in):
        n, b, input_size = x.shape
        d = self.NC
        ch0 = state_in if state_in is not None else np.zeros((2,b,d))
        c0 = ch0[0]
        h0 = ch0[1]
        
        xphpb = self.WLSTM.shape[0] # x plus h plus bias, lol
        Hout = np.zeros((n, b, d)) # hidden representation of the LSTM (gated cell content)
        IFOG = np.zeros((n, b, d * 4)) # input, forget, output, gate (IFOG)
        IFOGf = np.zeros((n, b, d * 4)) # after nonlinearity
        C = np.zeros((n, b, d)) # cell content
        Ct = np.zeros((n, b, d)) # tanh of cell content
        Hin = np.empty((n, b, xphpb)) # input [1, xt, ht-1] to each tick of the LSTM
        Hin[:,:,1:input_size+1] = x
        Hin[:,:,0] = 1 # bias
        
        context = {
            "IFOGf":    IFOGf,
            "IFOG":     IFOG,
            "C":        C,
            "Ct":       Ct,
            "Hin":      Hin,
            "c0":       c0,
        }
        return context
    
    def init_state(self, mb):
        return np.zeros((2, mb, self.NC))
        
    def forward(self, t, xt, s, context):

        IFOGf = context['IFOGf']
        IFOG = context['IFOG']
        C = context['C']
        Ct = context['Ct']
        Hin = context['Hin']

        b = len(xt)

        prevc, prevh = s

        b, input_size = xt.shape
        d = self.NC

        Hin[t,:,input_size+1:] = prevh
        # compute all gate activations. dots: (most work is this line)
        IFOG[t] = Hin[t].dot(self.WLSTM)
        IFOGf[t,:,:3*d] = 1.0/(1.0+np.exp(-IFOG[t,:,:3*d])) # sigmoids; these are the gates
        IFOGf[t,:,3*d:] = np.tanh(IFOG[t,:,3*d:]) # tanh
        #print("prevc:", prevc)
        cout = C[t] = IFOGf[t,:,:d] * IFOGf[t,:,3*d:] + IFOGf[t,:,d:2*d] * prevc
        #print("C:",C[t])
        Ct[t] = np.tanh(C[t])
        #print("Ct:", Ct[t])
        hout = IFOGf[t,:,2*d:3*d] * Ct[t]
        #print("y=", hout)
        return hout, np.array((cout, hout)), context
        
    def backward(self, t, gy_t, gstate_t, gw, context):
        
        #print(f"--- backward[{t}]")

        b,d = gy_t.shape
        dc_out, dh_out = gstate_t

        dWLSTM = gw[0]
        
        #print("context:", context)

        IFOGf = context['IFOGf']
        IFOG = context['IFOG']
        C = context['C']
        Ct = context['Ct']
        Hin = context['Hin']
        c0 = context['c0']

        dIFOGf = np.zeros((b, d*4))
        dIFOG = np.zeros((b, d*4))
        
        dIf = dIFOGf[:,:d]                  # aliases
        dFf = dIFOGf[:,d:2*d]
        dGf = dIFOGf[:,2*d:3*d]         
        dOf = dIFOGf[:,3*d:]

        dHout = dh_out + gy_t
        #print("gy_t=", gy_t, "  dHout=", dHout)
        input_size = self.WLSTM.shape[0] - d - 1 # -1 due to bias

        tanhCt = Ct[t]
        dGf[...] = tanhCt * dHout 
        #dIFOGf[t,:,2*d:3*d] = tanhCt * dHout    # [nb, Nc] * [nb, Nc] -> [nb, Nc]
        dC = dc_out + (1-tanhCt**2) * (IFOGf[t,:,2*d:3*d] * dHout)
        S = c0 if t == 0 else C[t-1]

        dFf[...] = S * dC
        dS = IFOGf[t,:,d:2*d]*dC
    
        dIf[...] = IFOGf[t,:,3*d:] * dC
        dOf[...] = IFOGf[t,:,:d] * dC

        #print("dIFOGf=", dIFOGf)
        dIFOG[:,3*d:] = (1 - IFOGf[t,:,3*d:] ** 2) * dOf
        z = IFOGf[t,:,:3*d] 
        dIFOG[:,:3*d] = (z*(1.0-z)) * dIFOGf[:,:3*d]
        
        #print("dIFOG=", dIFOG)
        #print("W=", self.WLSTM)
        
        wdelta = np.dot(Hin[t].T, dIFOG)
        #print("wdelta:", wdelta.shape,"   dWLSTM:", dWLSTM.shape)
        dWLSTM += wdelta
        dHin = dIFOG.dot(self.WLSTM.T)
    
        # backprop the identity transforms into Hin
        gx = dHin[:,1:input_size+1]
        gHin = dHin[:,input_size+1:] 

        #print("dS:", dS.shape,"   dHin:", dHin.shape)
        return gx, [dWLSTM], np.array([dS, gHin])
