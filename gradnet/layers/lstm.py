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
        print("lstm.configure: inp.Shape:", inp.Shape)
        assert len(shape) == 2, f"Expected shape with 3 dimensions, got {shape}"
        self.Nin = shape[-1]

        print("lstm.configure: Nin:", self.Nin, "  NC:", self.NC)

        self.WLSTM = rng.normal(size=(self.Nin + self.NC + 1, 4 * self.NC),
                scale= 1.0 / np.sqrt(self.Nin + self.NC))
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
        

    def check_config(self, inputs):
        assert len(inputs) == 1
        inp = inputs[0]
        shape = inp.Shape
        assert len(shape) == 2
        assert shape[-1] == self.Nin
        return (shape[0] if self.ReturnSequences else 1, self.Nout)
        
    def compute(self, xs, state_in):
        assert len(xs) == 1
        x = xs[0]
        
        # assume x in [batch, time, width]
        # transpose so that x is [time, batch, width]
        
        #print(self, "x:", x.shape)
        
        X = np.transpose(x, (1, 0, 2))
        
        if self.Reversed:
            X = X[::-1,:,:]

        n,b,input_size = X.shape
        d = self.NC # hidden size
        if state_in is None:
            state_in = np.zeros((2, b, d))
            
        c0 = state_in[0, :, :]
        h0 = state_in[1, :, :]
        
        # Perform the LSTM forward pass with X as the input
        xphpb = self.WLSTM.shape[0] # x plus h plus bias, lol
        #print "Hout size:", Hout.shape
        IFOG = np.zeros((n, b, d * 4)) # input, forget, output, gate (IFOG)
        IFOGf = np.zeros((n, b, d * 4)) # after nonlinearity
        
        CHout = np.zeros((2, n, b, d))
        C =     CHout[0, :, :, :]           # view
        Hout =  CHout[1, :, :, :]           # view
        
        #Hout = np.zeros((n, b, d)) # hidden representation of the LSTM (gated cell content)
        #C = np.zeros((n, b, d)) # cell content

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
            # non-linearities
            IFOGf[t,:,:3*d] = 1.0/(1.0+np.exp(-IFOG[t,:,:3*d])) # sigmoids; these are the gates
            IFOGf[t,:,3*d:] = np.tanh(IFOG[t,:,3*d:]) # tanh
            # compute the cell activation
            prevc = C[t-1] if t > 0 else c0
            C[t] = IFOGf[t,:,:d] * IFOGf[t,:,3*d:] + IFOGf[t,:,d:2*d] * prevc
            Ct[t] = np.tanh(C[t])
            Hout[t] = IFOGf[t,:,2*d:3*d] * Ct[t]
            prevh = Hout[t]
        
        context = {
            'CHout':CHout,
            'Ct': Ct,
            'IFOGf': IFOGf,
            'IFOG': IFOG,
            'Hin': Hin,
            'ch0': state_in
        }
        y = Hout.transpose((1,0,2))
        if not self.ReturnSequences:    y = y[:,-1,:]
        #print(self, "y:", y.shape)
        return y, CHout[:,n-1,:,:], context
        
    def grads(self, y_grads, s_out_grads, xs, y, context):
        dchn = s_out_grads
        #print(self, "y_grads:", y_grads.shape)
        if self.ReturnSequences:
            assert len(y_grads.shape) == 3          # [mb, t, w]
            y_grads = y_grads.transpose((1,0,2))    # -> [t, mb, w]
            if self.Reversed:
                y_grads = y_grads[::-1,...]
            dHout = y_grads[-1]
        else:
            assert len(y_grads.shape) == 2          # [mb, w]
            dHout = y_grads

        IFOGf = context['IFOGf']
        IFOG = context['IFOG']
        CHout = context['CHout']
        Ct = context['Ct']
        Hin = context['Hin']
        ch0 = context['ch0']

        C = CHout[0]
        Hout = CHout[1]
        c0 = ch0[0]
        h0 = ch0[1]

        n,b,d = Hout.shape

        input_size = self.WLSTM.shape[0] - d - 1 # -1 due to bias

        # backprop the LSTM
        dIFOG = np.empty(IFOG.shape[1:])        # remove t dimension
        dIFOGf = np.empty(IFOGf.shape[1:])      # remove t dimension
        
        #print(self, "IFOGf:", IFOGf.shape, "  dIFOGf:", dIFOGf.shape)
        
        dWLSTM = np.zeros(self.WLSTM.shape)
        #dHin = np.zeros(Hin.shape[1:])          # remove t dimension
        dC = np.zeros(C.shape)
        dX = np.zeros((n,b,input_size))
        
        dch0 = np.zeros((2, b, d))
        dc0 = dch0[0]            # view
        dh0 = dch0[1]            # view
        
        if dchn is not None: 
            dC[n-1,:,:] = dchn[0]           # carry over gradients from later
            dHout += dchn[1]

        for t in reversed(range(n)):
            tanhCt = Ct[t]
            dIFOGf[:,2*d:3*d] = tanhCt * dHout     # [nb, Nc] * [nb, Nc] -> [nb, Nc]
            # backprop tanh non-linearity first then continue backprop
            dC[t] += (1-tanhCt**2) * (IFOGf[t,:,2*d:3*d] * dHout)
        
            if t > 0:
                dIFOGf[:,d:2*d] = C[t-1] * dC[t]
                dC[t-1] += IFOGf[t,:,d:2*d] * dC[t]
            else:
                dIFOGf[:,d:2*d] = c0 * dC[t]
                dc0[...] = IFOGf[t,:,d:2*d] * dC[t]
            dIFOGf[:,:d] = IFOGf[t,:,3*d:] * dC[t]
            dIFOGf[:,3*d:] = IFOGf[t,:,:d] * dC[t]
        
            # backprop activation functions
            dIFOG[:,3*d:] = (1 - IFOGf[t,:,3*d:] ** 2) * dIFOGf[:,3*d:]
            y = IFOGf[t,:,:3*d]
            #print(self," dIFOG:", dIFOG.shape, "  y:", y.shape)
            dIFOG[:,:3*d] = (y*(1.0-y)) * dIFOGf[:,:3*d]
        
            # backprop matrix multiply
            dWLSTM += np.dot(Hin[t].transpose(), dIFOG)
            dHin = dIFOG.dot(self.WLSTM.T)
        
            # backprop the identity transforms into Hin
            dX[t] = dHin[:,1:input_size+1]
            if t > 0:
                dHout = dHin[:,input_size+1:]
                if self.ReturnSequences:
                    dHout = dHout + y_grads[t-1]
            else:
                dh0[...] = dHin[:,input_size+1:]
                
        dX = dX.transpose((1,0,2))
        return [dX], [dWLSTM], dch0
        