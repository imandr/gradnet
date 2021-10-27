import numpy as np
from .layer import Layer

def softmax(x):
    expx = np.exp(x-np.max(x, axis=-1))
    return expx/np.sum(expx, axis=-1, keepdims=True)    

def attention(key, data, sharpness=1.0):
    # ++++
    # returns [mb, capacity]
    return softmax(np.einsum("mcl,ml->mc", data, key))

def softmax_jacobian(n, a):
    # returns [m,c,c]
    eye = np.eye(n)[None,:,:]
    return a[:,None,:]*(eye-a[:,:,None])    

def updated(key_w, data0, alpha, p, sharpness=1.0):
    a = attention(key_w, b, sharpness)
    w = a[:,:,None] * p[:,None,:]
    return data0 + alpha*(w-data0)

def step(key_r, data0, key_w, alpha, p, sharpness=1.0):
    # +++
    a_w = attention(key_w, data0, sharpness)
    w = a_w[:,:,None] * p[:,None,:]
    data1 = data0 + alpha*(w-data0)
    a_r = attention(key_r, data1, sharpness)
    g = np.sum(a_r[:,:,None]*data1[:,:,:], axis=1)
    return g, data1

def da_dk_jac(key, data, sharpness=1.0):
    # +++
    # returns [m,c,l] = dA[m,c]/dK[m,l]
    a = attention(key, data, sharpness)   # [m,c]
    jac = softmax_jacobian(data.shape[1], a)
    return sharpness*np.einsum("mcx,mxl->mcl", jac, data)

def da_dd_jac(key, data, sharpness=1.0):
    # +++
    # returns [m,c,c,l]
    a = attention(key, data, sharpness)   # [m,c]
    jac = softmax_jacobian(data.shape[1], a)
    return sharpness*jac[:,:,:,None]*key[:,None,None,:]
    
def dd_dkw_jacobian(key_w, data0, alpha, p, sharpness=1.0):
    # +++
    # key[m,l]
    # data[m,c,l]
    # p[m,l]
    # returns: dW/dKw[m,c,l,l]
    a_jac = da_dk_jac(key_w, data0, sharpness)
    return alpha * np.einsum("ma,mcb->mcab", p, a_jac)

def dd_dp_jacobian(key_w, data0, alpha, sharpness=1.0):
    # +++
    # key_w[m,l]
    # data[m,c,l]
    # returns [m,c,l,l]
    a = attention(key_w, data0, sharpness) # [m,c]
    eye = np.eye(data.shape[-1])
    return alpha * a[:,:,None,None] * eye[None,None,:,:]

def dg_dd_jacobian(key_r, data1, sharpness=1.0):
    # returns [m,c,l,l]
    a = attention(key_r, data1, sharpness)   # [m,c]
    eye = np.eye(data1.shape[-1])
    return a[:,None,:,None]*eye[None,:,None,:] + np.einsum("mak,maij->mkij", data1, da_dd_jac(key_r, data1, sharpness))

def dg_dp_jacobian(key_r, data0, data1, key_w, alpha, p, sharpness=1.0):
    # returns [m,l,l]
    return np.einsum("mcab,mcab->mab", 
            dg_dd_jacobian(key_r, data1, sharpness),
            dd_dp_jacobian(key_w, data0, alpha, sharpness)
    )

def dg_dkw_jacobian(key_r, data0, data1, key_w, alpha, p, sharpness=1.0):
    # returns [m,l,l]
    return np.einsum("mcab,mcab->mab", 
            dg_dd_jacobian(key_r, data1, sharpness),
            dd_dkw_jacobian(key_w, data0, alpha, sharpness)
    )

def dg_dkr_jacobian(key_r, data1, sharpness=1.0):
    # +++
    # dG[m,l]/dKr[m,l]   -> [m,l,l]
    return np.einsum("mca,mcb->mab", data1, da_dk_jac(key_r, data1, sharpness))



class Memory(RNNLayer)

    #
    # Input: [Kr, kW, P]
    # Input state: [Kw, P] from previous step
    # Output state: [Kw, P]
    # Output: [G]

    RSharpness = 1.0
    WSharpness = 1.0
    Alpha = 0.5
    
    def __init__(self, capacity, data_length):
        self.MinibatchSize = None
        self.Data = None
        self.L = data_length
        self.C = capacity
        
    def configure(self, inputs):
        assert len(inputs) == 1 and inputs[0].Shape[-1] == self.L*3+1
        return (self.L,)
        
    check_confgiuration = confgure
    
    def init_state(self, mb):
        self.MinibatchSize = mb
        self.Data = np.random.zeros((mb, self.C, self.L))
        return np.zeros((mb, 2, L))     # Kw[0] and P[1]
        
    @property
    def params(self):
        return None
        
    def init_context(self, x, state_in):
        T, b, d = x.shape       # T, minibatch, width (=L*3)
        assert b == self.MinibatchSize
        assert x.shape(-1) == self.L*3+1
        data = np.empty((T, b, self.L))
        return data
        
    def forward(self, t, x, s, context):
        # return y, c
        raise NotImplementedError()
        return y, c, context
        
    def backward(self, t, gy_t, gstate_t, gw, context):
        # given dL/dc = gc and dL/dy = gy and accumulated dL/dw = gw return dL/dx, dL/ds and updated dL/dw
        # initial gw is None
        raise NotImplementedError()
        return gx, gw, gs       # gx is ndarray, not list !
        

        
        
        
    
        
        
        
        
        
