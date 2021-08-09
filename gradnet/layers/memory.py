import numpy as np
from .layer import Layer

def softmax(x):
    expx = np.exp(x-np.max(x, axis=-1))
    return expx/np.sum(expx, axis=-1, keepdims=True)    

class Memory(Layer)

    RSharpness = 1.0
    WSharpness = 1.0
    Alpha = 0.5
    
    def __init__(self, capacity, data_length, minibatch_size):
        self.MinibatchSize = minibatch_size
        self.Data = np.random.random((minibatch_size, capacity, data_length))*2-1
        self.L = data_length
        self.C = capacity
        
    def configure(self, inputs):
        assert len(inputs) == 1 and inputs[0].Shape[-1] == self.L*3
        return (self.L,)
        
    check_confgiuration = confgure
        
    def attention(self, key, data, sharpness):
        # returns [mb, capacity]
        return softmax(np.einsum("mcl,ml->mc", data, key))
        
    def attention_grad(self, key, data, sharpness):
        # returns [m,c,l] = dA[m,c]/dK[m,l]
        a = np.einsum("ml,mcl->mc", key, data)      # [m, c]
        eye = np.eye(self.C)[None,:,:]
        jac = a[:,None,:]*(eye-a[:,:,None])     # softmax jacobian J[m,i,j]= dS(m,i)/da(m,j)
        return sharpness*np.einsum("mcx,mxl->mcl", jac, data)
        
    def read(self, key):
        # key is array [mbatch, data_length]
        attention = self.attention(key, self.Data, self.RSharpness)
        return np.einsum("mcl,mc->ml", self.Data, attention)
        
    def write(self, key, data):
        # key: [mb, data_length]
        # data: [mb, data_length]
        # Data: [mb, capacity, data_length]
        attention = self.attention(key, self.Data, self.WSharpness)      # [mb, capacity]
        self.Data += self.Alpha * attention[:,:,None]*(data[:,None,:] - self.Data)
        
    def grad_kr(self, key, data, sharpness):
        # -> [m,l,l], dG[m,l]/dKr[m,l]
        grad_attention = self.attention_grad(key, data, sharpness)
        return np.einsum("mca,mcb->mab", data, grad_attention)
        
    def grad_kw(self, key, p, data):
        attention = self.attention(key, data, self.WSharpness)
        grad_attention = self.attention_grad(key, data, self.WSharpness)
        return np.einsum("mca,mcb->mab", self.Alpha*attention[:,:,None]*p[:,None,:]+data, grad_attention)
    
    
        
        
        
        
        
        
