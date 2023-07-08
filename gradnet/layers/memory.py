import numpy as np
from .rnn import RNNLayer

D = 2.0

def softmax(x):
    expx = np.exp(x-np.max(x, axis=-1, keepdims=True))
    return expx/np.sum(expx, axis=-1, keepdims=True)    

def attention(key, data, sharpness):
    # ++++
    # returns [mb, capacity]
    return softmax(np.einsum("mcl,ml->mc", data, key)*sharpness[:, None]*D)

def step(m, key_w, s_w, p, key_r, s_r):
    a_w = attention(key_w, m, s_w)
    n = m*(1-a_w[:,:,None]) + p[:,None,:]*a_w[:,:,None]
    a_r = attention(key_r, n, s_r)
    q = np.einsum("mc,mcl->ml", a_r, n)
    return q, a_r, n, a_w 

def softmax_jacobian(n, a):
    # returns [m,c,c]
    eye = np.eye(n)[None,:,:]
    return a[:,None,:]*(eye-a[:,:,None])    

def da_dk(key, data, a, sharpness):
    # +++
    # returns [m,c,l] = dA[m,c]/dK[m,l]
    jac = softmax_jacobian(data.shape[1], a)
    return D*sharpness[:,None,None]*np.einsum("mcx,mxl->mcl", jac, data)

def da_ds(data, key, a):
    # +++
    jac = softmax_jacobian(data.shape[1], a)
    return D*np.einsum("mib,mba,ma->mi", jac, data, key)
    
def da_dd(key, data, a, sharpness):
    jac = softmax_jacobian(data.shape[1], a)
    #print("sharpness:", sharpness.shape)
    return D*sharpness[:,None,None,None]*np.einsum("mij,mk->mijk", jac, key)

def dq_dar(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w):
    # +++
    return n.transpose((0,2,1))

def dq_dn(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w):
    # +++
    eye = np.eye(m.shape[-1])
    dadn = da_dd(key_r, n, a_r, s_r)
    #print("dadn:", dadn.shape)
    return a_r[:,None,:,None]*eye[None,:,None,:] + np.einsum("maj,maik->mjik", n, dadn)

def dn_dp(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w):
    # +++
    eye = np.eye(m.shape[-1])
    return a_w[:,:,None,None]*eye[None,None,:,:]
    
def dn_daw(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w):
    # +++
    eye = np.eye(m.shape[1])
    return (p[:,None,:,None] - m[:,:,:,None])*eye[None,:,None,:]

def dn_dm(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w):
    # +++
    eye_ki = np.eye(m.shape[1])[None,:,None,:,None]
    eye_jn = np.eye(m.shape[-1])[None,None,:,None,:]
    dadm = da_dd(key_w, m, a_w, s_w)
    return (1-a_w[:,None,None,:,None])*eye_ki*eye_jn + \
        np.einsum("mikn,mij->mijkn", dadm, p[:,None,:] - m[:,:,:])

#
# derived
#
        
def dq_dp(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w):
    # +++
    # [m,l,l]
    return np.einsum("miab,mabj->mij", 
        dq_dn(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w),
        dn_dp(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w)
    )

def dq_dkr(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w):
    # +++
    # [m,l,l]
    return np.einsum("mia,maj->mij", 
        dq_dar(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w),
        da_dk(key_r, n, a_r, s_r)
    )

def dq_dm(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w):
    # +++
    # dq/dn*dn/dm: [m,l,c,l]*[m,c,l,c,l] -> [m,l,c,l]
    return np.einsum("miab,mabjk->mijk", 
        dq_dn(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w),
        dn_dm(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w)
    )

def dq_dkw(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w):
    # +++
    # dq/dn*dn/dAw*dAw/dKw: [m,l,c,l]*[m,c,l,c]*[m,c,l] -> [m,l,l]
    return np.einsum("miab,mabc,mcj->mij", 
        dq_dn(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w),
        dn_daw(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w),
        da_dk(key_w, m, a_w, s_w)
    )

def dq_dsw(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w):
    # +++
    # dq/dn*dn/dAw*dAw/dSw: [m,l,c,l]*[m,c,l,c]*[m,c] -> [m,l]
    return np.einsum("miab,mabc,mc->mi", 
        dq_dn(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w),
        dn_daw(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w),
        da_ds(m, key_w, a_w)
    )

def dq_dsr(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w):
    # +++
    # dq/dAr*dAr/dSr: [m,l,c]*[m,c] -> [m,l]
    return np.einsum("mia,ma->mi",
        dq_dar(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w),
        da_ds(n, key_r, a_r)
    )

def dn_dsw(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w):
    # +++
    # dN/dAw*dAw/dSw: [m,c,l,c]*[m,c] -> [m,c,l]
    return np.einsum("mija,ma->mij",
        dn_daw(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w),
        da_ds(m, key_w, a_w)
    )

def dn_dkw(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w):
    # +++
    # dN/dAw*dAw/dKw: [m,c,l,c]*[m,c,l] -> [m,c,l,l]
    return np.einsum("mija,mak->mijk",
        dn_daw(m, key_w, s_w, p, key_r, s_r, a_r, n, a_w),
        da_dk(key_w, m, a_w, s_w)
    )
   
    


class Memory(RNNLayer):

    #
    # X: <Kw, P, Kr, Sw, Sr>
    # Input state: N[t-1]=M[t] from previous step
    # Output state: N[t]=M[t+1]
    # Output: [Q]

    RSharpness = 1.0
    WSharpness = 1.0
    Alpha = 0.5
    
    def __init__(self, capacity, data_length, return_sequences=False, **args):
        RNNLayer.__init__(self, return_sequences=return_sequences, **args)
        
        self.L = data_length
        self.C = capacity
        self.D = 5.0
        self.ReturnSequences = return_sequences
        
    def configure(self, inputs):
        assert len(inputs) == 1 and inputs[0].Shape[-1] == self.L*3+2
        return (inputs[0].Shape[0], self.L) if self.ReturnSequences else (self.L,)
        
    check_confgiuration = configure
    
    def init_state(self, mb):
        data = np.zeros((mb, self.C, self.L))
        return data
        
    def init_context(self, x, state_in):
        T, b, d = x.shape       # T, minibatch, width (=L*3)
        assert x.shape[-1] == self.L*3+2
        data_in = state_in
        data_record = np.empty((T+1,)+data_in.shape)
        data_record[-1,...] = data_in
        Aw = np.empty((T, b, self.C))
        Ar = np.empty((T, b, self.C))
        context = (data_record, Aw, Ar, x.copy())
        return context
        
    def forward(self, t, x, s, context):
        # X: <Kw, P, Kr, Sw, Sr>
        #print("Memory.forward: x:", x.shape, x)
        L, C = self.L, self.C
        N_record, Aw_record, Ar_record, x_record = context
        Kw = x[:,:L]
        P  = x[:,L:2*L]
        Kr = x[:,2*L:3*L]
        Sw = x[:,3*L]
        Sr = x[:,3*L+1]
        M = s
        Q, Ar, N, Aw = step(M, Kw, Sw, P, Kr, Sr)
        
        N_record[t] = N
        Aw_record[t] = Aw
        Ar_record[t] = Ar
        
        return Q, N, context
        
    def backward(self, t, gy_t, gstate_t, gw, context):
        # given dL/dc = gc and dL/dy = gy and accumulated dL/dw = gw return dL/dx, dL/ds and updated dL/dw
        # initial gw is None

        L, C = self.L, self.C
        N_record, Aw_record, Ar_record, x = context
        Nt = N_record[t]
        Mt = N_record[t-1]
        Awt = Aw_record[t]
        Art = Ar_record[t]
        Kwt = x[t,:,:self.L]
        Pt = x[t,:,self.L:self.L*2]
        Krt = x[t,:,self.L*2:self.L*3]
        Swt = x[t,:,3*L]      
        Srt = x[t,:,3*L+1]
        
        jac_dq_dSw = dq_dsw(Mt, Kwt, Swt, Pt, Krt, Srt, Art, Nt, Awt)
        jac_dq_dKw = dq_dkw(Mt, Kwt, Swt, Pt, Krt, Srt, Art, Nt, Awt)
        jac_dq_dKr = dq_dkr(Mt, Kwt, Swt, Pt, Krt, Srt, Art, Nt, Awt)
        jac_dq_dP = dq_dp(Mt, Kwt, Swt, Pt, Krt, Srt, Art, Nt, Awt)
        jac_dq_dSr = dq_dsr(Mt, Kwt, Swt, Pt, Krt, Srt, Art, Nt, Awt)
        jac_dq_dSw = dq_dsw(Mt, Kwt, Swt, Pt, Krt, Srt, Art, Nt, Awt)
        
        jac_dn_dP = dn_dp(Mt, Kwt, Swt, Pt, Krt, Srt, Art, Nt, Awt)
        jac_dn_dSw = dn_dsw(Mt, Kwt, Swt, Pt, Krt, Srt, Art, Nt, Awt)
        jac_dn_dKw = dn_dkw(Mt, Kwt, Swt, Pt, Krt, Srt, Art, Nt, Awt)
        
        jac_dN_dM = dn_dm(Mt, Kwt, Swt, Pt, Krt, Srt, Art, Nt, Awt)
        jac_dQ_dM = dq_dm(Mt, Kwt, Swt, Pt, Krt, Srt, Art, Nt, Awt)
        
        gx = np.concatenate(
            [
                np.einsum("mij,mi->mj", jac_dq_dKw, gy_t) + np.einsum("mabj,mab->mj", jac_dn_dKw, gstate_t),
                np.einsum("mij,mi->mj", jac_dq_dP, gy_t) + np.einsum("mabj,mab->mj", jac_dn_dP, gstate_t),
                np.einsum("mij,mi->mj", jac_dq_dKr, gy_t),
                (np.einsum("mi,mi->m", jac_dq_dSw, gy_t) + np.einsum("mcl,mcl->m", jac_dn_dSw, gstate_t))[:,None],
                np.einsum("mi,mi->m", jac_dq_dSr, gy_t)[:,None]
            ],
            axis=-1
        )
        
        gs = np.einsum("mijkl,mij->mkl", jac_dN_dM, gstate_t) + \
            np.einsum("macl,ma->mcl", jac_dQ_dM, gy_t)
        return gx, gw, gs       # gx is ndarray, not list !
        

        
        
        
    
        
        
        
        
        
