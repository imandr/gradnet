import numpy as np, math

class LayerOptimizer(object):
    
    def __init__(self, params, optimizer):
        self.ParamOptimizer = optimizer
        self.Contexts = [None for _ in params]
        
    def __str__(self):
        return f"[LayerOptimizer {self.ParamOptimizer.__class__.__name__}]"
    
    def apply_deltas(self, grads, params):
        #print(self,".deltas: grads:", grads)
        new_contexts = []
        deltas = []
        for c, g, p in zip(self.Contexts, grads, params):
            d, c = self.ParamOptimizer.deltas(c, g, p)
            new_contexts.append(c)
            p[...] += d
        self.Contexts = new_contexts
        return deltas
        
class SingleParamOptimizer(object):
    
    def __str__(self):
        return f"[ParamOptimizer {self.__class__.__name__}]"

class SGD(SingleParamOptimizer):

    # deals with single layer param
        
    def __init__(self, learning_rate=0.001, momentum=0.0):
        self.Eta = learning_rate
        self.Momentum = momentum
        
    def deltas(self, last_deltas, grads, params):
        #print("SGD.deltas: grads:", grads)
        deltas = -self.Eta*grads
        #print("SGD: grads:", grads)
        #print("SGD: deltas:", deltas)
        if self.Momentum:
            if last_deltas is not None:
                deltas = self.Momentum*last_deltas + (1.0-self.Momentum)*deltas
        return deltas, deltas
        
class Adam(SingleParamOptimizer):
    
    Epsilon = 1e-6
    
    def __init__(self, learning_rate=0.001, beta1 = 0.9, beta2 = 0.9):
        self.Eta = learning_rate
        self.B1 = beta1
        self.B2 = beta2
        
    def deltas(self, context, grads, params):
        #print("Adam: grads:", grads)
        g2 = grads*grads
        if context is not None:
            #print("    context:", context)
            m, m2, t = context
            m += (1.0-self.B1)*(grads-m)
            m2 += (1.0-self.B2)*(g2-m2)
        else:
            m = grads
            m2 = g2
            t = 1
        m_ = m/(1-self.B1**t)
        m2_ = m2/(1-self.B2**t)
        deltas = -self.Eta*m_/(np.sqrt(m2_ + self.Epsilon))
        #print("    deltas:", deltas)
        return deltas, (m, m2, t+1)
        
class AcceleratedDescent(SingleParamOptimizer):
    
    Epsilon = 1e-5

    def __init__(self, learning_rate = 0.1, momentum = 1.1, power = 2):
        self.Power = power
        self.Eta = learning_rate
        self.Momentum = momentum
        
    def deltas(self, context, g, params):
        deltas = -self.Eta*g
        if context is None:
            old_g = g
            old_deltas = deltas
        else:
            old_deltas, old_g = context
        f = (np.mean(-old_deltas*g) + self.Epsilon)/   \
            ( (math.sqrt(np.mean(old_deltas*old_deltas))*math.sqrt(np.mean(g*g))) + self.Epsilon )
        w = self.Momentum*((1+f)/2)**self.Power
        deltas = w*old_deltas + (1.0-w)*deltas
        #print("ad: f:", f, "   w:", w, "   deltas:", np.mean(deltas*deltas))
        return deltas, (deltas, g)
        
def get_optimizer(name, *params, **args):
    opt_class = {
        "sgd":  SGD,
        "adam": Adam,
        "ad":   AcceleratedDescent
    }
    return opt_class[name.lower()](*params, **args)

