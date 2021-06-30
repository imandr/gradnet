from .util import make_list

class Layer(object):
    
    def __init__(self, name=None, optimizer=None, activation=None):
        self.Name = name
        self.PGradSum = None
        self.XGradSum = None
        self.Configured = False
        self.Optimizer = optimizer
        self.Params = []
        if isinstance(activation, str):
            from .activations import get_activation
            self.Activation = get_activation(activation)
        else:
            self.Activation = activation
        
    def __str__(self):
        return "[Layer %s %s]" % (self.__class__.__name__, self.Name or "")        

    def link(self, inputs):
        from .graphs import Link
    
        #print(self, ".link(): inputs:", inputs)
        inputs = make_list(inputs)
        if not self.Configured:
            shape = self.configure(inputs)
            #print("     self.Shape -> ", self.Shape)
            self.Configured = True
        else:
            shape = self.check_configuration(inputs)
        return Link(self, shape, inputs)
    
    __call__ = link

    def reset_grads(self):
        self.PGradSum = None
        self.XGradSum = None
    
    def compute(self, xs, in_state):
        y, out_state, context = self.call(xs, in_state)
        if self.Activation is not None:
            z, _, a_context = self.Activation.call(y, None)       # assume activation is stateless
            context = (context, y, a_context)
            y = z
        return y, out_state, context
        
    def backprop(self, ygrads, sgrads, xs, y, context):
        #print(self,".backprop: xs:", xs)
        if self.Activation is not None:
            z = y
            y_context, y, a_context = context
            ygrads, _, _ = self.Activation.grads(ygrads, None, [y], z, a_context)       # assumes activation is a stateless and param-less
            x_grads, p_grads, s_in_grads = self.grads(ygrads, sgrads, xs, y, y_context)
        else:
            x_grads, p_grads, s_in_grads = self.grads(ygrads, sgrads, xs, y, context)
        if self.PGradSum is None:
            self.PGradSum = p_grads
            self.XGradSum = x_grads
        else:
            for g, g1 in zip(self.PGradSum, p_grads):
                g[...] += g1
            for g, g1 in zip(self.XGradSum, x_grads):
                g[...] += g1
        return x_grads, s_in_grads
                
    def apply_deltas(self):
        if self.PGradSum is None:
            #print(f"WARNING: apply_grads() is called for layer {self.Name} while its grads were not initialized. Ignoring")
            return
            
        deltas = self.Optimizer.deltas(self.PGradSum)
        if False:
            import math
            import numpy as np
            rms = [math.sqrt(np.mean(d*d)) for d in deltas]
            print(self," deltas rms:", rms)
        for d, p in zip(deltas, self.params):
            p[...] += d
            
        self.PGradSum = None
            
    # overridables

    def configure(self, inputs):
        pass
        # return shape
        
    def check_configuration(self, inputs):
        pass
        # return shape
        
    def call(self, inputs, in_state):
        raise NotImplementedError()
        return y, out_state, context
        
    def grads(self, y_grads, s_out_grads, xs, y, context):
        # y_grads: [mb, ...] - dL/dy[j]
        # returns:
        # x_grads : [mb, ...] - grad per sample
        # p_grad : [...] - average grad over the minibatch
        # s_in_grads : [mb, ...] - grad per sample
        raise NotImplementedError()
        return x_grads, p_grads, s_in_grads

        

        

