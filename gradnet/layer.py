from .util import make_list
from .optimizers import LayerOptimizer

class Layer(object):
    
    def __init__(self, name=None, activation=None):
        self.Name = name
        self.PGradSum = None
        self.NSamples = 0
        self.Configured = False
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
            
        lnk = Link(self, shape, inputs)
        if self.Activation is not None:
            assert isinstance(self.Activation, Layer)
            lnk = self.Activation.link([lnk])
        return lnk
    
    __call__ = link
    
    def set_optimizer(self, param_optimizer):
        assert self.Configured, f"Layer {self}: Can not set layer optimizer before it is configured"
        self.Optimizer = LayerOptimizer(self.params, param_optimizer)
        
    def reset_gradients(self):
        self.PGradSum = None
        self.NSamples = 0
    
    def ____compute(self, xs, in_state):
        y, out_state, context = self.call(xs, in_state)
        if self.Activation is not None:
            z, _, a_context = self.Activation.call(y, None)       # assume activation is stateless
            context = (context, y, a_context)
            y = z
        return y, out_state, context
        
    def ____backprop(self, ygrads, sgrads, xs, y, context):
        if self.Activation is not None:
            z = y
            y_context, y, a_context = context
            ygrads, _, _ = self.Activation.grads(ygrads, None, [y], z, a_context)       # assumes activation is a stateless and param-less
            ygrads = ygrads[0]
            x_grads, p_grads, s_in_grads = self.grads(ygrads, sgrads, xs, y, y_context)
        else:
            x_grads, p_grads, s_in_grads = self.grads(ygrads, sgrads, xs, y, context)
        #print(self,".backprop: ygrads:", ygrads.shape, "   pgrads:", [g.shape for g in p_grads] if p_grads else "-")
        nsamples = len(ygrads)
        if self.PGradSum is None:
            self.PGradSum = p_grads
            self.NSamples = nsamples
        else:
            for g, g1 in zip(self.PGradSum, p_grads):
                g[...] += g1
            self.NSamples += nsamples
        return x_grads, s_in_grads
                
    def compute(self, xs, in_state):
        #print(self,".compute()")
        y, out_state, context = self.call(xs, in_state)
        return y, out_state, context
        
    def backprop(self, ygrads, sgrads, xs, y, context):
        x_grads, p_grads, s_in_grads = self.grads(ygrads, sgrads, xs, y, context)
        #print(self,".backprop: ygrads:", ygrads.shape, "   pgrads:", [g.shape for g in p_grads] if p_grads else "-")
        nsamples = len(ygrads)
        if self.PGradSum is None:
            self.PGradSum = p_grads
            self.NSamples = nsamples
        else:
            for g, g1 in zip(self.PGradSum, p_grads):
                g[...] += g1
            self.NSamples += nsamples
        #print(self, ".backprop: pgardsum:", self.PGradSum)
        #print(self, ".backprop: ygrads:", ygrads, " -> x_grads:", x_grads)
        return x_grads, s_in_grads
                
    def apply_deltas(self):
        deltas = None
        if self.PGradSum is not None and self.NSamples > 0:
            #grads = [g/self.NSamples for g in self.PGradSum]
            deltas = self.Optimizer.apply_deltas(self.PGradSum, self.params)
        self.reset_gradients()
        return deltas
        
    def set_weights(self, weights):
        if not self.Configured:
            raise RunTimeError("Layer is not configured")
        self._set_weights(weights)
                


    # overridables

    def _set_weights(self, weights):
        raise NotImpletemtedError()

    def get_weights(self):
        return None

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
        # p_grad : [...] - sum of grad over the minibatch
        # s_in_grads : [mb, ...] - grad per sample
        raise NotImplementedError()
        return x_grads, p_grads, s_in_grads

        

        

