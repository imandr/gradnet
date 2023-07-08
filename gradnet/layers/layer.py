from ..util import make_list
from ..optimizers import LayerOptimizer
from ..graphs import Node
import numpy as np

class Input(Node):
    def __init__(self, shape, name=None):
        self.Shape = shape      # tensor shape without the minibatch dimension
        self.Values = None
        self.XGradSum = None
        self.Inputs = []
        self.Layer = None
        self.Name = name
        
    def __str__(self):
        name = self.Name or "(unnamed)"
        return f"[Input {name} {self.Shape}]"
        
    def set(self, values):
        assert len(values.shape)-1 == len(self.Shape)        # remove the minibatch dimension
        assert all(s is None or s == n for s, n in zip(self.Shape, values.shape[1:])), "Incompatible shape for input layer. Expected %s, got %s" % (('*',)+self.Shape, values.shape)
        self.Values = values
        self.XGradSum = np.zeros(values.shape)
        
    def compute(self):
        return self.Values
        
    def backprop(self, grads):
        self.XGradSum[...] += grads
        
    def reset_gradsients(self):
        self.XGradSum = None
        
class Constant(Node):
    def __init__(self, value=None, name=None):
        self.Values = np.ones((1,)) if value is None else value
        self.Inputs = []
        self.Layer = None
        self.Name = name
        
    def __str__(self):
        name = self.Name or "(unnamed)"
        return f"[Constant {name} {self.Value}]"
        
    def compute(self):
        return self.Value
        
    def backprop(self, grads):
        pass
        
    def reset_gradsients(self):
        pass
        
class Layer(object):
    
    def __init__(self, name=None, activation=None):
        self.Name = name
        self.Params = {}
        self.PGradSum = None
        self.NSamples = 0
        self.Configured = False
        self.StateInGradients = self.XGradients = self.WeightsGradients = None     # saved for inofrmation purposes. Not used by the Layer itself

        if isinstance(activation, str):
            from ..activations import get_activation
            self.Activation = get_activation(activation)
        else:
            self.Activation = activation

    def __str__(self):
        return "[Layer %s %s]" % (self.__class__.__name__, self.Name or "")        

    def link(self, *inputs):
        from ..graphs import Node
        
        if len(inputs) == 1:
            if isinstance(inputs[0], (list, tuple)):
                inputs = list(inputs[0])
            else:
                inputs = [inputs[0]]
        else:
            inputs = list(inputs)
    
        #print(self, ".link(): inputs:", inputs)

        if not self.Configured:
            shape = self.configure(inputs)
            #print("     self.Shape -> ", self.Shape)
            self.Configured = True
        else:
            shape = self.check_configuration(inputs)
            
        lnk = Node(self, shape, inputs)
        if self.Activation is not None:
            assert isinstance(self.Activation, Layer)
            lnk = self.Activation.link([lnk])
        return lnk
    
    __call__ = link
    
    def set_optimizer(self, param_optimizer):
        assert self.Configured, f"Layer {self}: Can not set layer optimizer before it is configured"
        self.Optimizer = LayerOptimizer(self.weights, param_optimizer)
        
    def reset_gradients(self):
        #print("Layer.reset_gradients:", self)
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
        
        self.XGradients = x_grads                   # saved for inofrmation purposes. Not used by the Layer itself
        self.StateInGradients = s_in_grads          # saved for inofrmation purposes. Not used by the Layer itself
        
        return x_grads, s_in_grads
                
    def apply_deltas(self):
        deltas = None
        self.WeightsGradients = None if self.PGradSum is None else [g.copy() for g in self.PGradSum]           # saved for inofrmation purposes. Not used by the Layer itself
        if self.PGradSum is not None and self.NSamples > 0:
            #grads = [g/self.NSamples for g in self.PGradSum]
            deltas = self.Optimizer.apply_deltas(self.PGradSum, self.weights)
        self.reset_gradients()
        self.Deltas = deltas
        return deltas
        
    def set_weights(self, weights):
        if not self.Configured:
            raise RuntimeError("Layer is not configured")
        self._set_weights(weights)
                
    def as_jsonable(self):
        out = {
            "class": self.__class__.__name__,
            "name": self.Name,
            "params": self.params()
        }
        if self.Activation is not None:
            out["activation"] = self.Activation.as_jsonable()
        return out

    #
    # overridables
    #

    @property
    def weights(self):
        return []

    def params(self):
        return self.Params

    def _set_weights(self, weights):
        raise NotImplementedError()
        
    def get_weights(self):
        return [w.copy() for w in self.weights]
        
    def configure(self, inputs):
        raise NotImplementedError()
        pass
        # return shape
        
    def check_configuration(self, inputs):
        pass
        # return shape
        
    def compute(self, inputs, in_state):
        raise NotImplementedError(f"Layer {self.__class__.__name__} does not implement compute() method")
        return y, out_state, context
        
    def grads(self, y_grads, s_out_grads, xs, y, context):
        # y_grads: [mb, ...] - dL/dy[j]
        # returns:
        # x_grads : [mb, ...] - grad per sample
        # p_grad : [...] - sum of grad over the minibatch
        # s_in_grads : [mb, ...] - grad per sample
        raise NotImplementedError()
        return x_grads, p_grads, s_in_grads

    def check_gradients(self, input_shapes, minibatch=1, attempts=1000, 
                        xs = None, state_in="init",
                        include_state=True, include_value=True,
                        tolerance=0.001, 
                        relative_tolerance = 0.01, 
                        delta = 1.0e-4):
                        
        import numpy as np
        import random
        
        def tolerated(g1, g2):
            g12 = (g1+g2)/2
            return abs(g1-g2) < tolerance or \
                g12 != 0 and abs(g1-g2)/g12 < relative_tolerance
            
        
        def loss_y_and_s(y, s):
            if s is None:
                return np.sum(y), np.ones(y.shape), None
            else:
                s_is_list = isinstance(s, (list, tuple))
                if s_is_list:
                    ssum = sum(np.sum(si) for si in s)
                    sgrads = [np.ones(si.shape) for si in s]
                else:
                    ssum = np.sum(s)
                    sgrads = np.ones(s.shape)
                return np.sum(y) + ssum, np.ones(y.shape), sgrads
                
        def loss_s_only(y, s):
            s_is_list = isinstance(s, (list, tuple))
            if s_is_list:
                ssum = sum(np.sum(si) for si in s)
                sgrads = [np.ones(si.shape) for si in s]
            else:
                ssum = np.sum(s)
                sgrads = np.ones(s.shape)
            return ssum, np.zeros(y.shape), sgrads
                
        def loss_y_only(y, s):
            return np.sum(y), np.ones(y.shape), None
            
        from gradnet import Input
        
        # input shapes are shapes without the minibatch dimension
        input_shapes = make_list(input_shapes)
        inputs = [Input(s) for s in input_shapes]
        link = self.link(inputs)
        out_shape = link.Shape
        weights = self.get_weights()
        #print("check_grads: weights:", weights)
        xs = make_list(xs)
        if xs is None:
            x0s = [np.random.random((minibatch,)+s)*2-1 for s in input_shapes]
        else:
            x0s = xs
        #x0s = [np.ones((1,)+s) for s in input_shapes]
        w0s = [w.copy() for w in weights]
        if state_in == "init":
            _, s_in_0, _ = self.compute(x0s, None)       # just to get a sample of a possible input state
        else:
            s_in_0 = state_in
        #s_in_0[...] = 0.0

        y0, s_out_0, context = self.compute(x0s, s_in_0)
        #print("y0=", y0)
        #print("s_out_0=", s_out_0)
        
        loss = loss_y_only if not include_state else \
            (loss_s_only if not include_value else loss_y_and_s)
        
        l0, dldy, dlds = loss(y0, s_out_0)
        #print("check_gradients: dldy:", dldy.shape)
        #print("check_gradients: x0s:", [x.shape for x in x0s])
        
            

        #
        # have the layer colculate gradients
        #
        x_grads, w_grads, s_in_grads = self.grads(dldy, dlds, x0s, y0, context)
        #print("check_grads: point gradients:")
        #print("  x:", x_grads)
        #print("  w:", w_grads)
        #print("  s:", s_in_grads)


        x_errors = w_errors = s_errors = 0
        
        #
        # X gradients
        #


        for t in range(attempts):
            x1s = [x.copy() for x in x0s]
            i = random.randint(0, len(x1s)-1)
            xi = x1s[i]
            xif = xi.reshape((-1,))
            n = xif.shape[-1]
            j = random.randint(0, n-1)
            inx = np.unravel_index(j, xi.shape)
            xif[j] += delta

            y1, s_out, _ = self.compute(x1s, s_in_0)
            l1, _, _ = loss(y1, s_out)
            #print("y1:", y1)
            #print("x1s:", x1s)
            #print("Loss values", l0, l1)

            
            dldx = (l1-l0)/delta

            # compare to the gradients returned by the layer
            grad_i = x_grads[i]
            dldx_l = grad_i[inx]

            if not tolerated(dldx_l,dldx):
                print(f"==== Detected difference in dL/dx[{i}][{inx}]: computed:{dldx_l}, numericaly calculated:{dldx}")
                x_errors += 1
        #
        # Weights gradients
        #
        if w_grads:
            for t in range(attempts):
                w1s = [w.copy() for w in w0s]
                i = random.randint(0, len(w1s)-1)
                wi = w1s[i]
                wif = wi.reshape((-1,))
                n = wif.shape[-1]
                j = random.randint(0, n-1)
                inx = np.unravel_index(j, wi.shape)
                wif[j] += delta

                self.set_weights(w1s)
                y1, s_out, _ = self.compute(x0s, s_in_0)
                l1, _, _ = loss(y1, s_out)
                self.set_weights(w0s)

                dldw = (l1-l0)/delta

                # compare to the gradients returned by the layer
                grad_i = w_grads[i]
                #print("check_grads: i=",i,"   grad_i=", grad_i.shape, "   inx=", inx)
                dldw_l = grad_i[inx]
            
                if not tolerated(dldw_l, dldw):
                    print(f"==== Detected difference in dL/dw[{i}][{inx}]: computed:{dldw_l}, numericaly calculated:{dldw}")
                    w_errors += 1
        #
        # Input state gradients
        #
        if s_in_0 is not None and s_in_grads is not None:
            state_is_list = isinstance(s_in_0, (list, tuple))

            s_in = make_list(s_in_0)
            s_in_grads = make_list(s_in_grads)

            for t in range(attempts):
                s1 = [s.copy() for s in s_in]
                i = random.randint(0, len(s1)-1)
                si = s1[i]
                sif = si.reshape((-1,))
                n = sif.shape[-1]
                j = random.randint(0, n-1)
                inx = np.unravel_index(j, si.shape)
                sif[j] += delta

                y1, s_out, _ = self.compute(x0s, s1 if state_is_list else s1[0])
                l1, _, _ = loss(y1, s_out)

                dlds = (l1-l0)/delta

                # compare to the gradients returned by the layer
                dlds_l = s_in_grads[i][inx]

                if not tolerated(dlds_l, dlds):
                    print(f"==== Detected difference in dL/ds[{i}][{inx}]: computed:{dlds_l}, numericaly calculated:{dlds}")
                    s_errors += 1
        return (x_errors, w_errors, s_errors)
