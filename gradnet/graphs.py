import numpy as np
from .util import make_list

class Computer(object):
    
    def compute(self):
        raise NotImplementedError()
        return y
        
    def reset_cached(self):
        pass
        
    def reset_state(self):
        pass
        
    def add_grads(self, y_grads):
        raise NotImplementedError()
    
    def apply_grads(self):
        raise NotImplementedError()

    def grads(self, y_grads):
        raise NotImplementedError()
        return x_grads, p_grads

    def call(self, x):
        raise NotImplementedError()
        return y
        
    def apply_deltas(self, deltas):
        raise NotImplementedError()
        
class Node(object):
    
    def __init__(self, layer, shape, inputs):
        self.Inputs = inputs
        self.Layer = layer
        self.Shape = shape
        self.Xs = self.Y = self.StateGrads = self.InState = self.OutState = self.Context = None
        if self.Layer is not None:
            self.Layer.reset_gradients()

    def id(self):
        return id(self)

    def __hash__(self):
        return hash(id(self))
        
    def as_jsonable(self):
        return self.Layer.as_jsonable()

    def reset(self):
        #
        # get ready for next compute(), but do not reset accumulated grads, nor the state for stateful layers
        #
        self.Xs = self.Y = self.Context = None
        for i in self.Inputs:
            i.reset()
            
    def reset_state(self):
        #
        # reset state for stateful layers
        #
        self.InState = self.OutState = None
        for i in self.Inputs:
            i.reset_state()
            
    def reset_gradients(self):
        self.StateGrads = None
        if self.Layer is not None:
            self.Layer.reset_gradients()
        for i in self.Inputs:
            i.reset_gradients()
        
    def compute(self):
        if self.Y is None:
            self.InState = self.OutState
            self.Xs = [i.compute() for i in self.Inputs]
            self.Y, self.OutState, self.Context = self.Layer.compute(self.Xs, self.InState)
        return self.Y

    def backprop(self, y_grads, state_grads=None):
        #assert isinstance(y_grads, np.ndarray) and y_grads.shape[1:] == self.Shape
        xgrads, self.StateGrads = self.Layer.backprop(y_grads, state_grads, self.Xs, self.Y, self.Context)
        if xgrads is None:
            return
        assert isinstance(xgrads, list) and len(xgrads) == len(self.Inputs)
        #print(f"    x_grads from layer {self.Layer}:", [g.shape if g is not None else "" for g in xgrads])
        for xg, i in zip(xgrads, self.Inputs):
            if xg is not None:
                i.backprop(xg)
                
    def links(self, seen=None):
        if seen is None:    seen = set()
        for i in self.Inputs:
            if isinstance(i, Node):
                for l in i.links(seen):
                    if not id(l) in seen:
                        yield l
                        seen.add(id(l))
                if not id(i) in seen:
                    yield i
                    seen.add(id(i))
    
    def inputs_map_rec(self, update_map=None):
        # update_map: { node id -> (node, [input_id, ...])}
        if update_map is None:
            update_map = {}
        my_id = self.id()
        if my_id not in update_map:
            update_map[my_id] = (self, [n.id() for n in self.Inputs])
            for n in self.Inputs:
                n.inputs_map_rec(update_map)
        return update_map

        
