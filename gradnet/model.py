from .util import make_list
import numpy as np, json
from .optimizers import get_optimizer
from .losses import get_loss
import random

class GradientAccumulator(object):
    
    def __init__(self, model):
        self.Model = model
        
    def __enter__(self):
        self.Model.reset_losses()
        return self
        
    def accumulate(self, x, y_=None, data={}):
        return self.Model.accumulate_gradients(x, y_, data)
        
    __call__ = accumulate
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.Model.apply_deltas()
    
class Model(object):
    
    def __init__(self, inputs, outputs):
        self.Inputs = make_list(inputs)        # inputs
        self.Outputs = make_list(outputs)      # paths
        self.Losses = {}        # {name -> (loss, weight)}
        self.OutLosses = []
        self.Ys = None
        self.Optimizer = None
        self.Metrics = []
        self.LossValues = {}
        self.AllLayers = []
        self.Map = {}
        seen = set()
        for out in self.Outputs:
            self.AllLayers += list(self._layers_rec(out, seen))
            
    def add_loss(self, loss, weight=1.0, name=None):
        name = name or "loss_%d" % (len(self.Losses),)
        if isinstance(loss, str):
            loss = get_loss(loss)(*loss_args)
        self.Losses[name] = (loss, weight)
        return self
        
    def compile(self, optimizer=None, metrics=[], **optimizer_args):
        self.Metrics = metrics
        if optimizer is not None:
            if isinstance(optimizer, str):
                optimizer = get_optimizer(optimizer, **optimizer_args)
            for layer in self.AllLayers:
                layer.set_optimizer(optimizer)
        
    def _layers_rec(self, link, seen):
        layer = link.Layer
        if layer is not None and not id(layer) in seen:
            seen.add(id(layer))
            yield layer
        for lnk in link.Inputs:
            yield from self._layers_rec(lnk, seen)

    @property
    def layers(self):
        return self.AllLayers

    def get_weights(self):
        lst = []
        for l in self.layers:
            n = len(l.weights)
            if n:
                lst += list(l.get_weights())
        return lst

    def set_weights(self, source):
        self.update_weights(source, 1.0)

    def update_weights(self, source, alpha=1.0):
        if isinstance(source, Model):
            assert len(self.layers) == len(source.layers), "Source and destination models have different number of layers"
            source = source.get_weights()
        else:
            assert isinstance(source, list)
        i = 0
        saved = self.get_weights()
        for l in self.layers:
            n = len(l.weights)
            if n:
                if alpha != 1.0:
                    for w, w1 in zip(l.weights, source[i:i+n]):
                        w += alpha*(w1-w)
                else:
                    l.set_weights(source[i:i+n].copy())
                i += n
        assert i == len(source), "Not all source weights were used in update_weights()"
        return saved
        
    def save_weights(self, filename):
        weights = self.get_weights()
        if weights:
            np.savez(filename, *weights)
            
    def load_weights(self, filename):
        loaded = np.load(filename)
        self.set_weights([loaded[k] for k in loaded])

    def __setitem__(self, name, v):
        self.Map[name] = v
        
    def __getitem__(self, name):
        return self.Map[name]
        
    #
    # training workflow:
    #
    # reset_losses()
    # loop(
    #    call(inputs)
    #    backprop(y_, data)
    # )
    # apply_deltas()
        
    def compute(self, inputs):
        """Computes the values of the network given the inputs
        
        inputs:
            ndarray: converted to [ndarray]
            (ndarray, ndarray, ...): converted to [ndarray, ndarray, ...]
            Each ndarray must have additional, first minibatch dimension
        
        returns:
            [ndarray, ...]
        """
        
        
        #print("------------------- Model.compute() ---------------------")
        inputs = make_list(inputs)
        assert len(inputs) == len(self.Inputs)
        assert all(x.shape[1:] == inp.Shape for (x, inp) in zip(inputs, self.Inputs))
        for o in self.Outputs:
            o.reset()
        for i, x in zip(self.Inputs, inputs):
            #print("Model.compute(): setting input to:", x)
            i.set(x)
        self.Ys = [o.compute() for o in self.Outputs]
        return self.Ys
        
    predict = compute       # alias
        
    def backprop(self, y_=None, data={}):
        assert isinstance(data, dict)
        d = data.copy()
        d["y_"] = y_
        values = {}
        for name, (loss, weight) in self.Losses.items():
            #print("Model.backprop: computing loss", name)
            values[name] = lv = loss.compute(d)
            grads = loss.Grads
            #print(loss, ": grads:", None if grads is None else [(g.shape, np.mean(g*g)) if g is not None else "-" for g in grads])
            
            #print(f"model.backprop: loss[{name}]:", lv)
            #print("Model.backprop: loss:", name, loss.__class__.__name__, weight)
            if weight:
                self.LossValues[name] = self.LossValues[name] + lv
                #print("      backpropping...")
                loss.backprop(weight)
        return values
            
    def reset_losses(self):
        self.LossValues = {name:0.0 for name in self.Losses.keys()}
        for o in self.Outputs:
            o.reset_gradients()
            
    def reset_states(self):
        #print("Model.reset_state()")
        for o in self.Outputs:
            o.reset_state()
            
    reset_state = reset_states              # deprecated
            
    def apply_deltas(self):
        out = []
        for layer in self.AllLayers:
            deltas = layer.apply_deltas()
            if deltas is not None:
                out.append(deltas)
        return out
        
    def accumulate_gradients(self, x, y_=None, data={}):
        x = make_list(x)
        self.compute(x)
        return self.backprop(y_, data)
        
    def accumulator(self):
        return GradientAccumulator(self)

    def metrics(self, y_, metrics):
        #print("metrics: Ys:", self.Ys[:3], "   y_:", y_[:3], "  metrics:", metrics)
        return [m(y_, self.Ys[0]) for m in metrics] if y_ is not None else None
            
    def train_on_batch(self, x, y_=None, data={}, metrics=[]):
        with self.accumulator() as acc:
            loss_values = acc.accumulate(x, y_, data)
        return loss_values, self.metrics(y_, metrics)
            
    def fit(self, x, y_=None, data={}, batch_size=None, metrics=[], callbacks=[], shuffle=False):
        # will always reset state for each batch !
        # y_ is a single ndarray, not a list
        x = make_list(x)
        n = len(x[0])           # sample size

        assert y_ is None or isinstance(y_, np.ndarray)
        assert all(len(xi) == n for xi in x)
        assert y_ is None or len(y_) == n
        assert all(len(d) == n for d in data.values())

        #print("before shuffle: x:", *tuple(xi.shape for xi in x))

        if shuffle:
            inx = np.arange(n)
            random.shuffle(inx)
            print("inx:", inx.shape, inx[:10])
            if y_ is not None:
                y_ = y_[inx]
            x = [xi[inx] for xi in x]
            data = {k:d[inx] for k, d in data.items()}

        print("after shuffle: x:", *tuple(xi.shape for xi in x))

        if batch_size is None:  batch_size = n
        samples = 0
        for i in range(0, n, batch_size):
            #print("fit(): i, batch_size:", i, batch_size)
            xi = [xx[i:i+batch_size] for xx in x]
            yi_ = None if y_ is None else y_[i:i+batch_size]
            data_i = {k:d[i:i+batch_size] for k, d in data.items()}
            self.reset_state()
            loss_values, mvalues = self.train_on_batch(xi, yi_, data_i, metrics)
            samples += len(xi[0])
            for cb in callbacks:
                #print("callback:", cb)
                method = None
                if hasattr(cb, "train_batch_end"):
                    method = getattr(cb, "train_batch_end")
                elif callable(cb):
                    method = cb
                else:
                    continue
                method(samples, loss_values, mvalues)
            #print(loss_values, mvalues)

        return loss_values, mvalues
        
    def layer_gradients(self):
        out = []
        for l in self.layers:
            lst = l.PGradSum
            #print("Model.layer_gradients: layer:", l,"   grads:", l.PGradSum)
            if lst:
                out += lst
        return out
        
    def input_gradients(self):
        return [i.XGradSum for i in self.Inputs]
        
    def links(self):
        seen = set()
        for o in self.Outputs:
            for l in o.links(seen):
                yield l
            yield o

    def sorted_links(self):
        # returns topologically sorted layer list from inputs to outputs
        inputs_map = {}
        links_by_id = {}
        for l in self.links():
            lid = id(l)
            if lid not in links_by_id:
                inputs_map[lid] = set(i.id() for i in l.Inputs)
                links_by_id[lid] = l

        #
        # sort the links topologically, lowest inputs first
        #
        
        sorted_list = []
        while inputs_map:
            for lid, inputs in list(inputs_map.items()):
                if not inputs:
                    yield links_by_id[lid]
                    inputs_map.pop(lid)
                    # remove this node from all its parents
                    for _, parent_inputs in list(inputs_map.items()):
                        if lid in parent_inputs:    parent_inputs.remove(lid)
                    break
            else:
                raise ValueError("Circular links detected")
    
    def as_json(self):
        data = [l.as_jsonable() for l in self.sorted_links()]
        return json.dumps(data)
        
        

if __name__ == "__main__":
    from graphs import Input
    from layers import Dense
    from activations import get_activation
    from optimizers import get_optimizer
    from losses import get_loss
    from metrics import get_metric
    import numpy as np
    
    from tensorflow.keras.datasets import mnist
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    def one_hot(labels, n):
        out = np.zeros((len(labels), n))
        for i in range(n):
            out[labels==i, i] = 1.0
        return out
        
    x_train = (x_train/256.0).reshape((-1, 28*28))
    x_test = (x_test/256.0).reshape((-1, 28*28))
    n_train = len(x_train)
    y_train = one_hot(y_train, 10)
    y_test = one_hot(y_test, 10)
    
    np.set_printoptions(precision=4, suppress=True)
    
    relu = get_activation("relu")
    sgd = get_optimizer("SGD", learning_rate=0.005, momentum=0.5)
    cce = get_loss("cce")
    mse = get_loss("mse")
    accuracy = get_metric("accuracy")
    
    inp = Input((28*28,))
    dense1 = Dense(1024, name="dense1", activation="relu")(inp)
    top = Dense(10, name="top")(dense1)
    probs = get_activation("softmax", name="softmax")(top)
    l = cce(probs)
    
    model = Model([inp], [probs])
    model.compile(sgd, output_losses=[l])

    mbsize = 30

    for epoch in range(10):
        for i in range(0, n_train, mbsize):
            batch = x_train[i:i+mbsize]
            labels = y_train[i:i+mbsize]
            p, losses, _ = model.train(batch, labels, [])
            
            
        y = model.compute(x_test)
        y_ = y_test
        acc = accuracy(y_test, y[0])
        print("test accuracy:", acc)
        
    
    
    
    
    
    
    
    
    
    
    
    

        

    