from pythreader import Primitive, PyThread, synchronized, TaskQueue
from webpie import WPApp, WPHandler
import time, os.path, numpy as np, io, traceback, sys, json, math
from gradnet import serialize_weights, deserialize_weights

def to_bytes(s):
    if isinstance(s, str):
        s = s.encode("utf-8")
    return s

def to_str(s):
    if isinstance(s, memoryview):
        s = s.tobytes()
    if isinstance(s, bytes):
        s = s.decode("utf-8")
    return s

class Model(Primitive):
    
    IdleTime = 30*60
    DefaultAlpha = 0.01                # used to calculate moving averages
    DefaultBeta = 0.1                  # default weight update constant 

    def __init__(self, name, save_dir, weights=None, alpha=None):
        Primitive.__init__(self)
        self.Name = name
        self.Weights = None
        self.SaveFile = save_dir + "/" + name + "_params.npz"
        self.LastActivity = 0
        
        self.Alpha = alpha or self.DefaultAlpha
        self.RewardMA = self.RewardSqMA = None

    def beta(self, reward = None):
        if reward is None:
            return self.DefaultBeta
        if self.RewardMA is None:
            self.RewardMA = reward
            self.RewardSqMA = reward**2
            return self.DefaultBeta
        sigma = math.sqrt(self.RewardSqMA - self.RewardMA**2)
        if sigma == 0.0:
            return self.DefaultBeta
        r = (reward - self.RewardMA)/(sigma + 0.001)
        if r < 0:
            beta = math.exp(r)/(1.0 + math.exp(r))
        else:
            beta = 1.0/(1.0 + math.exp(-r))
        return beta

    def update_reward(self, reward, alpha=None):
        if alpha is None:   alpha = self.Alpha
        self.RewardMA += alpha*(reward - self.RewardMA)
        self.RewardSqMA += alpha*(reward**2 - self.RewardSqMA)

    @synchronized
    def get_weights(self):
        if self.Weights is None and os.path.isfile(self.SaveFile):
            self.load()
        self.LastActivity = time.time()
        return self.Weights
        
    @synchronized
    def set_weights(self, weights, reward=None):
        if isinstance(weights, bytes):
            weights = deserialize_weights(weights)
        self.Weights = weights
        self.RewardMA = reward
        self.RewardSqMA = reward**2
        self.LastActivity = time.time()
        self.save()

    @synchronized
    def save(self):
        if self.Weights is not None:
            np.savez(self.SaveFile, *self.Weights, reward=[self.RewardMA])
            
    @synchronized
    def load(self):
        loaded = np.load(self.SaveFile, allow_pickle=True)
        weights = []
        self.Reward = None
        for k in loaded:
            if k == "reward":
                self.Reward = loaded[k][0]
            else:
                weights.append(loaded[k])
        self.Weights = weights

    @synchronized
    def update(self, weights, reward=None):
        beta = self.beta(reward)
        if isinstance(weights, bytes):
            weights = deserialize_weights(weights)
        old_weights = self.get_weights()
        #print("Model.get: old_params:", old_params)
        if old_weights is None:
            self.set_weights(weights, reward)
        else:
            self.Weights = [old + beta * (new - old) for old, new in zip(old_weights, weights)]
            if reward is not None:
                self.update_reward(reward, beta)
        self.LastActivity = time.time()
        self.save()
        return self.Weights
    
    @synchronized
    def reset(self):
        last_params = self.Weights
        self.Weights = self.RewardMA = self.RewardSqMA = None
        try:    os.remove(self.SaveFile)
        except: pass
        return last_params
    
    @synchronized
    def offload_if_idle(self):
        if time.time() > self.LastActivity + self.IdleTime and self.Weights is not None:
            np.savez(self.SaveFile, *self.Weights)
            self.Weights = None
            
class Handler(WPHandler):
    
    Alpha = 0.2                 # for now
    
    def model(self, request, model, reward=None):

        if request.method == "GET":
            model = self.App.model(model, create=False)
            if model is None:
                return 404, "Not found"
            else:
                return 200, serialize_weights(model.get() or [])

        elif request.method == "DELETE":
            model = self.App.model(model)
            if model is None:
                return 404, "Not found"
            else:
                model.reset()
                return 200, "OK"

        elif request.method == "POST":
            model = self.App.model(model)
            model.set_weights(deserialize_weights(request.body))
            return 200, serialize_weights(model.get())
            
        elif request.method == "PUT":
            if alpha is not None:
                alpha = float(alpha)
            model = self.App.model(model)
            #print("handler: PUT: body:", request.body)
            if reward is not None: reward = float(reward)
            weights = model.update(deserialize_weights(request.body), reward)
            #print("handler: PUT: params:", params)
            return 200, serialize_weights(weights) if params else b''
            
        else:
            return 400, "Unsupported method"
    
    def models(self, request, relpath):
        return 200, json.dumps(list(self.App.models())), "text/json"
        
    
class App(WPApp):
    
    def __init__(self, save_dir, alpha):
        WPApp.__init__(self, Handler)
        self.Alpha = alpha
        self.SaveDir = save_dir
        self.Models = {}
    
    @synchronized
    def model(self, name, create=True):
        model = self.Models.get(name)
        if model is None:
            model = self.Models[name] = Model(name, self.SaveDir, self.Alpha)
        return model
        
    def models(self):
        return self.Models.keys()

if __name__ == "__main__":
    import getopt
    opts, args = getopt.getopt(sys.argv[1:], "a:s:p:")
    opts = dict(opts)
    alpha = float(opts.get("-a", 0.5))
    storage = opts.get("-s", "models")
    port = int(opts.get("-p", 8888))
    App(storage, alpha).run_server(port, logging=True, log_file="-")
