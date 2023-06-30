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
    Alpha = 0.5                 # default alpha
    Beta = 0.5                  # max alpha

    def __init__(self, name, save_dir, target_reward=10.0, params=None):
        Primitive.__init__(self)
        self.Name = name
        self.Params = None
        self.SaveFile = save_dir + "/" + name + "_params.npz"
        self.LastActivity = 0
        self.TargetReward = target_reward
        self.Reward = None

    def alpha(self, reward = None):
        if self.Reward is None:
            return self.Alpha
        r = (reward - self.Reward)/(abs(self.Reward - self.TargetReward) + 0.001)
        if r < -0:
            exp = math.exp(r)
            return exp/(exp + 1) * self.Beta
        else:
            return 1/(1+math.exp(-r)) * self.Beta

    @synchronized
    def get(self):
        if self.Params is None and os.path.isfile(self.SaveFile):
            self.load()
        self.LastActivity = time.time()
        return self.Params
        
    @synchronized
    def set(self, params, reward=None):
        self.Params = params
        self.Reward = reward
        self.LastActivity = time.time()
        self.save()

    @synchronized
    def save(self):
        if self.Params is not None:
            np.savez(self.SaveFile, *self.Params, reward=[self.Reward])
            
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
        self.Params = weights

    @synchronized
    def update(self, params, reward=None, alpha=None):
        if alpha is None:
            alpha = self.alpha(reward)
        if isinstance(params, bytes):
            params = deserialize_weights(params)
        old_params = self.get()
        #print("Model.get: old_params:", old_params)
        if old_params is None:
            self.Params = params
        else:
            self.Params = [old + alpha * (new - old) for old, new in zip(old_params, params)]
        if reward is not None:
            self.Reward = reward if self.Reward is None else self.Reward + alpha * (reward - self.Reward)
        self.LastActivity = time.time()
        self.save()
        return self.Params
    
    @synchronized
    def reset(self):
        last_params = self.Params
        self.Params = self.Reward = None
        try:    os.remove(self.SaveFile)
        except: pass
        return last_params
    
    @synchronized
    def offload_if_idle(self):
        if time.time() > self.LastActivity + self.IdleTime and self.Params is not None:
            np.savez(self.SaveFile, *self.Params)
            self.Params = None
            
class Handler(WPHandler):
    
    Alpha = 0.2                 # for now
    
    def model(self, request, model, alpha=None):

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
            model.set(deserialize_weights(request.body))
            return 200, serialize_weights(model.get())
            
        elif request.method == "PUT":
            if alpha is not None:
                alpha = float(alpha)
            model = self.App.model(model)
            #print("handler: PUT: body:", request.body)
            params = model.update(deserialize_weights(request.body), alpha=alpha)
            #print("handler: PUT: params:", params)
            return 200, serialize_weights(params) if params else b''
            
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
