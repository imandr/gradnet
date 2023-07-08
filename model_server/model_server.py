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
    DefaultAlpha = 0.1                # used to calculate moving averages
    DefaultBeta = 0.2                  # default weight update constant 

    def __init__(self, name, save_dir, weights=None, beta=None):
        Primitive.__init__(self)
        self.Name = name
        self.Weights = None
        self.SaveFile = save_dir + "/" + name + "_weights.npz"
        self.MetaFile = save_dir + "/" + name + "_meta.json"
        self.LastActivity = 0
        
        self.Alpha = self.DefaultAlpha
        self.RewardMA = self.RewardSqMA = None
        self.Sigma = 1.0
        self.Beta = beta or self.DefaultBeta
        
    @property
    def reward(self):
        return self.RewardMA

    def beta(self, reward=None):
        if reward is None:
            if self.RewardMA is None:
                return 1.0          # single tariner ?
            return self.Beta
        if self.RewardMA is None:
            self.RewardMA = reward
            self.RewardSqMA = reward**2
            return self.Beta
        sigma = self.Sigma
        if sigma == 0.0:
            return self.Beta
        r = (reward - self.RewardMA)/(sigma + 0.001)
        if r < 0:
            beta = math.exp(r)/(1.0 + math.exp(r))
        else:
            beta = 1.0/(1.0 + math.exp(-r))
        return beta

    @synchronized
    def update_sigma(self, reward):
        alpha = self.Alpha
        if self.RewardMA is None:
            self.RewardMA = reward
            self.RewardSqMA = reward**2
        old_reward_ma = self.RewardMA
        self.RewardMA += alpha*(reward - self.RewardMA)
        self.RewardSqMA += alpha*(reward**2 - self.RewardSqMA)
        sigma = math.sqrt(self.RewardSqMA - self.RewardMA**2)
        old_sigma = self.Sigma
        self.Sigma += alpha*(sigma - self.Sigma)
        print("update_sigma: alpha:", alpha, "  reward:", reward)
        print("    reward_ma:", old_reward_ma, "->", self.RewardMA)
        print("    sigma:", old_sigma, "->", self.Sigma)

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
        if reward is None:
            self.RewardMA = self.RewardSqMA = None
        else:
            self.RewardMA = reward
            self.RewardSqMA = reward**2
        self.Sigma = 1.0
        self.LastActivity = time.time()
        self.save()

    @synchronized
    def save(self):
        if self.Weights is not None:
            np.savez(self.SaveFile, *self.Weights, reward=[self.RewardMA])
        json.dump({
            "reward_ma": self.RewardMA,
            "reward2_ma": self.RewardSqMA
        }, open(self.MetaFile, "w"))
            
    @synchronized
    def load(self):
        loaded = np.load(self.SaveFile, allow_pickle=True)
        weights = []
        self.RewardMA = self.RewardSqMA = None
        for k in loaded:
            if k == "reward":
                self.Reward = loaded[k][0]
            else:
                weights.append(loaded[k])
        self.Weights = weights
        try:
            meta = json.load(open(self.MetaFile, "r"))
            self.RewardMA = meta.get("reward_ma")
            self.RewardSqMA = meta.get("reward2_ma")
        except:
            pass

    @synchronized
    def update(self, weights, reward=None):
        if reward:
            self.update_sigma(reward)
        beta = self.beta(reward)
        print("Handler.update: beta:", beta)
        if isinstance(weights, bytes):
            weights = deserialize_weights(weights)
        old_weights = self.get_weights()
        #print("Model.get: old_weights:", old_weights)
        if old_weights is None:
            self.set_weights(weights, reward)
        else:
            self.Weights = [old + beta * (new - old) for old, new in zip(old_weights, weights)]
            print(f"Weights for {self.Name} updated using beta:", beta, "  rewardMA/reward:", self.RewardMA, reward, "  sigma:", self.Sigma)
        self.LastActivity = time.time()
        self.save()
        print(f"model {self.Name} saved")
        return self.Weights
    
    @synchronized
    def reset(self):
        last_weights = self.Weights
        self.Weights = self.RewardMA = self.RewardSqMA = None
        self.Sigma = 1.0
        try:    os.remove(self.SaveFile)
        except: pass
        return last_weights
    
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
                if reward:
                    return 200, json.dumps(model.reward), "text/json"
                else:
                    return 200, serialize_weights(model.get_weights() or [])

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
            
        elif request.method == "PATCH":
            model = self.App.model(model)
            #print("handler: PUT: body:", request.body)
            if reward is not None: reward = float(reward)
            weights = model.update(deserialize_weights(request.body), reward)
            #print("handler: PUT: weights:", weights)
            return 200, serialize_weights(weights) if weights else b''
            
        else:
            return 400, "Unsupported method"
    
    def models(self, request, relpath):
        return 200, json.dumps(list(self.App.models())), "text/json"
        
    
class App(WPApp):
    
    def __init__(self, save_dir, beta):
        WPApp.__init__(self, Handler)
        self.Beta = beta
        self.SaveDir = save_dir
        self.Models = {}
    
    @synchronized
    def model(self, name, create=True):
        model = self.Models.get(name)
        if model is None:
            model = self.Models[name] = Model(name, self.SaveDir, self.Beta)
        return model
        
    def models(self):
        return self.Models.keys()

if __name__ == "__main__":
    import getopt
    opts, args = getopt.getopt(sys.argv[1:], "b:s:p:")
    opts = dict(opts)
    beta = None
    if "-b" in opts:    beta = float(opts.get("-b"))
    storage = opts.get("-s", "models")
    port = int(opts.get("-p", 8888))
    App(storage, beta).run_server(port, logging=True, log_file="-")
