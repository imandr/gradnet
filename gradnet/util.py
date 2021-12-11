def make_list(x):
    if x is None:   return None
    return x if isinstance(x, list) else [x]

class Callback(object):
    
    def __init__(self, fire_rate=1.0, fire_interval=None):
        self.FireRate = fire_rate
        self.NextFire = self.FireInterval = fire_interval
        self.FireLevel = 0.0
        self.FireCount = 0
        
    def __call__(self, event, *params, **args):
        self.FireCount += 1
        self.FireLevel += self.FireRate
        if self.FireInterval is not None and self.FireCount < self.NextFire:
            return
        if self.FireLevel < 1.0:
            return

        if hasattr(self, event):
            getattr(self, event)(self, *params, **args)

        self.FireLevel -= 1.0
        if self.FireInterval is not None:
            self.NextFire += self.FireInterval
