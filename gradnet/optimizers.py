class LinkedOptimizer(object):
    
    def __init__(self, stub):
        self.Stub = stub
        self.Context = None
    
    def deltas(self, grads):
        deltas, self.Context = self.Stub.deltas(self.Context, grads)
        return deltas
        
class OptimizerStub(object):
    
    def __call__(self):
        return LinkedOptimizer(self)

class SGD(OptimizerStub):
    
    def __init__(self, learning_rate=0.001, momentum=0.0):
        self.Eta = learning_rate
        self.Momentum = momentum
        
    def deltas(self, last_deltas, grads):
        #print("SGD.deltas: grads:", grads)
        deltas = [-self.Eta*g for g in grads]
        #print("SGD: grads:", grads)
        #print("SGD: deltas:", deltas)
        if self.Momentum:
            if last_deltas is not None:
                deltas = [d + self.Momentum*(ld-d) for d, ld in zip(deltas, last_deltas)]
            last_deltas = deltas
        return deltas, last_deltas
        
def get_optimizer(name, *params, **args):
    return {
        "SGD":  SGD
    }[name](*params, **args)

