import numpy as np

class Metric(object):
    
    def compute(self, y_, y):
        raise NotImplementedError()
        
    def __call__(self, y_, y):
        return self.compute(y_, y)
        
class AccuracyMetric(Metric):
    
    def compute(self, y_, y):
        a_ = np.argmax(y_, axis=-1)
        a = np.argmax(y, axis=-1)
        #print(a_,a)
        return np.sum(a==a_)/len(a)
        
def get_metric(name):
    return {
        "accuracy": AccuracyMetric
    }[name]()