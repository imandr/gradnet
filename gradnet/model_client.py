import time, os.path, numpy as np, io, traceback, sys, requests
from .serialization import serialize_weights, deserialize_weights

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

class ModelClient(object):
    
    def __init__(self, model_name, url_head):
        self.URLHead = url_head
        self.ModelName = model_name
        
    def get(self):
        response = requests.get(self.URLHead + "/model/" + self.ModelName)
        if response.status_code == 404:
            return None
        if response.status_code // 100 != 2:
            print(response)
            print(response.text)
            response.raise_for_status()
        return deserialize_weights(response.content) if response.content else None
    
    def update(self, params):
        response = requests.put(self.URLHead + "/model/" + self.ModelName, data=serialize_weights(params))
        if response.status_code // 100 != 2:
            print(response)
            print(response.text)
            response.raise_for_status()
        weights = deserialize_weights(response.content)
        #print("ModelClient.update: deserialized:")
        #for w in weights:
        #    print(w.shape, w.data, w.data.readonly)
        return weights
        
    def reset(self):
        response = requests.delete(self.URLHead + "/model/" + self.ModelName)
        if response.status_code // 100 != 2:
            print(response)
            print(response.text)
            response.raise_for_status()
