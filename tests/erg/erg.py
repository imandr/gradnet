import random
import numpy as np

ERG = {
    
    -1: [('T', 0), ('P', 10)],
    
    0:  [('T', 1), ('P', 2)],
    1:  [('S', 1), ('X', 3)],
    3:  [('S', 5), ('X', 2)],
    2:  [('T', 2), ('V', 4)],
    4:  [('P', 3), ('V', 5)],
    5:  [('T', 100)],

    10:  [('T', 11), ('P', 12)],
    11:  [('S', 11), ('X', 13)],
    13:  [('S', 15), ('X', 12)],
    12:  [('T', 12), ('V', 14)],
    14:  [('P', 13), ('V', 15)],
    15:  [('P', 100)],

    100: [(' ', -1)]
}

#
# regularize the map to array
#

t2i = {t:i for i, t in enumerate(sorted(ERG.keys()))}

erg_new = []
for t, paths in sorted(ERG.items()):
    new_paths = {c:t2i[x] for c, x in paths}
    erg_new.append(new_paths)
    
ERGMap = erg_new

#print("ERG map:")
#for i, entry in enumerate(ERGMap):
#    print(i,':',entry)

Alphabet = set([' '])

for lst in ERG.values():
    for c, i in lst:
        Alphabet.add(c)
        
Alphabet = sorted(list(Alphabet))
AlphabetSize = len(Alphabet)
Eye = np.eye(len(Alphabet))
Vectors = {c:Eye[i] for i, c in enumerate(Alphabet)}
        
c2i = {c:i for i, c in enumerate(Alphabet)}
i2c = Alphabet

def encode(c):  return c2i[c]
def decode(i):  return i2c[i]

def vectorize(s):
    return np.array([Vectors[c] for c in s])
    
def unvectorize(es):
    inx = np.argmax(es, axis=-1)
    #print("unvectorize: inx:", inx.shape, inx)
    return "".join([i2c[i] for i in inx])
        
class ERGValidator(object):

    def __init__(self):
        self.T = 0
    
    def next(self, i_or_c_or_v):
        if isinstance(i_or_c_or_v, np.ndarray):
            i_or_c_or_v = int(np.argmax(i_or_c_or_v))
        if isinstance(i_or_c_or_v, str):
            c = i_or_c_or_v
        else:
            c = i2c[i_or_c_or_v]

        self.T = ERGMap[self.T].get(c)
        return self.T is not None
            
    @staticmethod
    def validate(sequence):
        v = ERGValidator()
        n = 0
        for c in sequence:
            if not v.next(c):
                break
            n += 1
        return n
                

class ERGGenerator:
    
    def __init__(self, length=None, as_vectors=False):
        self.t = -1
        self.AsVectors = as_vectors
        self.L = length
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.L == 0:
            raise StopIteration()
        choices = ERG[self.t]
        c, t = random.choice(choices)
        self.t = t
        if self.L is not None:
            self.L -= 1
        return Vectors[c] if self.AsVectors else c
        
    def choices(self):
        return tuple([c for c, t in ERG[self.t]])
        
    def encode(self, x):
        if type(x) is type('c'):
            v = np.zeros((len(Alphabet),))
            v[c2i[x]] = 1.0
            return v
        else:
            v = np.zeros((len(x), len(Alphabet)))
            for i, c in enumerate(x):
                v[i,c2i[xc]] = 1.0
            return v
            
    @staticmethod
    def initial(as_vector):
        c = random.choice(list(ERGMap[0].keys()))
        #print("initial: c:", c)
        if as_vector:
            c = Vectors[c]
        #print("       ->", c)
        return c
            
class ERGWordGenerator:
    
    def __init__(self):
        pass
        
    def __iter__(self):
        return self
        
    def word(self):
        word = ""
        for c in ERGGenerator():
            if c == ' ':
                break
            word += c
        return word
            
    def __next__(self):
        return self.generateWord()
            
if __name__ == '__main__':
    import sys
    
    print("--- 100 words ---")
    
    wg = ERGWordGenerator()
    for _ in range(100):
        print(wg.word())

    print("--- stream of 1000 letters ---")
    for c in ERGGenerator(1000):
        sys.stdout.write(c)
    print()
