import numpy as np, random

class Generator(object):
    
    def __init__(self, nvalues, recall_distance):
        #
        # Next number in the sequence x[t] is eirher the number generated x[t-recall_dostance], or
        # a random number from (x[t-1]-range...x[t-1]+range) % NWords but not x[t]
        #
        self.NValues = nvalues
        self.NWords = nvalues
        self.Vectors = np.eye(nvalues)
        self.Distance = recall_distance
        self.Eye = np.eye(nvalues)
        
    def initial(self, as_vector=True):
        i = random.randint(0,self.NWords-1)
        if as_vector:
            return self.Eye[i]
        else:
            return i
        
    def vectors_to_words(self, vectors):
        return np.argmax(vectors, axis=-1)
        
    def vectorize(self, x):
        return self.Eye[x]
        
    def generate(self, length, as_vectors=False):
        sequence = []
        while len(sequence) < length:
            i = random.randint(0,self.NValues-1)
            while i in sequence[-self.Distance:]:
                i = random.randint(0,self.NValues-1)
            sequence.append(i)
        return self.Eye[sequence] if as_vectors else sequence
        
    def validate(self, sequence):
        if isinstance(sequence, np.ndarray) and len(sequence.shape) > 1:
            sequence = np.argmax(sequence, axis=-1)
        length = len(sequence)
        n = 0
        for t, x in enumerate(sequence):
            t0 = max(0, t-self.Distance)
            if x in sequence[t0:t]:
                break
            n += 1
        return n

if __name__ == "__main__":
    g = Generator(10, 5)
    sequence = g.generate(50)
    print(sequence)
    print(g.validate(sequence))
                
                