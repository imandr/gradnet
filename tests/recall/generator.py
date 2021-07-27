import numpy as np, random

class Generator(object):
    
    def __init__(self, nvalues, recall_distance, r=1):
        #
        # Next number in the sequence x[t] is eirher the number generated x[t-recall_dostance], or
        # a random number from (x[t-1]-range...x[t-1]+range) % NWords but not x[t]
        #
        self.NValues = nvalues
        self.NWords = nvalues
        self.Vectors = np.eye(nvalues)
        self.Distance = recall_distance
        self.Range = r
        self.Eye = np.eye(nvalues)
        
    def vectors_to_words(self, vectors):
        return np.argmax(vectors, axis=-1)
        
    def vectorize(self, x):
        return self.Eye[x]
        
    def generate(self, length, as_vectors=False):
        sequence = []
        t = 0
        while t < length:
            if t == 0:
                i = random.randint(0,self.NValues-1)
            else:
                r = random.random()
                if t >= self.Distance and i == 0:
                    i = sequence[t-self.Distance]
                else:
                    j = sequence[t-1]
                    i = j
                    while i == j: 
                        i = random.randint(j-self.Range, j+self.Range)
                    i = i % self.NWords
            sequence.append(i)
            t += 1
        sequence = np.array(sequence[:length])
        #print("generate: sequence:", sequence)
        return self.Vectors[sequence] if as_vectors else sequence
        
    def validate(self, sequence):
        if isinstance(sequence, np.ndarray) and len(sequence.shape) > 1:
            sequence = np.argmax(sequence, axis=-1)
        length = len(sequence)
        n = 0
        for t, x in enumerate(sequence):
            if t > 0:
                if sequence[t-1] == 0 and t >= self.Distance:
                    if x != sequence[t-self.Distance]:
                        break
                else:
                    d = x-sequence[t-1]
                    if d < 0:   d += self.NWords
                    d = d % self.NWords
                    d = min(d, self.NWords-d)
                    if d <= self.Range or t >= self.Distance and x == sequence[t-self.Distance]:
                        pass
                    else:
                        break
            n += 1
        return n

if __name__ == "__main__":
    g = Generator(10, 5, 2)
    sequence = g.generate(50)
    print(sequence)
    print(g.validate(sequence))
                
                