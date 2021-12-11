import numpy as np, random
from erg import ERGGenerator, unvectorize, vectorize, Vectors, ERGValidator, AlphabetSize

class Generator(object):
    
    NWords = AlphabetSize
    
    def __init__(self):
        pass
        
    def initial(self, as_vector=True):
        return ERGGenerator.initial(as_vector)
        
    def vectors_to_words(self, vectors):
        return unvectorize(vectors)
        
    def vectorize(self, c):
        return Vectors[c]
        
    def generate(self, length, as_vectors=False):
        lst = list(ERGGenerator(length, as_vectors=as_vectors))
        if as_vectors:  lst = np.array(lst)
        return lst
        
    def validate(self, sequence):
        return ERGValidator.validate(sequence)

if __name__ == "__main__":
    g = Generator()
    sequence = g.generate(50)
    print(sequence)
    print(g.validate(sequence))
                
                