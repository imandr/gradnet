import numpy as np, random
import string

class XMLValidator(object):

    CTEXT = 0
    OPEN = 1
    CLOSE = 2
    DOT = 3
    INCOMPLETE_OPEN = 4
    INCOMPLETE_CLOSE = 5
    ERROR = -1
    
    
    def __init__(self):
        self.Stack = []

    def read_tag(self, text, i):
        # assume text starts with '<'
        j = i + 1
        close_tag = False
        value = ''
        if j < len(text):
            if text[j] == '/':
                j += 1
                close_tag = True
        while j < len(text) and not text[j] in '<>/.':
            value += text[j]
            j += 1
        if j >= len(text):
            return self.INCOMPLETE_CLOSE if close_tag else self.INCOMPLETE_OPEN, value, i
        elif j > i+1 and text[j] == '>':
            return self.CLOSE if close_tag else self.OPEN, value, j+1
        else:
            return self.ERROR, text[i:j], i
        
    def read_token(self, text, i):
        c = text[i]
        if c == '<':
            return self.read_tag(text, i)
        elif c == '.':
            return self.DOT, '.', i + 1
        else:
            j = i + 1
            while j < len(text) and not text[j] in ".</>":
                j += 1
            return self.CTEXT, text[i:j], j
    
    def tokenize(self, text):
        tokens = []
        i = 0
        while i < len(text):
            token, value, j = self.read_token(text, i)
            tokens.append((token, value, i, j))
            if token in (self.ERROR, self.INCOMPLETE_OPEN, self.INCOMPLETE_CLOSE):
                break
            i = j
        return tokens
    
    def run(self, text):
        tokens = self.tokenize(text)
        stack = []
        if tokens[0][0] != self.OPEN:
            return False, 0
        for token, value, i0, i1 in tokens:
            if token == self.ERROR:
                break
            elif token == self.OPEN:
                stack.append(value)
            elif token == self.CLOSE:
                if not stack or stack.pop() != value:
                    break
            elif token == self.DOT:
                if stack:
                    break
            elif token == self.CTEXT:
                if not stack:
                    break
            elif token == self.INCOMPLETE_CLOSE:
                if not stack or not stack.pop().startswith(value):
                    break
            elif token == self.INCOMPLETE_OPEN:
                pass
        else:
            return True, len(text)
        return False, i0
                    
class Generator(object):
    
    def __init__(self, nletters):
        self.NLetters = nletters
        self.Letters = string.ascii_letters[:nletters]
        self.Alphabet = self.Letters + "<>/."
        self.C2I = {c:i for i, c in enumerate(self.Alphabet)}
        self.AlphabetSize = len(self.Alphabet)  
        self.Vectors = np.eye(self.AlphabetSize)
        self.Open = nletters
        self.Close = nletters + 1
        self.Slash = nletters + 2
        self.Dot = nletters + 3
        
    def initial(self, as_vector=True):
        i = self.Open
        if as_vector:
            return self.Vectors[i]
        else:
            return self.Alphabet[i]
        
    def vectors_to_words(self, vectors):
        inx = np.argmax(vectors, axis=-1)
        return "".join(self.Alphabet[i] for i in inx)
        
    def vectorize(self, x):
        if isinstance(x, str):
            x = self.C2I[x]
        return self.Vectors[x]
        

    MaxTagLen = 4
    
    def generateTag(self):
        l = random.randint(1,self.MaxTagLen)
        
        tag = "".join(random.choice(self.Letters) for _ in range(l))
        return tag

    def generate(self, l, as_vectors=False):
        stack = []
        length = 0
        out = []
        
        while length < l:
            r = random.random()
            if not stack and length > 0:
                out.append(".")
            if not stack or r < 0.4:
                # open new tag
                tag = self.generateTag()
                stack.append(tag)
                t = "<%s>" % (tag,)
                length += len(t)
                out.append(t)
            elif r < 0.9:
                length += 1
                out.append(random.choice(self.Letters))
            else:
                tag = stack.pop()
                t = "</%s>" % (tag,)
                length += len(t)
                out.append(t)
        text = "".join(out)
        if as_vectors:
            return np.array([self.vectorize(x) for x in text])
        else:
            return text
        
    def validate(self, sequence):
        v = XMLValidator()
        if not isinstance(sequence, str):
            sequence = self.vectors_to_words(sequence)
        ok, l = v.run(sequence)
        return l

if __name__ == "__main__":
    g = Generator(4)
    sequence = g.generate(100, as_vectors=True)
    print(sequence)
    sequence = g.vectors_to_words(sequence)
    print(sequence)
    print(g.validate(sequence))
                
                