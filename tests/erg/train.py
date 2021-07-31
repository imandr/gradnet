from generator import Generator
import numpy as np, random

np.set_printoptions(precision=4, suppress=True, linewidth=132)

from gradnet import Model
from gradnet.layers import LSTM, LSTM_Z, Input, Dense
from gradnet.losses import LossBase

class NormalizedCategoricalCrossEntropy(LossBase):
    
    def compute(self, data):
        from gradnet.activations import SoftMaxActivation
        p_ = data["y_"]
        inp = self.Inputs[0]
        p = inp.Y
        
        #print("cce: p:", p)
        pmax = np.max(p, axis=-1, keepdims=True)
        self.Values = -np.sum(p_*np.log(np.clip(p/pmax, 1e-6, None)), axis=-1)             
        #print("CCE.values:", self.Values.shape, "  p:",p.shape, "  p_:", p_.shape)
        
        inp = self.Inputs[0]
        if isinstance(inp.Layer, SoftMaxActivation):
            # if the input layer is SoftMax activation, bypass it and send simplified grads to its input
            #print("CategoricalCrossEntropy: sending simplified grads")
            self.Grads = p - p_
        else:
            self.Grads = -p_/np.clip(p, 1.0e-2, None)
        #print("CCE.compute(): p:", p)
        #print("              p_:", p_)
        return self.value
        
    def backprop(self, weight=1.0):
        from gradnet.activations import SoftMaxActivation
        inp = self.Inputs[0]
        if isinstance(inp.Layer, SoftMaxActivation):
            # if the input layer is SoftMax activation, bypass it and send simplified grads to its input
            inp.Inputs[0].backprop(self.Grads*weight)
        else:
            #print("CategoricalCrossEntropy: sending grads to:", inp, inp.Layer)
            inp.backprop(self.Grads*weight)
                


def create_net(nwords, hidden=100):
    inp = Input((None, nwords))
    r1 = LSTM(hidden, return_sequences=True)(inp)
    #r2 = LSTM(hidden, return_sequences=True)(r1)
    probs = Dense(nwords, activation="softmax")(r1)
    model = Model(inp, probs)
    model.add_loss(NormalizedCategoricalCrossEntropy(probs), name="CCE")
    model.compile("adagrad", learning_rate=0.01)
    return model
    
def generate_from_model(model, g, length, batch_size):
    #print("------- generate ----------")
    model.reset_states()
    nwords = g.NWords
    eye = np.eye(nwords)
    
    row_v = np.array([g.initial(True) for _ in range(batch_size)])      # [b]
    rows = [row_v]
    
    for t in range(length-1):
        y = model.compute(row_v[:,None,:])[0][:,0,:]            # y: [mb, w], t=0
        pvec = y**3
        pvec = pvec/np.sum(pvec, axis=-1, keepdims=True)        # -> [mb, w]
        
        row_i = [np.random.choice(nwords, p=p) for p in pvec]
        row_v = eye[row_i]
        rows.append(row_v)
    
    rows = np.array(rows).transpose((1,0,2))    # -> [mb, t, w]
    #print("generate_from_model: rows:", rows.shape, rows)
    return [g.vectors_to_words(vectors) for vectors in rows]
    
def generate_batch(g, length, batch_size):
    
    #print("generate_batch(%s, %s)..." % (length, batch_size))
    
    sequences = np.array([g.generate(length+1, as_vectors=True) for _ in range(batch_size)])
    
    #print("sequences:", sequences.shape)
    
    x = sequences[:,:-1,:]
    y_ = sequences[:,1:,:]
    
    return x, y_
    
def test(model, g, length, batch_size, samples):
    #print(type(generated), generated.shape, generated)
    
    sum_valid = nsamples = 0
    while nsamples < samples:
        generated = generate_from_model(model, g, length, batch_size)
        sum_valid += np.sum([g.validate(s) for s in generated])
        nsamples += len(generated)
    return sum_valid/nsamples, generated[0]
    
    
def train(model, g, length, batch_size, goal):
    valid_ma = 0.0    
    steps = 0
    episodes = 0
    
    for iteration in range(100000):
        #print
        
        x, y_ = generate_batch(g, length, batch_size)
        losses, metrics = model.fit(x, y_)
        steps += length*batch_size
        episodes += batch_size
        
        if iteration and iteration % 100 == 0:
            average_valid, generated = test(model, g, length, batch_size, 100)
            valid_ma += 0.1*(average_valid-valid_ma)

            valid_length = g.validate(generated)
            print("%s<>%s" % (generated[:valid_length], generated[valid_length:]), " valid length:", valid_length)
            print("Batches:", iteration, "  steps:", iteration*length*batch_size, "  loss/step:", losses["CCE"]/x.shape[1]/batch_size,
               "  average valid length:", average_valid)
            if average_valid >= goal:
                return episodes, steps, valid_ma
            
if __name__ == '__main__':
    import getopt, sys
    
    opts, args = getopt.getopt(sys.argv[1:], "b:l:")
    opts = dict(opts)
    
    batch_size = int(opts.get("-b", 20))
    length = int(opts.get("-l", 50))
        
    g = Generator()
    model = create_net(g.NWords)
    episodes, steps, valid_ma = train(model, g, length, batch_size, length*0.95)
    print("episodes=", episodes, "  steps=", steps, "   valid_length=", valid_ma)