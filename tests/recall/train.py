from generator import Generator
import numpy as np, random

np.set_printoptions(precision=4, suppress=True, linewidth=132)

from gradnet import Model
from gradnet.layers import LSTM, LSTM_Z, Input, Dense

def create_net(nwords, hidden=100):
    inp = Input((None, nwords))
    r1 = LSTM(hidden, return_sequences=True)(inp)
    #r2 = LSTM(hidden, return_sequences=True)(r1)
    probs = Dense(nwords, activation="softmax")(r1)
    model = Model(inp, probs)
    model.add_loss("cce", probs, name="CCE")
    model.compile("adagrad", learning_rate=0.01)
    return model
    
def generate_from_model(model, g, length, batch_size):
    #print("------- generate ----------")
    model.reset_state()
    nwords = g.NWords
    
    rows = []
    row = [random.randint(0, nwords-1) for _ in range(batch_size)]      # [w]
    rows.append(row)
    
    for t in range(length-1):
        x = np.array([g.vectorize(xi) for xi in row])
        y = model.compute(x[:,None,:])[0][:,0,:]            # y: [mb, w], t=0
        pvec = y**3
        pvec = pvec/np.sum(pvec, axis=-1, keepdims=True)        # -> [mb, w]
        
        row = [np.random.choice(nwords, p=p) for p in pvec]
        rows.append(row)
        
    rows = np.array(rows)           # [t,mb]
    return rows.transpose((1,0))
    
def generate_batch(g, length, batch_size):
    
    #print("generate_batch(%s, %s)..." % (length, batch_size))
    
    sequences = np.array([g.generate(length+1, as_vectors=True) for _ in range(batch_size)])
    
    #print("sequences:", sequences.shape)
    
    x = sequences[:,:-1,:]
    y_ = sequences[:,1:,:]
    
    return x, y_
    
    
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
        
        if iteration and iteration % 10 == 0:
            generated = generate_from_model(model, g, length, batch_size)[0]
            #print(type(generated), generated.shape, generated)
            
            valid_length = g.validate(generated)
            valid_ma += 0.1*(valid_length-valid_ma)
            if iteration % 100 == 0:
                print(generated[:valid_length], "*", generated[valid_length:], " valid length:", valid_length)
                print("Batches:", iteration, "  steps:", iteration*length*batch_size, "  loss/step:", losses["CCE"]/x.shape[1],
                   "  moving average:", valid_ma)
        if valid_ma >= goal:
            return episodes, steps, valid_ma
            
if __name__ == '__main__':
    nwords = 10
    length = 50
    distance = 5
    r = 2
    batch_size = 5
    g = Generator(nwords, distance, r)
    model = create_net(nwords)
    episodes, steps, valid_ma = train(model, g, length, batch_size, length*0.95)
    print("episodes=", episodes, "  steps=", steps, "   valid_length=", valid_ma)