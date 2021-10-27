from gradnet import Model
from gradnet.layers import LSTM, Input

def create_model(nodes, hidden):
    inp = Input((None, nodes*2+1))
    lstm_1 = LSTM(hidden, return_sequences=True)(inp)
    lstm_2 = LSTM(nodes, return_sequences=False)(lstm_1)
    
    return Model(inp, lstm_2)