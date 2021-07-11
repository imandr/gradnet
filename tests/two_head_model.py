from gradnet import Input, Model
from gradnet.layers import Dense
from gradnet.activations import get_activation
from gradnet.optimizers import get_optimizer
from gradnet.losses import get_loss

import numpy as np

nin = 5
hidden = 7
nout_1 = 2
nout_2 = 3

mb=11

#
# create model
#

inp = Input((nin,))
hidden = Dense(hidden, name="hidden", activation="relu")(inp)
out1 = Dense(nout_1, name="out_1", activation="softmax")(hidden)
out2 = Dense(nout_2, name="out_2")(hidden)

model = Model([inp], [out1, out2])
model.add_loss(get_loss("cce")(out1))

for t in (0,1):
    x = np.random.random((mb, nin))
    y_ = np.random.random((mb, nout_1))
    model.reset_losses()
    y1, y2 = model.call([x])
    print(y1, y2)
    print(model.backprop(y_))
    
