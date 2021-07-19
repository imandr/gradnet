import numpy as np
from cconv import convolve, pool, pool_back
import math, time
from .. import Layer
from ..util import make_list
from numpy.random import default_rng

rng = default_rng()

def convolve_xw(inp, w, mode):
    # inp: (nb, nx, ny, nc_in)
    # w: (nx, ny, nc_in, nc_out)
    # returns (nb, x, y, nc_out)
    
    mode = 0 if mode == 'valid' else 1
    
    inp = inp.transpose((0,3,1,2))
    w = w.transpose((3,2,0,1))
    return convolve(inp, w, mode).transpose((0,2,3,1))
    
def convolve_xy(x, y):
    # x: (nb, nx, ny, nc_in)
    # y: (nb, mx, my, nc_out)       (mx,my) < (nx,ny)
    # returns (fx, fy, nc_in, nc_out)
    
    x = x.transpose((3,0,1,2))
    y = y.transpose((3,0,1,2))
    return convolve(x, y, 0).transpose((2,3,0,1))
    
class Conv2D(Layer):
    def __init__(self, filter_x, filter_y, out_channels, **args):
        # filter_xy_shape is (fh, fw) 
        Layer.__init__(self, **args)
        #print self.InShape
        self.filter_x = filter_x
        self.filter_y = filter_y
        self.out_channels = out_channels
        self.filter_shape = None                # will be initialized when configured
        self.W = self.b = None

    def configure(self, inputs):        
        assert len(inputs) == 1
        inp = inputs[0]
        in_shape = inp.Shape
        assert len(in_shape) == 3
        self.in_channels = in_shape[-1]
        self.filter_shape = (self.filter_x, self.filter_y, self.in_channels, self.out_channels)

        nin = np.prod(self.filter_shape[:3])   # x*y*in_channels
        self.W = np.asarray(rng.normal(size=self.filter_shape, scale=1.0/math.sqrt(nin)), dtype=np.float32)
        self.b = np.zeros(self.out_channels, dtype=np.float32)
        return (in_shape[0]-self.filter_x+1, in_shape[1]-self.filter_y+1, self.out_channels)
        
    def check_configuration(self, inputs):
        assert len(inputs) == 1
        inp = inputs[0]
        in_shape = inp.Shape
        assert len(in_shape) == 3
        assert shape[-1] == self.filter_shape[-2]
        return (in_shape[0]-self.filter_x+1, in_shape[1]-self.filter_y+1, self.out_channels)

    def compute(self, xs, in_state):
        y = convolve_xw(xs[0], self.W, 'valid') + self.b
        return y, None, None

    def grads(self, gY, _, xs, y, context):
        assert len(xs) == 1
        x = xs[0]
        #print "conv.bprop"
        n_imgs = x.shape[0]

        #print x.shape, gY.shape

        gW = convolve_xy(x, gY)
        gb = np.sum(gY, axis=(0, 1, 2))

        w_flip = self.W[::-1,::-1,:,:]
        w_flip = np.transpose(w_flip, (0,1,3,2))
        gx = convolve_xw(gY, w_flip, 'full')
        return [gx], [gW, gb], None
        
    @property
    def params(self):
        return [self.W, self.b]
        
    def _set_weights(self, weights):
        w, b = weights
        #print("Dense._set_weights: w,b:", w.shape, b.shape)
        assert w.shape == self.W.shape
        assert b.shape == self.b.shape
        self.W = w
        self.b = b 
        
    def get_weights(self):
        return [self.W, self.b]

class Pool(Layer):
    
    params = []
    
    def __init__(self, pool_w, pool_h, mode='max', **args):
        Layer.__init__(self, **args)
        self.Mode = mode
        self.pool_w, self.pool_h = pool_w, pool_h
        
    def configure(self, inputs):
        assert isinstance(inputs, list) and len(inputs) == 1
        inp = inputs[0]
        input_shape = inp.Shape
        assert len(input_shape) == 3
        return self.output_shape(input_shape)

    check_config = configure

    def compute(self, xs, _):
        #print "x:", self.X.dtype, self.X
        assert isinstance(xs, list) and len(xs) == 1
        x = xs[0]
        y, pool_index = pool(x, self.pool_h, self.pool_w)
        return y, None, pool_index

    def grads(self, gy, _, xs, y, context):
        x = xs[0]
        pool_index = context
        gx = pool_back(gy, pool_index, self.pool_h, self.pool_w, 
            x.shape[1], x.shape[2])
        return [gx], [], None

    def output_shape(self, input_shape):
        return ((input_shape[0]+self.pool_h-1)//self.pool_h,
                 (input_shape[1]+self.pool_w-1)//self.pool_w,
                 input_shape[2]
                 )

