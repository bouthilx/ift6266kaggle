from theano.gof.op import get_debug_values
import theano.tensor as T
from theano import config
from theano import function

from pylearn2.models.mlp import ConvRectifiedLinear, Layer
from pylearn2.space import Conv2DSpace, VectorSpace
from pylearn2.linear import conv2d
from pylearn2.utils import sharedX

import numpy as np

import sys

from pylearn2.costs.cost import Cost

class AxeRatioCost(Cost):
    supervised = True

    def __call__(self, model, X, Y):

        batch_size = 32
        image_size = 96

        Y_hat = model.fprop(X)

        print "Warning: the size of the axe is set manually"
        Yx_hat = Y_hat[:,:image_size] 
        Yy_hat = Y_hat[:,image_size:]

        Yx = Y[:,:image_size]
        Yy = Y[:,image_size:]

        epsylon = 1e-10
 
        costMatrix = T.matrix()
        max_x = T.argmax(Yx,axis=1)
        max_y = T.argmax(Yy,axis=1)

        costMatrix = T.sqr(T.log((Yx+epsylon)/(Yx[range(batch_size),max_x]+epsylon)[:,None]) - T.log((Yx_hat+epsylon)/(Yx_hat[range(batch_size),max_x]+epsylon)[:,None]))
        costMatrix += T.sqr(T.log((Yy+epsylon)/(Yy[range(batch_size),max_y]+epsylon)[:,None]) - T.log((Yy_hat+epsylon)/(Yy_hat[range(batch_size),max_y]+epsylon)[:,None]))

        costMatrix *= T.neq(T.sum(Y,axis=1),0)[:,None]

        cost = costMatrix.sum(axis=1).mean()
        return cost

class MapRatioCost(Cost):
    supervised = True

    def __call__(self, model, X, Y):
        from theano import function

        batch_size = 32
        image_size = 96

        Y_hat = model.fprop(X)

        epsylon = 1e-10

        max_i = T.argmax(Y,axis=1)

        costMatrix = T.sqr(T.log((Y+epsylon)/(Y[range(batch_size),max_i]+epsylon)[:,None]) - T.log((Y_hat+epsylon)/(Y_hat[range(batch_size),max_i]+epsylon)[:,None]))

        costMatrix *= T.neq(T.sum(Y,axis=1),0)[:,None]
        cost = costMatrix.sum(axis=1).mean()
        return cost

class KeypointMappingCost(Cost):
    supervised = True

    def __call__(self, model, X, Y):
        Y_hat = model.fprop(X)

       
        costMatrix = T.sqr(Y-Y_hat)
        costMatrix *= T.neq(T.sum(Y,axis=1),0)[:,None]

        cost = costMatrix.sum(axis=1).mean()
        return cost

class FlatteningLayer(Layer):
    """
        Flatten a shaped output as matrices
    """
    def __init__(self,layer_name):
        self.layer_name = layer_name

        # no params
        self._params = []

    def get_weight_decay(self, coeff):
        return 0.

    def set_input_space(self,space):
        self.input_space = space
        self.output_space = VectorSpace(space.shape[0]*space.shape[1])
        print "linearized output : ",self.output_space.dim

    def fprop(self, state_below):
        return self.input_space.format_as(state_below,self.output_space)

    def cost(self,Y, Y_hat):
        return self.cost_from_cost_matrix(self.cost_matrix(Y,Y_hat))

    def cost_matrix(self, Y, Y_hat):
        return T.sqr(Y-Y_hat)

    def cost_from_cost_matrix(self,cost_matrix):
        return cost_matrix.sum(axis=1).mean()

class CrushedAxes(FlatteningLayer):
    """
    """
    def set_input_space(self,space):
        self.input_space = space
        self.output_space = VectorSpace(len(space.shape)*space.shape[0])
        print "output_space", self.output_space.dim

        pass

    def fprop(self, state_below):

        from theano import function
        return self.stack(state_below)

    def stack(self,state_below):
        return T.concatenate([T.sum(state_below,axis=2).flatten(2),
                              T.sum(state_below,axis=3).flatten(2)],axis=1)


    def cost_matrix(self, Y, Y_hat):
        return T.sqr(Y-Y_hat)

class SoftArgmax(CrushedAxes):
    """
        only works for data with 1 channel (no rgb)
    """
    def set_input_space(self,space):
        self.input_space = space

        self.needs_reformat = not isinstance(space, Conv2DSpace)

        if self.needs_reformat:
            space = Conv2DSpace(shape=[int(np.sqrt(space.dim))]*2,
                num_channels = 1,
                axes = ('b', 'c', 0, 1))
            print "reformat to",space.shape

        self.new_space = space

        self.output_space = VectorSpace(len(space.shape))

        print "output_space",self.output_space.dim

    def fprop(self, state_below):
        if self.needs_reformat:
            state_below = self.input_space.format_as(state_below,self.new_space)

        x = T.sum(state_below,axis=2).flatten(2)
        y = T.sum(state_below,axis=3).flatten(2)
        argmax_x = self._argmax(x*1000.0)
        argmax_y = self._argmax(y*1000.0)

        stack = T.concatenate([argmax_x[:,None],
                               argmax_y[:,None]],axis=1)
        return stack

    def _softmax(self,X):
        Z = T.exp(X)
        return Z/(T.sum(Z,axis=[1,2,3])[:,None,None,None])

    def _argmax(self,X):
        return T.sum(T.nnet.softmax(X)*np.arange(self.new_space.shape[0]).astype(config.floatX),axis=1)

class PositionalConv(ConvRectifiedLinear):

    def __init__(self,
                 output_channels,
                 kernel_groups,
                 kernel_shape,
                 pool_stride,
                 layer_name,
                 irange = None,
                 border_mode = 'valid',
                 sparse_init = None,
                 include_prob = 1.0,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 left_slope = 0.0,
                 max_kernel_norm = None,
                 pool_type = 'max',
                 detector_normalization = None,
                 output_normalization = None,
                 crop_border= False,
                 centered=False):

        ConvRectifiedLinear.__init__(self,
                 output_channels,
                 kernel_shape,
                 kernel_shape, #pool_shape,
                 pool_stride,
                 layer_name,
                 irange,
                 border_mode,
                 sparse_init,
                 include_prob,
                 init_bias,
                 W_lr_scale,
                 b_lr_scale,
                 left_slope,
                 max_kernel_norm,
                 pool_type,
                 detector_normalization,
                 output_normalization)

        self.kernel_groups = kernel_groups
        self.crop_border = crop_border
        self.centered = centered

    def set_input_space(self, space):
        """ Note: this resets parameters! """

        self.input_space = space
        rng = self.mlp.rng

        if self.border_mode == 'valid':
            output_shape = [self.input_space.shape[0] - self.kernel_shape[0] + 1,
                self.input_space.shape[1] - self.kernel_shape[1] + 1]
        elif self.border_mode == 'full':
            output_shape = [self.input_space.shape[0] + self.kernel_shape[0] - 1,
                    self.input_space.shape[1] + self.kernel_shape[1] - 1]

        self.detector_space = Conv2DSpace(shape=output_shape,
                num_channels = self.output_channels,
                axes = ('b', 'c', 0, 1))

        if self.irange is not None:
            assert self.sparse_init is None
            self.transformer = conv2d.make_random_conv2D(
                    irange = self.irange,
                    input_space = self.input_space,
                    output_space = self.detector_space,
                    kernel_shape = self.kernel_shape,
                    batch_size = self.mlp.batch_size,
                    subsample = (1,1),
                    border_mode = self.border_mode,
                    rng = rng)
        elif self.sparse_init is not None:
            self.transformer = conv2d.make_sparse_random_conv2D(
                    num_nonzero = self.sparse_init,
                    input_space = self.input_space,
                    output_space = self.detector_space,
                    kernel_shape = self.kernel_shape,
                    batch_size = self.mlp.batch_size,
                    subsample = (1,1),
                    border_mode = self.border_mode,
                    rng = rng)
        W, = self.transformer.get_params()
        W.name = 'W'

        self.b = sharedX(self.detector_space.get_origin() + self.init_bias)
        self.b.name = 'b'

        print 'Input shape: ', self.input_space.shape
        print 'Detector space: ', self.detector_space.shape

        if self.mlp.batch_size is None:
            raise ValueError("Tried to use a convolutional layer with an MLP that has "
                    "no batch size specified. You must specify the batch size of the "
                    "model because theano requires the batch size to be known at "
                    "graph construction time for convolution.")

        assert self.pool_type in ['max', 'mean']

        dummy_detector = sharedX(self.detector_space.get_origin_batch(self.mlp.batch_size))
        if self.pool_type == 'max':
            dummy_p = max_pool(bc01=dummy_detector, pool_shape=self.pool_shape,
                    pool_stride=self.pool_stride,
                    image_shape=self.detector_space.shape,
                    output_channels = self.output_channels)
        elif self.pool_type == 'mean':
            dummy_p = mean_pool(bc01=dummy_detector, pool_shape=self.pool_shape,
                    pool_stride=self.pool_stride,
                    image_shape=self.detector_space.shape)
        dummy_p = dummy_p.eval()
        self.tmp_output_space = Conv2DSpace(shape=[dummy_p.shape[2], dummy_p.shape[3]],
                num_channels = 1,#self.output_channels, 
                axes = ('b', 'c', 0, 1) )
   
        if self.crop_border:
            self.output_space = self.input_space
        else:
            self.output_space = self.tmp_output_space

        print 'Output space: ', self.output_space.shape

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        z = self.transformer.lmul(state_below) + self.b
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        d = z * (z > 0.) + self.left_slope * z * (z < 0.)

        self.detector_space.validate(d)

        if not hasattr(self, 'detector_normalization'):
            self.detector_normalization = None

        if self.detector_normalization:
            d = self.detector_normalization(d)

        assert self.pool_type in ['max', 'mean']
        if self.pool_type == 'max':
            p = max_pool(bc01=d, pool_shape=self.pool_shape,
                    pool_stride=self.pool_stride,
                    image_shape=self.detector_space.shape,
                    output_channels = self.output_channels)
        elif self.pool_type == 'mean':
            p = mean_pool(bc01=d, pool_shape=self.pool_shape,
                    pool_stride=self.pool_stride,
                    image_shape=self.detector_space.shape)

        self.output_space.validate(p)

        if not hasattr(self, 'output_normalization'):
            self.output_normalization = None

        if self.output_normalization:
            p = self.output_normalization(p)
       
        if self.crop_border:
            if self.centered:
                return self._centered(p,self.tmp_output_space.shape,self.input_space.shape)

            return p[:,:,:self.input_space.shape[0],:self.input_space.shape[1]]
        else:
            return p

    def _centered(self,X,X_shape,new_shape):
        min_dx = (X_shape[0] - new_shape[0])/2.0
        max_dx = int(min_dx +0.5)
        min_dx = int(min_dx)
        min_dy = (X_shape[1] - new_shape[1])/2.0
        max_dy = int(min_dy +0.5)
        min_dy = int(min_dy)

        return X[:,:,min_dx:-max_dx,min_dy:-max_dy]
        
    def cost_matrix(self, Y, Y_hat):
        return T.sqr(Y-Y_hat)

    def cost_from_cost_matrix(self,cost_matrix):
        return cost_matrix.sum(axis=1).mean()

    def cost(self,Y,Y_hat):
        print "Oh!"
        print Y.dtype
        print Y_hat.dtype
        s = T.neq(Y_hat,Y).sum()
        return s
        
def max_pool(bc01, pool_shape, pool_stride, image_shape, output_channels):
    """
    Theano's max pooling op only support pool_stride = pool_shape
    so here we have a graph that does max pooling with strides

        ****** to change
    bc01: minibatch in format (batch size, channels, rows, cols)
    pool_shape: shape of the pool region (rows, cols)
    pool_stride: strides between pooling regions (row stride, col stride)
    image_shape: avoid doing some of the arithmetic in theano
    """
    
    mx = None
    for feature_map in xrange(output_channels):
#        print feature_map
        if mx is None:
            mx = bc01[:,feature_map]
        else:
            mx = T.maximum(mx,bc01[:,feature_map])

    mx = mx.reshape((bc01.shape[0],1,image_shape[0],image_shape[1]))
    return mx

def mean_pool(bc01, pool_shape, pool_stride, image_shape):
    return bc01.mean(1).reshape((bc01.shape[0],1,image_shape[0],image_shape[1]))
