import numpy as np
import theano.tensor as T
import theano

from PositionalConv import PositionalConv
from pylearn2.space import Conv2DSpace, VectorSpace
from pylearn2.models.mlp import ConvRectifiedLinear

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function thanh or the
    sigmoid function (defined here by a ``SigmoidalLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, np_rng, numvis, numhid, L1=0.0, L2=0.0):

        self.numvis = numvis
        self.numhid = numhid

        self.L1 = L1
        self.L2 = L2

        self.Ws = []
        self.bs = []
        for i,h in enumerate(zip([numvis]+numhid,numhid)):
            h_in,h_out = h
            b = 4*np.sqrt(6./(h_in+h_out))
            W_init = np.asarray( 
                         np_rng.uniform( low=-b, high=b, 
                                            size=(h_in, h_out)), dtype=theano.config.floatX)
            self.Ws.append(theano.shared(value = W_init, name ='W_%d' % i))
           
            self.bs.append(theano.shared(value=np.zeros(h_out, dtype=theano.config.floatX), name ='b_%d' % i))
            print i, h_in,h_out

        self.params = []
        for W,b in zip(self.Ws,self.bs):
            self.params.append(W)
            self.params.append(b)

    def set_params(self,params):
        for oldparam, newparam in zip(self.params,params):
            oldparam.set_value(newparam.get_value())

    def fprop(self,X):
        for W,b in zip(self.Ws,self.bs)[:-1]:
            X = self.rl(T.dot(X,W)+b)
            
        W = self.Ws[-1]
        b = self.bs[-1]
        return T.dot(X,W)+b

    def rl(self,X):
        """
            Rectified linear
        """
        return T.tanh(X)#X*(X>0.)

    def cost(self,X,Y):
        Y_hat = self.fprop(X)
        costMatrix = (Y_hat-Y)*(Y_hat-Y)
        costMatrix *= T.neq(Y,-1)
        costMatrix = T.sqrt(T.sum(costMatrix,axis=1))
        L1 = sum([abs(W).sum() for W in self.Ws])

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        L2_sqr = sum([(W ** 2).sum() for W in self.Ws])

        return T.mean(costMatrix) + self.L1*L1# + self.L2*L2_sqr

    def errors(self,X,Y):
        return self.cost(X,Y)

class ConvMLP(object):
    def __init__(self, np_rng, shape, kernels, numhid, batch_size, L1=0.0, L2=0.0):

        self.convLayers = []
        for kernel in kernels:
            self.convLayers.append(ConvRectifiedLinear(#PositionalConv(
                output_channels = 64,
                irange = .05,
                kernel_shape = kernel,
                pool_shape = [4,4],
                pool_stride = [1,1],
                pool_type = 'max',
                layer_name = 'h0',
                border_mode = 'full',
                max_kernel_norm = 1.9365
            ))
            self.convLayers[-1].mlp = self

        self.rng = np_rng # not theano??
        self.batch_size = batch_size

        self.input_space = Conv2DSpace(shape,num_channels=1)
        for convLayer in self.convLayers:
            input_space = Conv2DSpace(shape,num_channels=1)
            convLayer.set_input_space(input_space)
            shape = convLayer.output_space.shape

        self.conv_output_space = VectorSpace(np.prod(self.convLayers[-1].output_space.shape))

        self.numvis = np.prod(self.convLayers[-1].output_space.shape)
        self.numhid = numhid

        self.L1 = L1
        self.L2 = L2

        self.Ws = []
        self.bs = []
        for i,h in enumerate(zip([self.numvis]+numhid,numhid)):
            h_in,h_out = h
            b = 4*np.sqrt(6./(h_in+h_out))
            W_init = np.asarray( 
                         np_rng.uniform( low=-b, high=b, 
                                            size=(h_in, h_out)), dtype=theano.config.floatX)
            self.Ws.append(theano.shared(value = W_init, name ='W_%d' % i))
           
            self.bs.append(theano.shared(value=np.zeros(h_out, dtype=theano.config.floatX), name ='b_%d' % i))
            print i, h_in,h_out

        self.params = []
        for convLayer in self.convLayers:
            for param in convLayer.get_params():
                self.params.append(param)
        for W,b in zip(self.Ws,self.bs):
            self.params.append(W)
            self.params.append(b)

    def set_params(self,params):
        for oldparam, newparam in zip(self.params,params):
            oldparam.set_value(newparam.get_value())

    def fprop(self,X):
        for convLayer in self.convLayers:
            X = convLayer.fprop(X)

        X = self.convLayers[-1].output_space.format_as(X,self.conv_output_space) # flatten images

        for W,b in zip(self.Ws,self.bs)[:-1]:
            X = self.rl(T.dot(X,W)+b)
            
        W = self.Ws[-1]
        b = self.bs[-1]
        return T.dot(X,W)+b

    def rl(self,X):
        """
            Rectified linear
        """
        return T.tanh(X)#X*(X>0.)

    def cost(self,X,Y):
        Y_hat = self.fprop(X)
        costMatrix = (Y_hat-Y)*(Y_hat-Y)
        costMatrix *= T.neq(Y,-1)
        costMatrix = T.sqrt(T.sum(costMatrix,axis=1))
        L1 = sum([abs(W).sum() for W in self.Ws])

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        L2_sqr = sum([(W ** 2).sum() for W in self.Ws])

        return T.mean(costMatrix)# + self.L1*L1 + self.L2*L2_sqr

    def errors(self,X,Y):
        return self.cost(X,Y)

if __name__=="__main__":

    import theano.tensor as T
    import train as sgd
    import sys
    import csv
    datadir = "/data/ContestDataset"

    numpy_rng  = np.random.RandomState(2355)

    from pylearn2.datasets.preprocessing import ShuffleAndSplit, Pipeline, GlobalContrastNormalization
    from keypoints_dataset import FacialKeypointDataset

    test = 'test' in sys.argv

    if test:
        stt,spt,stv,spv = 0, 160, 161, 321
    else:
        stt,spt,stv,spv = 0, 5760, 5761, 7041

    preprocess = [GlobalContrastNormalization()]

    from contestTransformerDatasetWithLabels import TransformerDataset
    from transformerWithLabels import TransformationPipeline, Scaling, Translation, Rotation, Flipping

    transformer = TransformationPipeline(
            seed=2355,
            input_space = Conv2DSpace (
                shape= [96, 96],
                num_channels= 1),
            transformations = [
#                Scaling(p=0.8,
#                    fct_settings = dict(
#                        loc=1.0,scale=0.1)),
                Translation(p=0.8,
                    fct_settings = dict(
                        loc=0.0,scale=0.8)),
#                Rotation(p=1.0,
#                    fct_settings = dict(
#                        loc=0.0,scale=10.0)),
                Flipping(p=0.33)])

    trainset = FacialKeypointDataset(
                    base_path = datadir,
                    which_set = 'train',
                    fit_preprocessor=True,
                    fit_test_preprocessor=True,
                    preprocessor= Pipeline(items=preprocess+[ShuffleAndSplit(3235,stt,spt)]))

    validset = FacialKeypointDataset(
                    base_path = datadir,
                    which_set = 'train',
                    fit_preprocessor=True,
                    fit_test_preprocessor=True,
                    preprocessor= Pipeline(items=preprocess+[ShuffleAndSplit(3235,stv,spv)]))

    testset = FacialKeypointDataset(
                    base_path = datadir,
                    which_set = 'public_test',
                    fit_preprocessor=True,
                    fit_test_preprocessor=True,
                    preprocessor= Pipeline(items=preprocess))

    
    strain = trainset
    valid  = validset

    batch_size = 64*2
    print strain.X.shape[1]
#    model = MLP( numpy_rng, strain.X.shape[1], [5000,300,30], 0.0, 0.0)
#    model = MLP( numpy_rng, strain.X.shape[1], [96*96*2,96*96,96*96/5,96*96/5,30], 0.0, 0.0)
#    model = MLP( numpy_rng, strain.X.shape[1], [96*96*2,96*96,96*96/5,96*96/30,30], 0.0, 0.0)
#    model = MLP( numpy_rng, strain.X.shape[1], [96*96*4,96*96*3/2,96*96*2/3,96*96/5,96*96/30,30])
#    model = MLP( numpy_rng, strain.X.shape[1], [96*96*4,96*96,96*96*3/2,96*96*2/3,96*96/5,96*96/30,30])
#    model = MLP( numpy_rng, strain.X.shape[1], [96*96*2,96*96*3/2,96*96,96*96*2/3,96*96/5,96*96/30,30])

    model = ConvMLP( numpy_rng, [96,96], [[5,5],[5,5]], [104*104,104*104/5,104*104/30,30], batch_size=batch_size)
    conv = True
    out_path = "mlp_%s.csv" % "_".join([str(hid) for hid in model.numhid])
    if conv:
        out_path = "conv_"+out_path
    print "will save in \"%s\"" % out_path

    if conv:
        strain.X = strain.X.reshape((strain.X.shape[0],96,96,1))
        valid.X = valid.X.reshape((valid.X.shape[0],96,96,1))

    if conv:
        x = T.tensor4() # because x is tensor4 for conv
        y = T.matrix()
    else:
        x = T.matrix()
        y = T.matrix()
    sgd.train(strain.X,valid.X,model,
              train_labels=strain.y,valid_labels=valid.y,
              learning_rate=0.01,
              batch_size=batch_size,
              epochs=5000,wait_for=20,epsylon=0.0001,
              x = x,
              y = y,
              aug=1.01)#,
#              transformer=transformer)

    if conv:
        X = T.tensor4()
    else:
        X = T.matrix()
    f = theano.function([X],model.fprop(X))
    if conv:
        strain.X[:10] = strain.X[:10].reshape((10,96,96,1))
    Y_hat = f(strain.X[:10])
    Y = strain.y[:10]

    for y_hat,y in zip(Y_hat,Y):
        print y_hat[:2], y[:2]

    y = []
    for x in testset.X:
        if conv:
            x = x.reshape((1,96,96,1))
        else:
            x = x.reshape((1,96*96))
        y.append(f(x))

    y = np.concatenate(y)
    print "results shape"
    print y.shape

    submission = []
    with open('submissionFileFormat.csv', 'rb') as cvsTemplate:
        reader = csv.reader(cvsTemplate)           
        for row in reader:
            submission.append(row)                 

    mapping = dict(zip(['left_eye_center_x',       
                        'left_eye_center_y',       
                        'right_eye_center_x',      
                        'right_eye_center_y',       
                        'left_eye_inner_corner_x',  
                        'left_eye_inner_corner_y',  
                        'left_eye_outer_corner_x',  
                        'left_eye_outer_corner_y',
                        'right_eye_inner_corner_x',
                        'right_eye_inner_corner_y',
                        'right_eye_outer_corner_x',
                        'right_eye_outer_corner_y',
                        'left_eyebrow_inner_end_x',
                        'left_eyebrow_inner_end_y',
                        'left_eyebrow_outer_end_x',
                        'left_eyebrow_outer_end_y', 
                        'right_eyebrow_inner_end_x',
                        'right_eyebrow_inner_end_y',
                        'right_eyebrow_outer_end_x',
                        'right_eyebrow_outer_end_y',
                        'nose_tip_x',
                        'nose_tip_y',        
                        'mouth_left_corner_x',
                        'mouth_left_corner_y',
                        'mouth_right_corner_x',
                        'mouth_right_corner_y',
                        'mouth_center_top_lip_x',
                        'mouth_center_top_lip_y',
                        'mouth_center_bottom_lip_x',
                        'mouth_center_bottom_lip_y'], range(30)))

    for row in submission[1:]:
        imgIdx = int(row[1]) - 1
        keypointName = row[2]
        keyPointIndex = mapping[keypointName]
        row.append(y[imgIdx, keyPointIndex])

    with open(out_path, 'w') as cvsTemplate:
        writer = csv.writer(cvsTemplate)
        for row in submission:
            writer.writerow(row)
