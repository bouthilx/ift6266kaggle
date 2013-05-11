from pylearn2.models.mlp import Tanh, Linear, Softmax, MLP, Conv2DSpace, WeightDecay, RectifiedLinear
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.costs.cost import MethodCost, SumOfCosts
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.termination_criteria import MonitorBased
from pylearn2.train import Train
from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.datasets.preprocessing import Standardize, ShuffleAndSplit
from pylearn2.base import StackedBlocks
from pylearn2.costs.mlp.missing_target_cost import MissingTargetCost

from theano import tensor, function
import numpy as np
import sys

from fkd.PositionalConv import PositionalConv, KeypointMappingCost, MatchTargetSpace, FlatteningLayer, CrushedAxes, SoftArgmax
from fkd.FacialKeypointsMap import FacialKeypointMap
from keypoints_dataset import FacialKeypointDataset
from fkd.MonitorFilters import MonitorFilters, MonitorResults

import pylearn2.utils.serial as serial

import pylab
import os

base_path = "/data/lisatmp/ift6266h13/ContestDataset"
#base_path = "/home/xavier/data/ContestDataset"

def load_dataset(file_path,dclass,config,force=False):
    if file_path[-4:]!=".pkl":
        file_path += '.pkl'

    if not os.path.exists(file_path) or force:
        print "building dataset..."
        d = dclass(**config)
#        serial.save(file_path,d)
    else:
        d = serial.load(file_path)

    return d

output_shape = [96,96,1]
#output_shape = [100,100,1]
#output_shape = [104,104,1]
#output_shape = [111,111,1]

def load_predatasets(test):
    if test:
        stt,spt,stv,spv = 0, 160, 0, 160
    else:
        stt,spt,stv,spv = 0, 5760, 5761, 7041

    print test
    print stt,spt,stv,spv

    pretrainset = load_dataset('pretrainset',
                    FacialKeypointMap,
                    dict(
                        base_path = base_path,
                        which_set = 'train',
#                        start = 0,
#                        stop = 5760,
                        output_shape = output_shape,
                        axes = ['b','c', 0, 1],
#                        fit_preprocessor=True,
#                        fit_test_preprocessor=True,
#                        preprocessor=Standardize()),
                        preprocessor=ShuffleAndSplit(3235,stt,spt)),
                        force=True)

    prevalidset = load_dataset('prevalidset',
                    FacialKeypointMap,
                    dict(
                        base_path = base_path,
                        which_set = 'train',
#                        start = 5761,
#                        stop = 7041,
                        output_shape = output_shape,
                        axes = ['b','c', 0, 1],
#                        fit_preprocessor=True,
#                        fit_test_preprocessor=True,
#                        preprocessor=Standardize()),
                        preprocessor=ShuffleAndSplit(3235,stv,spv)),
                        force=True)

    return pretrainset,prevalidset

def load_traindatasets():

    trainset = load_dataset('trainset',
                    FacialKeypointDataset,
                    dict(
                        base_path = base_path,
                        which_set = 'train',
                        start = 0,
                        stop = 501,
                        output_shape = output_shape,
                        axes = ['b','c', 0, 1]))

    validset = load_dataset('validset',
                    FacialKeypointDataset,
                    dict(
                        base_path = base_path,
                        which_set = 'train',
                        start = 501,
                        stop = 1001,
                        output_shape = output_shape,
                        axes = ['b','c', 0, 1]))

    return trainset,validset

def load_testdataset():

    testset = load_dataset('testset',
                    FacialKeypointDataset,
                    dict(
                        base_path = base_path,
                        which_set = 'train',
                        start = 1001,
                        stop = 1501,
                        output_shape = output_shape,
                        axes = ['b','c', 0, 1]))

    return testset

def get_Tanh(name,structure,**args):
    n_input, n_output = structure
    config = {
        'layer_name':name,
        'dim':n_output,
        'sparse_init': 15
    }

    config.update(args)

    return Tanh(**config)

def get_Linear(name,structure,**args):
    n_input, n_output = structure
    config = {
        'layer_name':name,
        'dim':n_output,
    }

    config.update(args)

    return Linear(**config)

def get_PosConvLayer(name,**args):
    config = {
        'layer_name': name,
#        'output_channels': 16,
#        'output_channels': 32,
        'output_channels': 64,
        'irange': .05,
        'kernel_groups': 10,
#        'kernel_shape': [16, 16],
        'kernel_shape': [9, 9],
#        'kernel_shape': [5, 5],
        'pool_stride': [3, 3],
#        'max_kernel_norm': 1.9365,
        'border_mode': 'full',
#        'output_normalization': MatchTargetSpace((96,96),(100,100)),
    }
 
    config.update(args)

    return PositionalConv(**config)

def get_mlp(layers,nvis,**args):
    config = {
        'layers' : layers,
        'nvis' : nvis
    }

    config.update(args)

    return MLP(**config)

def get_Conv(layers,**args):
    config = {
        'batch_size': 32,
        'input_space': Conv2DSpace (
            shape = [96, 96],
            num_channels = 1,
            axes = ['b','c',0,1]),
        'layers' : layers
    }

    config.update(args)

    return MLP(**config)

def get_layer_trainer_sgd(model,lr,cost,trainset,validset,save_path):
    train_algo = SGD(
        batch_size = 32,
        learning_rate = lr,
        monitoring_dataset = {'train':trainset,
                              'valid':validset
        },
        cost = cost,
        termination_criterion = MonitorBased(
            channel_name = "valid_objective",
            N = 2)
    )

    extensions = [#MonitorBasedSaveBest(
#        channel_name = 'valid_objective',
#        save_path = save_path),
                 MonitorFilters(
        save_path = 'pretrainfilters_max_99_lr'+str(lr),
        param='W'),
                 MonitorResults(
        save_path = 'pretrainresults_max_99_lr'+str(lr))]

    return Train(model = model, algorithm = train_algo,
                 extensions=extensions, dataset = trainset)

class StackedMLPs(StackedBlocks):
    def __call__(self,inputs):
        repr = [inputs]

        for layer in self._layers:
            outputs = layer.fprop(repr[-1])
            repr.append(outputs)

        return repr

def build_conv(lr,pretrainset,prevalidset,trainset,validset):

    print "Building layers..."
    layers = []
    layers.append(get_PosConvLayer('h0'))
#    layers.append(FlatteningLayer('h1'))
#    layers.append(get_Tanh('h1',[np.prod(output_shape),96*96]))
#    layers.append(Softmax(max_col_norm= 1.9365,layer_name= 'h1',n_classes= 96*96, istdev= .05))
    layers.append(RectifiedLinear(layer_name='h1',dim= 96*96,irange=0.01))
#    layers.append(SoftArgmax('h2'))
    print "done"

    # construct layer trainers
    print "Building trainers..."
    layer_trainers = []
    layer_trainers.append(
        get_layer_trainer_sgd(
            get_Conv(layers),
            lr,
            SumOfCosts(costs=[
                KeypointMappingCost(),
#                WeightDecay(
#                    coeffs= [0.1,0.1]),
            ]),
            pretrainset,prevalidset,
            "pretrained.pkl"))

    return layer_trainers

def build_mlp(pretrainset,prevalidset,trainset,validset):

    design_matrix = trainset.get_design_matrix()
    pre_n_output = design_matrix.shape[1]
    n_input = design_matrix.shape[1]
    n_output = 30

    print "Building layers..."
    layers = []
    
    structure = [[n_input,1000],[1000,pre_n_output],
                 [pre_n_output,1000],[1000,n_output]]
    # layer 0 : 
    layers.append(get_Tanh('h0',structure[0]))
    # layer 1 :
    layers.append(get_Linear('h1',structure[1]))
    # layer 2 :
    layers.append(get_Tanh('h2',structure[2]))
    # layer 3 :
    layers.append(get_Linear('y',structure[3]))
    print "done"

    # construct training sets for different layers?
    # nop...

    # construct layer trainers
    print "Building trainers..."
    layer_trainers = []
    layer_trainers.append(
        get_layer_trainer_sgd(
            MLP(layers[:2],n_input),
            MethodCost(
                method = 'cost_from_X',
                supervised = 1
            ),
            pretrainset,prevalidset,
            "pretrained.pkl"))

    layer_trainers[-1].model._params = []

    layer_trainers.append(
        get_layer_trainer_sgd(
            MLP(layers[2:]),
            MissingTargetCost(),
            TransformerDataset( raw = trainset, 
                transformer = StackedMLPs([layer_trainers[-1].model]) ),
            TransformerDataset( raw = validset, 
                transformer = StackedMLPs([layer_trainers[-1].model]) ),
            "best_full.pkl"))

    print "done"

    return layer_trainers

def main():
    test = len(sys.argv)>1 and sys.argv[1]=='test'


    print "loading pretraining sets"
    pretrainset,prevalidset =  load_predatasets(test)
#    print "loading training sets"
#    trainset,validset = load_traindatasets()
#    print "loading test sets"
#    testset = load_testdatasets()

#    for lr in [1e-4,1e-5,1e-6,1e-7]:
    if test:
        layer_trainers = build_conv(1e-5,pretrainset,prevalidset,None,None)#trainset,validset)
        print '-----------------------------------'
        print ' Supervised pretraining'
        print '-----------------------------------'
        layer_trainers[0].main_loop()

        sys.exit(0)

    for lr in [1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]:
    #    layer_trainers = build_mlp(pretrainset,prevalidset,trainset,validset)
        try:
            layer_trainers = build_conv(lr,pretrainset,prevalidset,None,None)#trainset,validset)
    #    import theano
    #    theano.config.compute_test_value = 'warn'

            print '-----------------------------------'
            print ' Supervised pretraining'
            print '-----------------------------------'
            layer_trainers[0].main_loop()
        #    layer_trainers[0] = serial.load('pretrained.pkl')

#        X = model.get_input_space().make_theano_batch(name="%s[X]" % self.__class__.__name__)
#        self.f = function([X],model.fprop(X),on_unused_input='ignore')

#        dX = dataset.adjust_for_viewer(dataset.get_batch_topo(2))
#        y_hat = self.f(dX[0:1].astype('float32'))
#        y_hat = y_hat.reshape([1]+dataset.output_shape)
#        y = dataset.y[0:1].reshape([1]+dataset.output_shape)

#        pylab.imshow(y_hat[0],cmap="gray")
#        pylab.show()
        except BaseException as e:
            print e

    sys.exit(0)

    print '-----------------------------------'
    print ' Supervised training'
    print '-----------------------------------'
    layer_trainers[-1].main_loop()

#    premodel = serial.load('pretrained.pkl')
#    model = serial.load('best_full.pkl')
    premodel = layer_trainers[0].model
    model = layer_trainers[1].model

    premodel._params = []
    model._params = []

    X = tensor.matrix()
    y = model.fprop(X)

    f = function([X],y)

#    base_path = "/home/xavier/data/ContestDataset"
#    testset = FacialKeypointDataset(
#        base_path = base_path,
#        which_set = 'train',
#        start = 1001,
#        stop = 1500)

    T_testset = TransformerDataset( raw = testset, transformer = StackedMLPs([premodel]) )
    
    Xs = np.concatenate([batch for batch in T_testset.iterator(mode='sequential',batch_size=50,targets=False)])
    print Xs.shape
#    Xs = testset.X

    Y_hat = f(np.float32(Xs))
    Y = testset.y

#    Y_hat += 100.

    for i in range(10):
        print "Y:  ",Y[i]
        print "Yh: ",Y_hat[i]

    D = np.sqrt(np.sum(Y_hat**2 -2*Y_hat*Y + Y**2, axis=1))
    print D.shape
    print D.mean(0), D.std(0)

    stacked = StackedMLPs([premodel,model])
    serial.save("stacked.pkl", stacked, on_overwrite = 'backup')

if __name__ == '__main__':
    main()
