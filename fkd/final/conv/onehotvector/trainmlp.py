from pylearn2.models.mlp import Tanh, Linear, Softmax, MLP, Conv2DSpace, WeightDecay, RectifiedLinear
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.costs.cost import MethodCost, SumOfCosts
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.termination_criteria import MonitorBased
from pylearn2.train import Train
from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.datasets.preprocessing import Standardize, ShuffleAndSplit, Pipeline, ZCA, GlobalContrastNormalization

from pylearn2.base import StackedBlocks
from pylearn2.costs.mlp.missing_target_cost import MissingTargetCost

from theano import tensor, function
import numpy as np
import sys

from fkd.PositionalConv import PositionalConv, KeypointMappingCost, MatchTargetSpace, FlatteningLayer, CrushedAxes, SoftArgmax, MapRatioCost, AxeRatioCost
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

def load_predatasets(test,preprocess=None):
    if test:
        stt,spt,stv,spv = 0, 160, 0, 160
    else:
        stt,spt,stv,spv = 0, 5760, 5761, 7041

    print test
    print stt,spt,stv,spv

    if preprocess:
        items = [preprocess]
    else:
        items = []

    pretrainset = load_dataset('pretrainset',
                    FacialKeypointMap,
                    dict(
                        base_path = base_path,
                        which_set = 'train',
                        axes = ['b','c', 0, 1],
                        fit_preprocessor=True,
                        fit_test_preprocessor=True,
                        preprocessor= Pipeline(items=items+[ShuffleAndSplit(3235,stt,spt)]),
                        mode='axes',
                        keypoint=2),
                        force=True)

    prevalidset = load_dataset('prevalidset',
                    FacialKeypointMap,
                    dict(
                        base_path = base_path,
                        which_set = 'train',
                        axes = ['b','c', 0, 1],
                        fit_preprocessor=True,
                        fit_test_preprocessor=True,
                        preprocessor= Pipeline(items=items+[ShuffleAndSplit(3235,stv,spv)]),
                        mode='axes',
                        keypoint=2),
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
        'output_channels': 64,
        'irange': .05,
        'kernel_groups': 10,
        'kernel_shape': [9, 9],
        'pool_stride': [3, 3],
        'pool_type': 'mean',
        'border_mode': 'full',
        'crop_border': True,
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

def get_layer_trainer_sgd(model,lr,cost,trainset,validset,save_path,preprocess=""):
    train_algo = SGD(
        batch_size = 32,
        learning_rate = lr,
        monitoring_dataset = {'train':trainset,
                              'valid':validset
        },
        cost = cost,
        termination_criterion = MonitorBased(
            channel_name = "valid_objective",
            N = 6)
    )

    extensions = [#MonitorBasedSaveBest(
#        channel_name = 'valid_objective',
#        save_path = save_path),
                 MonitorFilters(
        save_path = 'pretrainfilters_max_1_551616_lr'+str(lr)+"_"+preprocess,
        param='W',layer=0),
                   MonitorFilters(
        save_path = 'pretrainfilters_max_2_551616_lr'+str(lr)+"_"+preprocess,
        param='W',layer=1),
                 MonitorResults(
        mode='full',map_layer=0,#axes_layer=-1,
        save_path = 'pretrainresults_max_1_551616_lr'+str(lr)+"_"+preprocess),
                 MonitorResults(
        mode='axes',map_layer=1,axes_layer=-1,
        save_path = 'pretrainresults_max_2_551616_lr'+str(lr)+"_"+preprocess)]


    return Train(model = model, algorithm = train_algo,
                 extensions=extensions, dataset = trainset)

class StackedMLPs(StackedBlocks):
    def __call__(self,inputs):
        repr = [inputs]

        for layer in self._layers:
            outputs = layer.fprop(repr[-1])
            repr.append(outputs)

        return repr

def build_conv(lr,pretrainset,prevalidset,trainset,validset,preprocess):

    print "Building layers..."
    layers = []
    layers.append(get_PosConvLayer('h0',kernel_shape= [5, 5],output_channels=32,crop_border=True,centered=True))
    layers.append(get_PosConvLayer('h1',kernel_shape= [16, 16],output_channels=16,crop_border=True,centered=True))
    layers.append(CrushedAxes('h2'))
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
#                MapRatioCost(),
#                AxeRatioCost(),
#                WeightDecay(
#                    coeffs= [0.1,0.1]),
            ]),
            pretrainset,prevalidset,
            "pretrained.pkl",
            preprocess))

    return layer_trainers

def main():
    test = len(sys.argv)>1 and sys.argv[1]=='test'


    print "loading pretraining sets"
    pretrainset,prevalidset =  load_predatasets(test,GlobalContrastNormalization())

    if test:
        layer_trainers = build_conv(1e-4,pretrainset,prevalidset,None,None,'')
        print '-----------------------------------'
        print ' Supervised pretraining'
        print '-----------------------------------'
        layer_trainers[0].main_loop()

        sys.exit(0)

    for lr in [1e-1,1e-3,1e-5,1e-7,1e-9]:
        try:
            layer_trainers = build_conv(lr,pretrainset,prevalidset,None,None,"gcnF")

            print '-----------------------------------'
            print ' Supervised pretraining'
            print '-----------------------------------'
            layer_trainers[0].main_loop()
        except BaseException as e:
            print e

    sys.exit(0)

if __name__ == '__main__':
    main()
