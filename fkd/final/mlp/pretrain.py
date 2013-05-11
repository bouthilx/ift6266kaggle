import os
import sys
import gc
HOME = os.environ['HOME']

from timer import Timer

import csv
import time
import numpy as np
import numpy.random
import pylab
from scipy import ndimage

import cA
import mlp

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import train as sgd

from pylearn2.datasets.preprocessing import Standardize, ShuffleAndSplit, Pipeline, ZCA, GlobalContrastNormalization
import pylearn2.utils.serial as serial
from keypoints_dataset import FacialKeypointDataset


##############
## settings ##
##############

if len(sys.argv)<2:
    print "Must define layer depth"
    sys.exit(0)

i = int(sys.argv[1])

test = 'test' in sys.argv

# first experiment

configs = {
    "dataset":'keypoints',
    "hid":[10000,1000,300,30],
    "contract":[0.001,0.001,0.0001],
    "lr":[0.00001,0.00001,0.00001],
    "epsylon":[0.01,0.01,0.001],
    "batch_size":64*2
}

# second experiment

configs = {
    "dataset":'keypoints',
    "hid":[96*96*4,96*96,96*96*3/2,96*96*2/3,96*96/5,96*96/30,30],
    "contract":[0.1,0.1,0.01,0.01,0.001,0.0001],
    "lr":[0.01,0.01,0.005,0.001,0.0005,0.0001],
    "epsylon":[0.01,0.01,0.001,0.001,0.0001,0.0001],
    "batch_size":64*2
}

save_path = "pretrain_%s" % '_'.join([str(hid) for hid in configs['hid']])

if test:
    save_path += "_test"

if not os.path.exists(save_path):
    os.mkdir(save_path)

if test:
    configs['dataset'] += "_test"

datadir = "/data/ContestDataset"
#datadir = "/data/lisatmp/ift6266h13/ContestDataset"


###############
## load data ##
###############

if test:
    stt,spt,stv,spv = 0, 160, 0, 160
else:
    stt,spt,stv,spv = 0, 5760, 5761, 7041

preprocess = [GlobalContrastNormalization()]
#preprocess = []

if i==0:
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

    trtrainset = trainset.X
    trvalidset = validset.X

else:
    trtrainset = np.load(save_path+"/trainset%d.npy" % (i-1)).astype(theano.config.floatX)
    trvalidset = np.load(save_path+"/validset%d.npy" % (i-1)).astype(theano.config.floatX)

if test:
    training_epochs = 10
else:
    training_epochs = 200

#######################
## train autoencoder ##
#######################

print "\n\n\n"
print "#################################"
print "## pretraining in auto-encoder ##"
print "#################################"

print "\n\n\n"
print "#############"
print "## configs ##"
print "#############"
print configs

print "instantiating and training model"

numpy_rng  = np.random.RandomState(2355)
theano_rng = RandomStreams(2355)

models = []
h_in, h_out = zip([trtrainset.shape[1]]+configs['hid'],configs['hid'])[i]
print h_in,h_out

model = cA.cA(numpy_rng=numpy_rng, theano_rng=theano_rng, 
              numvis=h_in, numhid=h_out, 
              activation=T.tanh,
              vistype="real", contraction=configs['contract'][i])

sgd.train(trtrainset,trvalidset,model,
          batch_size=configs['batch_size'],
          wait_for=20,
          learning_rate=configs['lr'][i],
          epochs=training_epochs,
          epsylon=configs['epsylon'][i],
          aug=1.01)

X = T.matrix()
encoding = model.hiddens(X)
f = theano.function([X],encoding)
trtrainset = np.vstack(sgd.iterate([trtrainset],[f],configs['batch_size'])[0])
trvalidset = np.vstack(sgd.iterate([trvalidset],[f],configs['batch_size'])[0])
f = None

print "save params"
for param in model.params[:2]:
    print param.name
    np.save(save_path+"/layer%d%s.npy" % (i,param.name),param.get_value())
np.save(save_path+"/trainset%d.npy" % i,trtrainset)
np.save(save_path+"/validset%d.npy" % i,trvalidset)
