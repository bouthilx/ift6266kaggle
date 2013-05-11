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

w_map = {'W':'W','b':'bhid'}

test = 'test' in sys.argv

save_path = "pretrain_%s" % '_'.join([str(hid) for hid in configs['hid']])

if test:          
    save_path += "_test"                                           

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

if test:
    training_epochs = 10
else:
    training_epochs = 1000

#######################
## train autoencoder ##
#######################

print "\n\n\n"
print "#############"
print "## configs ##"
print "#############"
print configs

#models = []
#print "load pretrained layers"
#for i in range(3):
#    models.append(serial.load('layer%d.pkl' % i))

print "\n\n\n"
print "##################"
print "## training mlp ##"
print "##################"

numpy_rng  = np.random.RandomState(2355)

testset = FacialKeypointDataset(
                base_path = datadir,
                which_set = 'public_test',
                fit_preprocessor=True,
                fit_test_preprocessor=True,
                preprocessor= Pipeline(items=preprocess))


strain = trainset
valid  = validset

model = mlp.MLP( numpy_rng, strain.X.shape[1], configs['hid'], 0.0, 0.0)

print "\n\n\n"
print "###############################"
print "## loading pretrained layers ##"
print "###############################"

for i, param in enumerate(model.params[:-2]): # except output layer
    print i/2,param.name,
    value = np.load(save_path+"/layer%d%s.npy" % (i/2,w_map[param.name[0]])) 
    print value.shape
    param.set_value(value)

sgd.train(strain.X,valid.X,model,
          train_labels=strain.y,valid_labels=valid.y,
          learning_rate=0.001,
          batch_size=configs['batch_size'],
          epochs=epochs,wait_for=20,
          epsylon=0.0001,
          aug=1.01)

X = T.matrix()
f = theano.function([X],model.fprop(X))
Y_hat = f(strain.X[:10])
Y = strain.y[:10]

for y_hat,y in zip(Y_hat,Y):
    print y_hat[:2], y[:2]

y = np.vstack(sgd.iterate([testset.X],[f],configs['batch_size'])[0])

print "results shape"
print y.shape

out_path = save_path+"/test.csv"

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
