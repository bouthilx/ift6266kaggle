import numpy as np
import csv
import pylab as pl
from scipy import ndimage
import scipy
import sys

from pylearn2.gui import patch_viewer

NB = 36

reader = csv.reader(open("/home/xavier/data/ContestDataset/keypoints_train.csv"))

y_list = []
X_list = []

# Discard header
reader.next()

nonfull = []
i = 0

for row in reader:
    y = row[:-1]
    x = row[-1]

    X = x.split(' ')
    X = map(lambda x: float(x),X)
    X_list.append(X)

    y = map(lambda y: float(y) if y else -1,y)
    y_list.append(y)
    if -1 in y:
        nonfull.append(i)

    i += 1
    
    if len(nonfull)>NB+1:
        break

Xs = np.asarray(X_list)
Ys = np.asarray(y_list)

print nonfull
X = np.vstack((Xs[:NB/2],Xs[nonfull[:NB/2]]))
Y = np.vstack((Ys[:NB/2],Ys[nonfull[:NB/2]]))

print X.shape[1:3]

X = X.reshape((X.shape[0],np.sqrt(X.shape[1]),np.sqrt(X.shape[1]),1))

# no layer
pv = patch_viewer.PatchViewer( (int(np.sqrt(NB)), int(np.sqrt(NB))), X.shape[1:3], is_color = False)

for i in xrange(NB):
    for j in xrange(0,Y.shape[1],2):
        kx = Y[i,j]
        ky = Y[i,j+1]
        if kx > 0 and ky > 0:
            X[i,ky+1:ky+1,kx-1:kx+1] = 0
        elif kx!=-1 or ky!=-1:
            raise RuntimeError("WTF? x=%f y=%f" % (kx, ky))
    pv.add_patch(X[i,:,:,0], activation = 0.0, rescale = True)

pv.save("0.png")

# first layer
Z = 0.5*0.5

s = ndimage.zoom(X[0,:,:,0],Z)
pv = patch_viewer.PatchViewer( (int(np.sqrt(NB)), int(np.sqrt(NB))), s.shape[:2], is_color = False)

for i in xrange(NB):
    for j in xrange(0,Y.shape[1],2):
        kx = Y[i,j]
        ky = Y[i,j+1]
        if kx > 0 and ky > 0:
            X[i,ky-2:ky+2,kx-2:kx+2] = 0
        elif kx!=-1 or ky!=-1:
            raise RuntimeError("WTF? x=%f y=%f" % (kx, ky))
    s = ndimage.zoom(X[i,:,:,0],Z)
    print s.shape
    pv.add_patch(s, activation = 0.0, rescale = True)

pv.save("1.png")

# second layer
Z = 0.5
sx = 96/2/2
sy = 96/2

s = ndimage.zoom(X[0,sy:sy+96/2,sx:sx+96/2,0],Z)
pv = patch_viewer.PatchViewer( (int(np.sqrt(NB)), int(np.sqrt(NB))), s.shape[:2], is_color = False)

for i in xrange(NB):
    for j in xrange(0,Y.shape[1],2):
        kx = Y[i,j]
        ky = Y[i,j+1]
        if kx > 0 and ky > 0:
            X[i,ky-2:ky+2,kx-2:kx+2] = 0
        elif kx!=-1 or ky!=-1:
            raise RuntimeError("WTF? x=%f y=%f" % (kx, ky))
    s = ndimage.zoom(X[i,sy:sy+96/2,sx:sx+96/2,0],Z)
    print s.shape
    pv.add_patch(s, activation = 0.0, rescale = True)

pv.save("2.png")

# third layer
sx = 96/2/2
sy = 96/2+96/4

pv = patch_viewer.PatchViewer( (int(np.sqrt(NB)), int(np.sqrt(NB))), X[0,sy:sy+96/4,sx:sx+96/4,:].shape[:2], is_color = False)

for i in xrange(NB):
    for j in xrange(0,Y.shape[1],2):
        kx = Y[i,j]
        ky = Y[i,j+1]
        if kx > 0 and ky > 0:
            X[i,ky-2:ky+2,kx-2:kx+2] = 0
        elif kx!=-1 or ky!=-1:
            raise RuntimeError("WTF? x=%f y=%f" % (kx, ky))
    s = X[i,sy:sy+96/4,sx:sx+96/4,:]
    print s.shape
    pv.add_patch(s, activation = 0.0, rescale = True)

pv.save("3.png")

# fourth layer
sx = 96/2/2+96/8
sy = 96/2+96/4

pv = patch_viewer.PatchViewer( (int(np.sqrt(NB)), int(np.sqrt(NB))), X[0,sy:sy+96/8,sx:sx+96/8,:].shape[:2], is_color = False)

for i in xrange(NB):
    for j in xrange(0,Y.shape[1],2):
        kx = Y[i,j]
        ky = Y[i,j+1]
        if kx > 0 and ky > 0:
            X[i,ky-2:ky+2,kx-2:kx+2] = 0
        elif kx!=-1 or ky!=-1:
            raise RuntimeError("WTF? x=%f y=%f" % (kx, ky))
    s = X[i,sy:sy+96/8,sx:sx+96/8,:]
    print s.shape
    pv.add_patch(s, activation = 0.0, rescale = True)

pv.save("4.png")

# produce heat-maps
X = Xs
Y = Ys

mean = X.mean(axis=0)
std = X.std(axis=0)

X = (X - mean) / (1e-4 + std)

X = X.reshape((X.shape[0],np.sqrt(X.shape[1]),np.sqrt(X.shape[1]),1))

h_maps = []
pts = np.vstack(tuple([np.hstack((np.ones((1,X.shape[1])).T*i,np.arange(X.shape[1]).reshape((1,X.shape[1])).T)) for i in range(X.shape[1])]))

for x,y in zip(X,Y):
    maps = x.copy().reshape((x.shape[0],x.shape[0]))

    for j in xrange(0,y.shape[0],2):
        kx = int(y[j])
        ky = int(y[j+1])
        print maps.shape
        if kx > 0 and ky > 0:
            print "point"
            print ky,kx
            print maps[ky-1:ky+1,kx-1:kx+1]
            maps[ky-1:ky+1,kx-1:kx+1] = 0
        elif kx!=-1 or ky!=-1:
            raise RuntimeError("WTF? x=%f y=%f" % (kx, ky))
        else:
            print "no"
 
    print maps.shape
    pl.imsave("kp.png",maps,cmap="gray")

    maps = np.zeros((Y.shape[1]/2,X.shape[1],X.shape[2]))

    for j in xrange(0,Y.shape[1],2):

        kx = int(y[j])
        ky = int(y[j+1])

        if kx > 0 and ky > 0:
#            D = np.array([kx,ky]) - pts
#            D2 = D*D
            print np.array([kx,ky]).reshape((1,2))
            print pts
            D2 = scipy.spatial.distance.cdist(np.array([kx,ky]).reshape((1,2)),pts)#,'minkowski',4)
#            D2 = np.sum(np.abs(np.array([kx,ky]).reshape((1,2))-pts),axis=1)
            print D2
#            print "around ",kx,ky
#            print "pts"
#            for z in range(7):
#                print pts[(int(kx)-3+z)*x.shape[0]+int(ky)-3:(int(kx)-3+z)*x.shape[0]+int(ky)+4]
#            for z in range(10):
#                print D2[0,(int(kx)-3+z)*x.shape[0]+int(ky)-3:(int(kx)-3+z)*x.shape[0]+int(ky)+4]
            print "D2 shape"
            print D2.shape
            D2 = D2.reshape((x.shape[0],x.shape[0]))
            print D2[ky,kx]
            print D2[kx-2:kx+2,ky-2:ky+2]
            maps[j/2] = 1.0/D2.T

            maps[j/2] = maps[j/2]/np.max(maps[j/2])
            pl.imsave("hm%d.png" % (j/2),maps[j/2],cmap="gray")
 
        elif kx!=-1 or ky!=-1:
            raise RuntimeError("WTF? x=%f y=%f" % (kx, ky))
    pl.imsave("hm.png",np.sum(maps,axis=0),cmap="gray")
#   pl.imsave("hm.png", np.multiply(np.sum(maps,axis=0)+0.1,x.reshape(x.shape[0],x.shape[0])),cmap="gray")

    break
    h_maps.append(maps)
