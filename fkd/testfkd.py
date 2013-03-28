import numpy as np
import csv
import pylab as pl
from scipy import ndimage

from pylearn2.gui import patch_viewer

NB = 36

reader = csv.reader(open("training.csv"))

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

X = np.asarray(X_list)
Y = np.asarray(y_list)

print nonfull
X = np.vstack((X[:NB/2],X[nonfull[:NB/2]]))
Y = np.vstack((Y[:NB/2],Y[nonfull[:NB/2]]))

print X.shape[1:3]

X = X.reshape((X.shape[0],np.sqrt(X.shape[1]),np.sqrt(X.shape[1]),1))

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
