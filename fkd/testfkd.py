import numpy as np
import csv
import pylab as pl

from pylearn2.gui import patch_viewer

from dispims import dispims

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

#pl.imshow(X[nonfull[i]],
pv = patch_viewer.PatchViewer( (int(np.sqrt(NB)), int(np.sqrt(NB))), X.shape[1:3], is_color = False)

for i in xrange(NB):
    print Y.shape
    for j in xrange(0,Y.shape[1],2):
        kx = Y[i,j]
        ky = Y[i,j+1]
        if kx > 0 and ky > 0:
            X[i,ky-2:ky+2,kx-2:kx+2] = 0
        elif kx!=-1 or ky!=-1:
            raise RuntimeError("WTF? x=%f y=%f" % (kx, ky))
    pv.add_patch(X[i,:,:,:], activation = 0.0, rescale = True)

pv.show()
