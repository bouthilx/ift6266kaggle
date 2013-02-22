from scipy import ndimage, misc
import pylab as pl
import numpy as np

def translate(images,mean=0.0,std=10.0,nb):
    ts = []
    for i in xrange(nb):
        ts.append(images+np.random.normal(mean,std))
    
    return images

def zoom(images):
    return images

def rotate(images):
    return images

def flip(images):
    return images

def clutter(images):
    return images

def noise(images):
    # blur

    # sharpen

    # noise
    return images

transformations = [translate,zoom,rotate,flip,clutter,noise]

def generate(transformation,images,rate=0.0):
    if rate:
        pass

    return np.vstack((images,transformation(images)))

#images = [ndimage.zoom(misc.lena(),0.9)]
images = np.array([misc.lena()])

for transform in transformations:
    images = generate(transform,images)

print images.shape
#for image in images:
#    pl.imshow(image,cmap="gray")
#    pl.show()
