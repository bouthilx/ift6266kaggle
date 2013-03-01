import numpy as np
from scipy import ndimage, misc

class TransformationPipeline:
    """
        Apply transformations sequentially
    """
    # shape should we switched to Conv2DSpace or something similar to make space conversion more general
    def __init__(self,transformations,seed=None,shape=None):
        self.transformations = transformations
        self.shape = shape

        if seed:
            np.random.RandomState(seed)

    def perform(self, X):
        shape = X.shape
        if self.shape:
            X = X.reshape(tuple([X.shape[0]]+self.shape))

        for transformation in self.transformations:
            X = transformation.perform(X)

        return X.reshape(shape)

class TransformationPool:
    """
        Apply one of the transformations given the probability distribution
        default : equal probability for every transformations
    """
    def __init__(self,transformations,p_distribution=None,seed=None,shape=None):
        self.transformations = transformations
        self.shape = shape

        if p_distribution:
            self.p_distribution = p_distribution
        else:
            self.p_distribution = np.ones(len(transformations))/(.0+len(transformations))

        if seed:
            np.random.RandomState(seed)

    def perform(self, X):
        shape = X.shape
        if self.shape:
            X = X.reshape(tuple([X.shape[0]]+self.shape))

        print "perform Pool"

        for i in xrange(X.shape[0]):
            # randomly pick one transformation according to probability distribution
            t = np.array(self.p_distribution).cumsum().searchsorted(np.random.sample(1))

            X[i] = self.transformations[t].perform(X[i].reshape((1,48,48)))

        return X

def gen_mm_random(f,args,min,max,max_iteration=50):
    if not min:
        min = -1e99

    if not max:
        max = 1e99

    i = 0
    r = f(**args)
    while (r < min or max < r) and i < max_iteration:
        r = f(**args)
        i += 1

    if i >= max_iteration:
        raise RuntimeError("Were not able to generate a random value between "+str(min)+" and "+str(max)+" in less than "+str(i)+" iterations")

    return r

class RandomTransformation:
    def __init__(self,p):
        self.p = p

    def perform(self,X):
        
        for i in xrange(X.shape[0]):
            p = np.random.random(1)[0]
            if p < self.p:
                X[i] = self.transform(X[i])

        return X

    def transform(self,X):
        return X

class Translation(RandomTransformation):
    def __init__(self,p=1,mean=0.0,std=1.0,min=None,max=None):
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())

        self.I = np.identity(2)

    def transform(self,x):

        rx = gen_mm_random(np.random.normal,dict(loc=self.mean,scale=self.std),self.min,self.max)
        ry = gen_mm_random(np.random.normal,dict(loc=self.mean,scale=self.std),self.min,self.max)

        return ndimage.affine_transform(x,self.I,offset=(rx,ry))

class Scaling(RandomTransformation):
    def __init__(self,p=1,mean=1.0,std=0.1,min=0.1,max=None):
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())

        pass

    def transform(self, x):

        zoom = gen_mm_random(np.random.normal,dict(loc=self.mean,scale=self.std),self.min,self.max)
        shape = x.shape

        im = ndimage.zoom(x,zoom)
        try:
            im = self._crop(im,shape)
        except ValueError as e:
            im = x

        if im.shape[0] != shape[0]:
            return x

        return im

    def _crop(self,x,shape):
        # noise or black pixels?
#        y = np.random.rand(*shape)*255
        y = np.zeros(shape)

        if x.shape[0] < shape[0]:
            y[int((y.shape[0]-x.shape[0])/2.0+0.5):int(-(y.shape[0]-x.shape[0])/2.0),
              int((y.shape[1]-x.shape[1])/2.0+0.5):int(-(y.shape[1]-x.shape[1])/2.0)] = x
        else:
            y = x[int((x.shape[0]-y.shape[0])/2.0+0.5):int(-(x.shape[0]-y.shape[0])/2.0),
                  int((x.shape[1]-y.shape[1])/2.0+0.5):int(-(x.shape[1]-y.shape[1])/2.0)]
        return y

class Rotation(RandomTransformation):
    def __init__(self,p=1,mean=0.0,std=10.0,min=None,max=None):
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())
        pass

    def transform(self,x):

        degree = gen_mm_random(np.random.normal,dict(loc=self.mean,scale=self.std),self.min,self.max)

        return ndimage.rotate(x,degree,reshape=False)

class GaussianNoise(RandomTransformation):
    def __init__(self,p,sigma=1.0):
        """
            Sigma: from 1.0 to 5.0
        """
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())
        pass

    def transform(self,x):

        return ndimage.gaussian_filter(x,self.sigma)

class Sharpening(RandomTransformation):
    def __init__(self,p,sigma1=3,sigma2=1,alpha=30):
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())
        pass

    def transform(self,x):

        blurred_l = ndimage.gaussian_filter(x, self.sigma1)

        filter_blurred_l = ndimage.gaussian_filter(blurred_l, self.sigma2)
        return blurred_l + self.alpha * (blurred_l - filter_blurred_l)

class Denoising(RandomTransformation):
    def __init__(self,p,sigma1=2,sigma2=3,alpha=0.4):
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())
        pass

    def transform(self,x):
        noisy = x + self.alpha * x.std() * np.random.random(x.shape)

        gauss_denoised = ndimage.gaussian_filter(noisy, self.sigma1)

        return ndimage.median_filter(noisy, self.sigma2)

class Flipping(RandomTransformation):
    def __init__(self,p=1,min=None,max=None):
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())
        pass

    def transform(self,x):

        return np.fliplr(x)
      
class Occlusion:
    def __init__(self,p=0.05,nb=10,mean=4,std=10.0,min=2,max=1.0):
        self.__dict__.update(locals())
        pass

    def perform(self,X):
        
        for i in xrange(X.shape[0]):
            X[i] = self.transform(X[i])

        return X

    def transform(self,image):
        nb = np.sum(np.random.random(self.nb)<self.p)
        for i in xrange(nb):
            x = int(gen_mm_random(np.random.random_sample,dict(),
                                  0.0,
                                  min(self.max,
                                      (image.shape[0]-self.mean-1)/(.0+image.shape[0]))
                    )*image.shape[0]
                )

            y = int(gen_mm_random(np.random.random_sample,dict(),
                                  0.0,
                                  min(self.max,
                                      (image.shape[1]-self.mean-1)/(.0+image.shape[1]))
                    )*image.shape[1]
                )

            dx = int(gen_mm_random(np.random.normal,
                               dict(loc=self.mean,scale=self.std),
                               self.min,image.shape[0]-x))
            dy = int(gen_mm_random(np.random.normal,
                               dict(loc=self.mean,scale=self.std),
                               self.min,image.shape[1]-y))

            image[x:(x+dx),y:(y+dy)] = np.zeros((dx,dy)).astype(int)

        return image

class HalfFace(RandomTransformation):
    def __init__(self,p=0.05):
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())
        pass

    def transform(self,image):
        x_center = image.shape[0]/2
        y_center = image.shape[1]/2
        if np.random.rand(1) >= 0.5:
            image[:,:y_center] = np.zeros(image[:,:y_center].shape)#(np.random.rand(*image[:x_center,:y_center].shape)*255).astype(int)
        else:
            image[:,y_center:] = np.zeros(image[:,:y_center].shape)#(np.random.rand(*image[:,y_center:].shape)*255).astype(int)
        return image
