from keypoints_dataset import FacialKeypointDataset
import scipy

import matplotlib
import numpy as np

class FacialKeypointMap(FacialKeypointDataset):

    modes = ['full','axes','coordinates']

    def __init__(self,which_set,
                 base_path='/data/lisatmp/ift6266h13/ContestDataset',
                 start=None,
                 stop=None,
                 preprocessor=None,
                 fit_preprocessor=False,
                 output_shape=None,
                 axes=('b', 0, 1, 'c'),
                 fit_test_preprocessor=False,
                 floating_positions=True,
                 mode='full',
                 keypoint=0):

        FacialKeypointDataset.__init__(self,
                 which_set,
                 base_path,
                 start,
                 stop,
                 preprocessor,
                 fit_preprocessor,
                 axes,
                 fit_test_preprocessor)
        
        self.mode = mode
        assert mode in self.modes
        self.keypoint = keypoint
        assert self.keypoint < self.y.shape[1]/2
        self.output_shape = output_shape
        self.floating_positions = floating_positions
        self.y_maps = None
#        y = self.get_targets()
#        self.y = y
   
    def get_targets(self):
        if self.y_maps is None:
            self.y_maps = self.build_targets(self.y,self.mode)
       
        return self.y_maps

    def build_targets(self,Y,mode):
        """
            returns a map for each keypoint
        """

        Ymaps = []
#        Fullmaps = []
#        print X.shape
    
        shape = self.view_converter.shape
        axes = self.view_converter.axes
        
#        nX = X
#        X = X.transpose((axes.index('b'),axes.index(0),axes.index(1),axes.index('c')))

        pts = np.vstack(tuple([np.hstack((np.ones((1,shape[0])).T*i,
                   np.arange(shape[1]).reshape((1,shape[1])).T)) for i in range(shape[0])]))

        for y in Y:
        
            maps = np.zeros((Y.shape[1]/2,shape[0],shape[1]))

            for j in xrange(0,Y.shape[1],2):

                kx = y[j]
                ky = y[j+1]

                if not self.floating_positions:
                    kx = int(kx+0.5)
                    ky = int(ky+0.5)

                if kx > 0 and ky > 0:
                    D2 = scipy.spatial.distance.cdist(np.array([kx,ky]).reshape((1,2)),pts,'minkowski',2)
#                    D2 = D2**20
#                    D2 = np.sqrt(D2.reshape((X.shape[1],X.shape[2])))
                    D2 = D2.reshape((shape[0],shape[1]))

#                    print np.max(D2),np.mean(D2),np.min(D2)
#                    print np.max(maps[j/2]),np.mean(maps[j/2]),np.min(maps[j/2])
#                    print np.max(maps[j/2]),np.mean(maps[j/2]),np.min(maps[j/2])

#                    print np.sum(np.isnan(maps[j/2]))
#                    maps[j/2] = maps[j/2] * np.isnan(maps[j/2])

#                    maps[j/2] = 1.0/D2.T
                    maps[j/2] = D2.T
                    # invert black and white
                    maps[j/2] = np.max(maps[j/2]) - maps[j/2]
                    maps[j/2] = maps[j/2]**10
#                    print np.max(maps[j/2]),np.mean(maps[j/2]),np.min(maps[j/2])

#                    maps[j/2,min(X.shape[1]-1,int(ky+0.5)),min(X.shape[2]-1,int(kx+0.5))] = 0.0
#                    print np.max(maps[j/2]),np.mean(maps[j/2]),np.min(maps[j/2])
#                    maps[j/2] = 1.0 - maps[j/2]
                    maps[j/2] = maps[j/2] - np.min(maps[j/2])
                    maps[j/2] = maps[j/2]/np.max(maps[j/2])
#                    maps[j/2] = maps[j/2]**20
#                    maps[j/2] = (maps[j/2]-np.min(maps[j/2]))/(np.max(maps[j/2]) -np.min(maps[j/2]))
#                    maps[j/2] = maps[j/2]/np.max(maps[j/2])

#                    rgb = np.zeros((maps[j/2].shape[0],maps[j/2].shape[1],3))
#                    for i in range(3):
#                       rgb[:,:,i] = maps[j/2]
#                    hsv = matplotlib.colors.rgb_to_hsv(rgb)
#                    hsv[:,:,2] = hsv[:,:,2] * 100
#                    rgb = matplotlib.colors.hsv_to_rgb(hsv)
#
#                    maps[j/2] = rgb[:,:,0]

                
#                    print np.max(maps[j/2]),np.mean(maps[j/2]),np.min(maps[j/2])

#                    maps[j/2] = maps[j/2].astype(int)
#                    print maps[j/2,0,0]
#                    print np.max(maps[j/2]),np.mean(maps[j/2]),np.min(maps[j/2])

#                    print int(ky+0.5),int(kx+0.5)
#                    print maps[j/2].shape
         
                elif kx!=-1 or ky!=-1:
                    raise RuntimeError("keypoint position error: x=%f y=%f" % (kx, ky))
            # all features separated
#            Ymaps.append(maps.reshape((Y.shape[1]/2,X.shape[1],X.shape[2],1)))
            # only feature #1
            if self.output_shape and self.output_shape[0]-maps[self.keypoint].shape[0]:
                tmp = np.zeros(self.output_shape)
                min_dx = (self.output_shape[0] - maps[self.keypoint].shape[0])/2.0
                max_dx = int(min_dx +0.5)
                min_dx = int(min_dx)
                min_dy = (self.output_shape[1] - maps[self.keypoint].shape[1])/2.0
                max_dy = int(min_dy +0.5)
                min_dy = int(min_dy)
#                print self.output_shape
#                print maps[self.keypoint].shape
#                print X.shape
#                print nX.shape
                tmp[min_dx:-max_dx,min_dy:-max_dy,:] = maps[self.keypoint].reshape((shape[0],shape[1],shape[2]))
            else:
                tmp = maps[self.keypoint]

            maps = maps[self.keypoint]

            if mode == 'full':
                Ymaps.append(maps.reshape(shape[0]*shape[1]))
            elif mode == 'axes':
                tmp = np.hstack([np.sum(maps,axis=0).flatten(),
                                 np.sum(maps,axis=1).flatten()])
#                print tmp.shape
#                print np.argmax(tmp)
                Ymaps.append(tmp.reshape(shape[0]+shape[1]))
            elif mode == 'coordinates':
                tmp = np.array([np.argmax(np.sum(maps,axis=0).flatten()),
                                np.argmax(np.sum(maps,axis=1).flatten())]).astype('float32')
                Ymaps.append(tmp)
            else:
                raise RuntimeError('Invalid mode %s' % mode)
#            print "shape"
#            print tmp.shape
#            print self.output_shape
#            Ymaps.append(tmp.transpose((2,0,1)).reshape((100*100,)))
#            Fullmaps.append(tmp.reshape((np.prod(self.output_shape,))))
#            tmp = np.array([np.argmax(np.sum(tmp,axis=1).flatten()),
#                            np.argmax(np.sum(tmp,axis=2).flatten())]).astype('float32')
#            print tmp.shape
#            Ymaps.append(tmp)
#            Ymaps.append(tmp.reshape((np.prod(self.output_shape))))
#            Ymaps.append(maps[self.keypoint].reshape((X.shape[1],X.shape[2],1)))
            # all features merged
#            Ymaps.append(np.sum(maps,axis=0).reshape((X.shape[1]*X.shape[2])))

#        self.fullmaps = np.array(Fullmaps)
        print "dataset shape"
        print np.array(Ymaps).shape

        return np.array(Ymaps)
