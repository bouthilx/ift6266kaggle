from pylearn2.train_extensions import TrainExtension
from pylearn2.gui import patch_viewer
import os
import numpy as np

from theano import function

class MonitorFilters(TrainExtension):
    """
        Makes the noise smaller over epochs
    """
    def __init__(self, save_path, param, layer=0):
        """
        """

        self.param = param
        self.save_path = save_path
        self.layer = layer
        self.count = 0
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
 
    def on_monitor(self, model, dataset, algorithm):
        save_path = self.save_path + "/%s_epoch-%d.png"
        params = model.get_params()
        print params
        param = filter(lambda a:a.name==self.param,params)[self.layer]
        value = param.get_value(borrow=True)
        print value.shape
        rows = int(np.sqrt(value.shape[0]))
        cols = int(value.shape[0]/rows + 1)

        # how can we know how to transpose correctly?
        value = value.transpose(0,2,3,1)
              
        pv = patch_viewer.PatchViewer( (rows, cols), value.shape[1:3], is_color = False)
        for i in xrange(value.shape[0]):
            pv.add_patch(value[i,:,:,:], activation = 0.0, rescale = True)
        pv.save(save_path % (self.param,self.count))
        self.count += 1

class MonitorResults(TrainExtension):
    """
        Makes the noise smaller over epochs
    """
    def __init__(self, save_path, mode='full', map_layer=0, axes_layer=-1):
        """
        """

        self.save_path = save_path
        self.mode = mode
        self.map_layer = map_layer
        self.axes_layer = axes_layer
        self.count = 0
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.f_map = None

    def build_fprop(self,X,model,layer):

        if layer==-1:
            layer = len(model.layers)-1
 


        for i in range(layer+1):
            X = model.layers[i].fprop(X)

        return X

    def scaledown(self,y,output_shape):
        return y[:,:output_shape[0],:output_shape[1],:output_shape[2]]


    def draw_axes(self,im,axes):
        im = im - np.min(im)
        t_max = np.max(im)
        if t_max!=0.0:
            im = im/t_max

        axes[:,:axes.shape[1]/2] = axes[:,:axes.shape[1]/2] - np.min(axes[:,:axes.shape[1]/2])
        t_max = np.max(np.max(axes[:,:axes.shape[1]/2]))
        if t_max!=0.0:
            axes[:,:axes.shape[1]/2] = axes[:,:axes.shape[1]/2]/t_max

        axes[:,axes.shape[1]/2:] = axes[:,axes.shape[1]/2:] - np.min(axes[:,axes.shape[1]/2:])
        t_max = np.max(axes[:,axes.shape[1]/2:])
        if t_max!=0.0:
            axes[:,axes.shape[1]/2:] = axes[:,axes.shape[1]/2:]/t_max

        new_im = np.zeros((im.shape[0],im.shape[1]+15,im.shape[2]+15,im.shape[3]))

        #copy image
        new_im[:,:im.shape[1],:im.shape[2],:] = im

        for i in xrange(1,11):
            new_im[:,-i,:-15,0] += np.abs(axes[:,:axes.shape[1]/2])
            new_im[:,:-15,-i,0] += np.abs(axes[:,axes.shape[1]/2:])
        
        return new_im

    def on_monitor(self, model, dataset, algorithm):
        save_path = self.save_path + "/epoch-%d.png"
        if not self.f_map:
            X = model.get_input_space().make_theano_batch(name="%s[X]" % self.__class__.__name__)
            self.f_map = function([X],self.build_fprop(X,model,self.map_layer),on_unused_input='ignore')
            if self.mode=="axes":
                self.f_axes = function([X],self.build_fprop(X,model,self.axes_layer),
                        on_unused_input='ignore')

        # get data where the keypoint is present
        targets = dataset.y
        idx = []
        i = 0
        while len(idx)<32:
            if targets[i][dataset.keypoint*2]>0:
                idx.append(i)
            i+=1

        dX = dataset.adjust_for_viewer(dataset.get_batch_topo(dataset.y.shape[0]))
        dX = dX[idx]

        if self.mode=='axes':
            y_axes = self.f_axes(dX.astype('float32'))

            # build crushed axes
            yh_map= self.f_map(dX.astype('float32'))
            yh_map = yh_map.transpose(0,2,3,1)
            yh_map = self.scaledown(yh_map,dataset.view_converter.shape)
            y_hat = self.draw_axes(yh_map,y_axes)
        else:
            y_hat = self.f_map(dX.astype('float32'))
            y_hat = y_hat.reshape([32]+dataset.view_converter.shape)

        if self.mode=='axes':

            full_y = dataset.build_targets(dataset.y[idx],'full')
            full_y = full_y.reshape([32]+dataset.view_converter.shape)

            axes_y = dataset.build_targets(dataset.y[idx],'axes')
            y = self.draw_axes(full_y,axes_y)
        else:
            y = dataset.build_targets(dataset.y[idx],'full').reshape([32]+dataset.view_converter.shape)
        pv = patch_viewer.PatchViewer( (1, 2), y_hat.shape[1:3], is_color = False)
        pv.add_patch(y_hat[0], activation = 0.0, rescale = True)
        pv.add_patch(y[0], activation = 0.0, rescale = True)
        pv.save(save_path % self.count)
        self.count += 1
