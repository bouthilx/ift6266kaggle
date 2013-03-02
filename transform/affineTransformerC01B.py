import numpy
from numpy.random import binomial
from PIL import Image

class ImageAffineTransformer:
    
    def __init__(self,
                 img_width = 48,
                 img_height = 48,
                 p_hsymetry = 0.5,
                 p_translation = 1.0,
                 min_translation_pixels = -5,
                 max_translation_pixels = +5,
                 p_rotation = 1.0,
                 min_rotation_degrees = -5,
                 max_rotation_degrees = +5,
                 seed = 123):
    
        self.p_hsymetry = p_hsymetry
        self.p_translation = p_translation
        self.min_translation_pixels = min_translation_pixels
        self.max_translation_pixels = max_translation_pixels
        self.p_rotation = p_rotation
        self.min_rotation_degrees = min_rotation_degrees
        self.max_rotation_degrees = max_rotation_degrees
        self.img_width = img_width
        self.img_height = img_height 
        self.seed = seed
        
        # Seed numpy.random to ensure reproducibility
        numpy.random.seed(self.seed)
    
    
    def perform(self, X):
                
        images = numpy.copy(X)
        for idx in range(images.shape[3]):
        
            img_arr = images[0,:,:,idx] ##################################
            
            # Perform a horizontal reflexion, maybe...
            if binomial(n=1,p=self.p_hsymetry) == 1:
                img_arr = img_arr[:,::-1]
                            
            # Perform a translation, maybe...
            if binomial(n=1,p=self.p_translation) == 1:
                h_translation = numpy.random.randint(self.min_translation_pixels,
                                                     self.max_translation_pixels + 1)
                v_translation = numpy.random.randint(self.min_translation_pixels,
                                                     self.max_translation_pixels + 1)                 
                
                # Perform horizontal translation 
                if h_translation < 0:
                    temp = img_arr[:,-h_translation:]
                    img_arr[:,:h_translation] = temp
                    img_arr[:,h_translation:] = 0
                elif h_translation > 0:
                    temp = img_arr[:,:-h_translation]
                    img_arr[:,h_translation:] = temp
                    img_arr[:,:h_translation] = 0
                             
                # Perform vertical translation 
                if v_translation < 0:
                    temp = img_arr[-v_translation:,:]
                    img_arr[:v_translation,:] = temp
                    img_arr[v_translation:,:] = 0
                elif v_translation > 0:
                    temp = img_arr[:-v_translation,:]
                    img_arr[v_translation:,:] = temp
                    img_arr[:v_translation,:] = 0        
             
            # Perform a rotation, maybe...
            if binomial(n=1,p=self.p_rotation) == 1:
                deg_rotation = numpy.random.randint(self.min_rotation_degrees,
                                                    self.max_rotation_degrees + 1)

                if deg_rotation != 0:
                    img = Image.fromarray(img_arr)
                    img = img.rotate(deg_rotation)
                    img_arr = numpy.array(img)
                  
            images[0,:,:,idx] = img_arr ####################################
         
        return images
        
if __name__ == '__main__':
    # Test on an image
    arr = numpy.vstack((
              numpy.array(Image.open('picture.png').convert("L")).reshape(1,600*800),
              numpy.array(Image.open('picture.png').convert("L")).reshape(1,600*800)
          ))
    arr = numpy.array(Image.open('picture.png').convert("L")).reshape(1,600*800)
    
    trans = ImageAffineTransformer(img_width = 600, img_height=800)
    Image.fromarray(trans.perform(arr)[0].reshape(600,800)).show()
    Image.fromarray(trans.perform(arr)[0].reshape(600,800)).show()
    Image.fromarray(trans.perform(arr)[0].reshape(600,800)).show()
            
        
        
        
