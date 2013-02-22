from scipy import ndimage, misc
import pylab as pl
import numpy as np

lena = misc.lena()

lx, ly = lena.shape

show = [#1,
        #2,
        #3,
        #4,
        #5,
        #6,
        #7,
        #8,
        #9,
        #10,
        #11,
        #12,
        13,
        14]

def zoom(im,c):

    y = np.random.rand(*im.shape)
    im = ndimage.zoom(im,c)

    if c < 1.0:
        y[(y.shape[0]-im.shape[0])/2.0:-(y.shape[0]-im.shape[0])/2.0,
          (y.shape[1]-im.shape[1])/2.0:-(y.shape[1]-im.shape[1])/2.0] = im
    else:
        y = im[(im.shape[0]-y.shape[0])/2.0:-(im.shape[0]-y.shape[0])/2.0,
              (im.shape[1]-y.shape[1])/2.0:-(im.shape[1]-y.shape[1])/2.0]

    return y

if 1 in show:
    pl.imshow(lena,cmap="gray")
    pl.show()

if 2 in show:
    pl.imshow(lena[lx/4:-lx/4,ly/4:-ly/4],cmap="gray")
    pl.show()

if 3 in show:
    pl.imshow(np.fliplr(lena),cmap="gray")
    pl.show()

if 4 in show:
    pl.imshow(ndimage.rotate(lena,45),cmap="gray")
    pl.show()

if 5 in show:
    pl.imshow(ndimage.rotate(lena,45,reshape=False),cmap="gray")
    pl.show()

if 6 in show:
    pl.imshow(ndimage.gaussian_filter(lena,sigma=3),cmap="gray")
    pl.show()

if 7 in show:
    pl.imshow(ndimage.gaussian_filter(lena,sigma=5),cmap="gray")
    pl.show()

if 8 in show:
    pl.imshow(ndimage.uniform_filter(lena,size=11),cmap="gray")
    pl.show()

if 9 in show:
    blurred_l = ndimage.gaussian_filter(lena, 3)

    filter_blurred_l = ndimage.gaussian_filter(blurred_l, 1)
    alpha = 30
    sharpened = blurred_l + alpha * (blurred_l - filter_blurred_l)

    pl.imshow(sharpened,cmap="gray")
    pl.show()

if 10 in show:
    noisy = lena + 0.4 * lena.std() * np.random.random(lena.shape)

    gauss_denoised = ndimage.gaussian_filter(noisy, 2)

    med_denoised = ndimage.median_filter(noisy, 3)

    pl.imshow(med_denoised,cmap="gray")
    pl.show()

if 11 in show:
    print "11"
    pl.imshow(zoom(lena,2),cmap="gray")
    pl.show()

if 12 in show:
    print "12"
    pl.imshow(zoom(lena,0.5),cmap="gray")
    pl.show()

if 13 in show:
    m = np.array([[1,0,10],
                  [0,1, 10],
                  [0,0,1]])

    # comment faire la translation??
    pl.imshow(ndimage.interpolation.affine_transform(lena,m),cmap="gray")
    pl.show()
