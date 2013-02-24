from scipy import ndimage, misc
import pylab as pl
import numpy as np
import csv

from timer import seconds_to_string, Timer

__all__ = ["transformations_dict",
           "load_dataset",
           "save_dataset",
           "apply_transformations",
           "show_samples"]

def randomindex(nb,length):
    nb = max(nb,1)

    random_list = []
    while nb > length:
        random_list += range(length)
        nb -= length

    rand = np.arange(length)
    np.random.shuffle(rand)
    random_list += list(rand[:nb])
    return random_list

    rand = []
    while len(rand) < nb:
        r = int(np.random.rand(1)*length)
        if r not in rand:
            rand.append(r)

    return rand

def imzoom(im,c):

    y = np.random.rand(*im.shape)
    im = ndimage.zoom(im,c)

    try:
        if c < 1.0:
            y[int((y.shape[0]-im.shape[0])/2.0+0.5):int(-(y.shape[0]-im.shape[0])/2.0),
              int((y.shape[1]-im.shape[1])/2.0+0.5):int(-(y.shape[1]-im.shape[1])/2.0)] = im
        else:
            y = im[int((im.shape[0]-y.shape[0])/2.0+0.5):int(-(im.shape[0]-y.shape[0])/2.0),
                   int((im.shape[1]-y.shape[1])/2.0+0.5):int(-(im.shape[1]-y.shape[1])/2.0)]
    except ValueError as e:
        return im


    return y

def zoom(Xs,ys,mean=1.0,std=0.1,ratio=10.0,t=None):
    print "scaling images..."

    rand_list = randomindex(Xs.shape[0]*ratio,Xs.shape[0])
    Xs = Xs[rand_list]
    ys = ys[rand_list]

    tx = []
    ty = []
    for X, y in zip(Xs,ys):
        rand = np.random.normal(mean,std)
        while rand < 0.0 or rand > 2.0:
            rand = np.random.normal(mean,std)
        zX = imzoom(X,rand)
        if zX.shape[0]==X.shape[0]:
            tx.append(zX)
            ty.append(y)

        if t: t.print_update(1)
    
    return np.array(tx), np.array(ty)

def translate(Xs,ys,mean=0.0,std=2.0,ratio=10.0,t=None):
    print "translating images..."

    rand_list = randomindex(Xs.shape[0]*ratio,Xs.shape[0])
    Xs = Xs[rand_list]
    ys = ys[rand_list]

    tx = []
    ty = []
    I = np.identity(2)
    for X,y in zip(Xs,ys):
        tx.append(ndimage.affine_transform(X,I,
            offset=(np.random.normal(mean,std),np.random.normal(mean,std))))
        ty.append(y)

    if t: t.print_update(1)

    return np.array(tx), np.array(ty)

def rotate(Xs,ys,mean=0.0,std=10.0,ratio=10.0,t=None):
    print "rotating images..."

    rand_list = randomindex(Xs.shape[0]*ratio,Xs.shape[0])
    Xs = Xs[rand_list]
    ys = ys[rand_list]

    tx = []
    ty = []
    for X, y in zip(Xs,ys):
        tx.append(ndimage.rotate(X,np.random.normal(mean,std),reshape=False))
        ty.append(y)

    if t: t.print_update(1)

    return np.array(tx), np.array(ty)

def clutter(images,mean=25,std=10.0,max_nb_per_image=2,nb=10,t=None):
    print "cluttering images..."
    nb = 20
    ts = []
    for i in xrange(nb):
        for image in images:
            for i in xrange(int(np.random.rand(1)*max_nb_per_image)):
                x, y = image.shape[0], image.shape[1]
                dx, dy = 1, 1
                while (x+dx > image.shape[0] or y+dy > image.shape[1] or
                       dx < 1.0 or dy < 1.0):
                    x = int(np.random.rand(1)*image.shape[0])
                    y = int(np.random.rand(1)*image.shape[1])
                    dx, dy = int(np.random.normal(mean,std)), int(np.random.normal(mean,std))

                tmp = image[:,:]
                tmp[x:x+dx,y:y+dy] = (np.random.rand(dx,dy)*255).astype(int)
                ts.append(tmp)

#            tmp = image[:,:]
#            tmp[:,image.shape[1]/2:] = (np.random.rand(image.shape[0],image.shape[1]/2)*255).astype(int)
#            ts.append(tmp)
#            tmp = image[:,:]
#            tmp[:image.shape[0]/2,:] = (np.random.rand(image.shape[0]/2,image.shape[1])*255).astype(int)
#            ts.append(tmp)

        if t: t.print_update(1)


    return images

def noise(Xs,ys,sigma=3,ratio=0.6,t=None):
    print "adding noise to images..."

    if ratio > 1.0:
        print "Do you really want a ratio of %f for noise?" % ratio
        print "Every images produced will always be similar"

    rand_list = randomindex(Xs.shape[0]*ratio,Xs.shape[0])
    Xs = Xs[rand_list]
    ys = ys[rand_list]

    tx = []
    ty = []

    for X, y in zip(Xs,ys):
        tx.append(ndimage.gaussian_filter(X,sigma))
        ty.append(y)

        if t: t.print_update(1)

    return np.array(tx), np.array(ty)

def sharpen(Xs,ys,sigma1=3,sigma2=1,alpha=30,ratio=0.6,t=None):
    print "sharpening images..."

    if ratio > 1.0:
        print "Do you really want a ratio of %f for sharpen?" % ratio
        print "Every images produced will always be similar"

    rand_list = randomindex(Xs.shape[0]*ratio,Xs.shape[0])
    Xs = Xs[rand_list]
    ys = ys[rand_list]

    tx = []
    ty = []

    for X, y in zip(Xs,ys):
        blurred_l = ndimage.gaussian_filter(X, sigma1)

        filter_blurred_l = ndimage.gaussian_filter(blurred_l, sigma2)
        sharpened = blurred_l + alpha * (blurred_l - filter_blurred_l)

        tx.append(sharpened)
        ty.append(y)

        if t: t.print_update(1)

    return np.array(tx), np.array(ty)

def denoise(Xs,ys,alpha=0.4,sigma1=2,sigma2=3,ratio=0.6,t=None):
    print "denoising images..."

    if ratio > 1.0:
        print "Do you really want a ratio of %f for denoise?" % ratio
        print "Every images produced will always be similar"

    rand_list = randomindex(Xs.shape[0]*ratio,Xs.shape[0])
    Xs = Xs[rand_list]
    ys = ys[rand_list]

    tx = []
    ty = []

    for X, y in zip(Xs,ys):
        noisy = X + alpha * X.std() * np.random.random(X.shape)

        gauss_denoised = ndimage.gaussian_filter(noisy, sigma1)

        med_denoised = ndimage.median_filter(noisy, sigma2)

        tx.append(med_denoised)
        ty.append(y)

        if t: t.print_update(1)

    return np.array(tx), np.array(ty)

def flip(Xs,ys,ratio=1,t=None):
    print "flipping images..."

    if ratio > 1.0:
        print "Do you really want a ratio of %f for flip?" % ratio
        print "Every images produced will always be similar"

    rand_list = randomindex(Xs.shape[0]*ratio,Xs.shape[0])
    Xs = Xs[rand_list]
    ys = ys[rand_list]

    tx = []
    ty = []
    for X, y in zip(Xs,ys):
        tx.append(np.fliplr(X))
        ty.append(y)

        if t: t.print_update(1)

    return np.array(tx), np.array(ty)

transformations_dict = {
                   "translate": translate,
                   "zoom"     : zoom,
                   "rotate"   : rotate,
                   "noise"    : noise,
                   "sharpen"  : sharpen,
                   "denoise"  : denoise,
                   "clutter"  : clutter,
                   "flip"     : flip 
                   }

def generate(transformation,settings,X,y,t=None):
    nX, ny = transformation(X,y,t=t,**settings)
    return nX, ny

def apply_transformations(X,y,transformations):
    
    nb = X.shape[0]
    for name, settings in transformations.items():
        nb = nb + nb * settings.get('ratio',2)

    print "A dataset of approximately %d images will we produced.." % nb
    t = Timer(nb)
    t.start()

    tX = X[:]
    ty = y[:]
    noise_transformations = []
    for noise_t in ['noise','sharpen','denoise']:
        if noise_t in transformations:
            settings = transformations.pop(noise_t)
            nX, ny = generate(transformations_dict[noise_t],settings,X,y,t=t)
            tX = np.vstack((tX,nX))
            ty = np.vstack((ty,ny))

    X = tX
    y = ty
    for name, settings in transformations.items():
        nX, ny = generate(transformations_dict[name],settings,X,y,t=t)
        X = np.vstack((X,nX))
        y = np.vstack((y,ny))

    print "it took",seconds_to_string(t.over())

    return X, y

# build image to show
def ptshow(images):
    i_results = np.zeros((images[0].shape[0]*int(np.sqrt(images.shape[0])),images[0].shape[1]*int(np.sqrt(images.shape[0]))))

    for i, image in enumerate(images):
        x  = (i/int(np.sqrt(images.shape[0])))*image.shape[0]
        y  = (i*image.shape[1])%(image.shape[1]*int(np.sqrt(images.shape[0])))

#        print x,x+image.shape[0],y,y+image.shape[1]
        
        i_results[x:x+image.shape[0],y:y+image.shape[1]] = image.reshape((image.shape[0],image.shape[1]))

    pl.imshow(i_results,cmap="gray")
    pl.show()

def load_dataset(file_path):
    print "loading dataset..."
    csv_file = open(file_path)
    reader = csv.reader(csv_file)

    # Discard header
    row = reader.next()

    Xs = []
    ys = []

    for row in reader:
        y_str, X_str = row
        ys.append(int(y_str))
        Xs.append(np.array([float(x) for x in X_str.split(" ")]))

    Xs = np.array(Xs)
    ys = np.array(ys)

    one_hot = np.zeros((ys.shape[0],np.max(ys)+1),dtype='float32')
    for i in xrange(ys.shape[0]):
        one_hot[i,ys[i]] = 1.
    ys = one_hot

    sqrt = np.sqrt(Xs.shape[1])
    Xs = Xs.reshape((Xs.shape[0],sqrt,sqrt))

    return Xs,ys

def show_samples(X,y,nb):
    rand = randomindex(nb,X.shape[0])
    ptshow(X[rand].reshape((nb,X.shape[1],X.shape[2])))

def save_dataset(X,y,file_path):
    print "saving extended dataset..."
    print "saving %d images..." % X.shape[0]

    f = open(file_path,"w")
    z = zip(X,y)
    np.random.shuffle(z)
    t = Timer(X.shape[0])
    t.start()
    for X, y in z:
        f.write("%d,\"%s\"\n" % (np.int(np.argmax(y)), " ".join([str(i) for i in X.flatten().astype(int)])))
        t.print_update(1)
    f.close()
    print "it took",seconds_to_string(t.over())
