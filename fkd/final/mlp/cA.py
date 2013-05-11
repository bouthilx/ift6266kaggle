"""
 This code is adapted from the tutorial introducing denoising auto-encoders (dA) using Theano.
"""

import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

np = numpy

class cA(object):
    """Contractive autoencoder"""

    def __init__(self, numpy_rng, theano_rng, numvis, numhid, vistype, contraction, activation):

        self.numvis = numvis
        self.numhid  = numhid
        self.contraction = contraction
        self.vistype = vistype
        self.theano_rng = theano_rng
        self.activation = activation

        self.W_init = numpy.asarray( numpy_rng.uniform( low  = -4*numpy.sqrt(6./(numhid+numvis)), high =  4*numpy.sqrt(6./(numhid+numvis)), size = (numvis, numhid)), dtype = theano.config.floatX)
        self.W = theano.shared(value = self.W_init, name ='W')

        self.bvis = theano.shared(value=numpy.zeros(numvis, dtype=theano.config.floatX), name='bvis')
        self.bhid = theano.shared(value=numpy.zeros(numhid, dtype=theano.config.floatX), name ='bhid')
        self.params = [self.W, self.bhid, self.bvis]

    def hiddens(self,X):
        return self.activation(T.dot(X, self.W) + self.bhid)

    def fprop(self,X):
        hiddens = self.hiddens(X)
        if self.vistype == 'binary':
            outputs = T.nnet.sigmoid(T.dot(hiddens, self.W.T) + self.bvis)
        elif self.vistype == 'real':
            outputs = self.activation(T.dot(hiddens, self.W.T) + self.bvis) #####

        return outputs

    def cost(self,X,test=None):
        hiddens = self.hiddens(X)
        outputs = self.fprop(X)

        if test is not None:
            print "test hiddens"
            f = theano.function([X],hiddens)
            print np.mean(f(np.random.random((32,96*96)).astype('float32')))
            print np.mean(f(test[:32]))

            print "test outputs"
            f = theano.function([X],outputs)
            print np.mean(f(np.random.random((32,96*96)).astype('float32')))
            print np.mean(f(test[:32]))

        if self.vistype == 'binary':
            L = - T.sum(X*T.log(outputs) + (1-X)*T.log(1-outputs), axis=1)
        elif self.vistype == 'real':
            L = T.sum(0.5 * ((X - outputs)**2), axis=1)

        if test is not None:
            print "test LR"
            f = theano.function([X],L)
            print np.mean(f(np.random.random((32,96*96)).astype('float32')))
            print np.mean(f(test[:32]))

        contractive_cost = T.sum( ((hiddens * (1 - hiddens))**2) * T.sum(self.W**2, axis=0), axis=1)

        if test is not None:
            print "test LC"
            f = theano.function([X],contractive_cost)
            print np.mean(f(np.random.random((32,96*96)).astype('float32')))
            print np.mean(f(test[:32]))

        L = L + self.contraction * contractive_cost

        if test is not None:
            print "test L"
            f = theano.function([X],L)
            print np.mean(f(np.random.random((32,96*96)).astype('float32')))
            print np.mean(f(test[:32]))

            print "test mean L"
            f = theano.function([X],T.mean(L))
            print f(np.random.random((32,96*96)).astype('float32'))
            print np.mean(f(test[:32]))

        return T.mean(L)

    def errors(self,X,test=None):
        return self.cost(X,test)

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum 

    def get_params(self):
        return numpy.concatenate([p.get_value().flatten() for p in self.params])

    def save(self, filename):
        numpy.save(filename, self.get_params())

    def load(self, filename):
        self.updateparams(numpy.load(filename))

    def get_cost_updates(self,X,learning_rate):
        self.cost(X)
        cost = self._cost
        gparams = self._grads

        updates = []
        for param, gparam in zip(self.params, gparams):
            print param,learning_rate,gparam
            updates.append((param, param - learning_rate * gparam))

        return cost, updates

class cdA(object):

    def __init__(self, numpy_rng, theano_rng, 
                       numclasses, numvis, numhid, vistype, contraction, orthogonality):

        self.numclasses = numclasses
        self.numvis = numvis
        self.numhid  = numhid
        self.contraction = contraction
        self.orthogonality = orthogonality
        self.vistype = vistype
        self.theano_rng = theano_rng

        self.W_init = numpy.asarray( numpy_rng.uniform( low  = -4*numpy.sqrt(6./(numhid+numvis)), high =  4*numpy.sqrt(6./(numhid+numvis)), size = (numvis, numhid/2)), dtype = theano.config.floatX)
        self.V_init = numpy.asarray( numpy_rng.uniform( low  = -4*numpy.sqrt(6./(numhid+numvis)), high =  4*numpy.sqrt(6./(numhid+numvis)), size = (numvis, numhid/2)), dtype = theano.config.floatX)
        self.U_init = numpy.asarray( numpy_rng.uniform( low  = -4*numpy.sqrt(6./(numhid+numvis)), high =  4*numpy.sqrt(6./(numhid+numvis)), size = (numhid/2,numclasses)), dtype = theano.config.floatX)

        self.W = theano.shared(value = self.W_init, name ='W')
        self.V = theano.shared(value = self.V_init, name ='V')
        self.U = theano.shared(value = self.U_init, name ='U')

        self.bw = theano.shared(value=numpy.zeros(numhid/2, dtype=theano.config.floatX), name='bw')
        self.bv = theano.shared(value=numpy.zeros(numhid/2, dtype=theano.config.floatX), name='bv')
        self.rho = theano.shared(value=numpy.zeros(numvis, dtype=theano.config.floatX), name='rho')
        self.bu = theano.shared(value=numpy.zeros(numclasses, dtype=theano.config.floatX), name='bu')

        self.params = [self.W, self.V, self.U, self.bw, self.bv, self.rho, self.bu]

    def hiddens(self,X):
        return [T.nnet.sigmoid(T.dot(X, self.V) + self.bv),
                T.nnet.sigmoid(T.dot(X, self.W) + self.bw)]

    def classif(self,X):
        hiddens = self.hiddens(X)
        return T.nnet.sigmoid(T.dot(hiddens[1],self.U) + self.bu)

    def fprop(self,X):
        hiddens = self.hiddens(X)
        outputs = T.dot(hiddens[0], self.V.T) + T.dot(hiddens[1], self.W.T) + self.rho

        return outputs

    def cost(self,X,Y,test=None):
        hiddens = self.hiddens(X)
        outputs = self.fprop(X)
        classif = self.classif(X)

        if self.vistype == 'binary':
            L = - T.sum(X*T.log(outputs) + (1-X)*T.log(1-outputs), axis=1)
        elif self.vistype == 'real':
            L = T.sum(0.5 * ((X - outputs)**2), axis=1)

        V_J = ((hiddens[0] * (1 - hiddens[0]))**2) * T.sum(self.V**2, axis=0)
        W_J = ((hiddens[1] * (1 - hiddens[1]))**2) * T.sum(self.W**2, axis=0)

        contractive_V_cost = T.sum(V_J,axis=1)
        contractive_W_cost = T.sum(W_J,axis=1)

        contractive_cost = contractive_V_cost + contractive_W_cost

        V_i = T.dot(hiddens[0]*(1-hiddens[0]),self.V.T)
        W_j = T.dot(hiddens[1]*(1-hiddens[1]),self.W.T)
        orthogonality_cost = T.sum(V_i*W_j,axis=1)

        # multiplied at the end such that if Y has no class the cost is 0
        classif_cost = -T.sum(Y*T.log(classif)+(1-Y)*T.log(1-classif),axis=1)*T.sum(Y,axis=1)

        cda = self.contraction * (contractive_cost + self.orthogonality*orthogonality_cost)

        L = L + cda + classif_cost

        return T.mean(L)

    def errors(self,X,Y):
        return self.cost(X,Y)

class dcA(object):
    """Denoizing classifier autoencoder"""

    def __init__(self, numpy_rn, theano_rng, numvis, numhid, corruption):

        self.numvis = numvis
        self.numhid  = numhid
        self.corruption = corruption 
        self.theano_rng = theano_rng

        self.Ws = []
        self.bs = []
        for i,h in enumerate(zip([numvis]+numhid,numhid)):
            h_in,h_out = h
            b = 4*numpy.sqrt(6./(h_in+h_out))
            W_init = numpy.asarray( 
                         numpy_rng.uniform( low=-b, high=b, 
                                            size=(h_in, h_out)), dtype=theano.config.floatX)
            self.Ws.append(theano.shared(value = W_init, name ='W_%d' % i))
           
            self.bs.append(theano.shared(value=numpy.zeros(h_out, dtype=theano.config.floatX), name ='b_%d' % i))
            print i, h_in,h_out

        for i,h_in in enumerate(reversed(numhid)):
            self.bs.append(theano.shared(value=numpy.zeros(h_in, dtype=theano.config.floatX), name ='b_%d' % (i+len(numhid))))
            print len(numhid)+i,h_in
        self.params = []
        for W,b in zip(self.Ws,self.bs):
            self.params.append(W)
            self.params.append(b)

        for b in self.bs[len(self.Ws):]:
            self.params.append(b)

        print len(self.Ws),len(self.bs),len(self.params)

    def corrupt(self,X,batch_size=64):
        return self.theano_rng.binomial(size = (batch_size,self.numvis), n=1, p=1-self.corruption, dtype=theano.config.floatX) * X

    def encoder(self,X):
        for W,b in zip(self.Ws,self.bs):
            print W.name,b.name
            X = self.rl(T.dot(X,W)+b)
        return X 

    def decoder(self,X):
        for W,b in zip(reversed(self.Ws),self.bs[len(self.Ws):]):
            print W.name,b.name
            X = self.rl(T.dot(X,W.T)+b)
        return X

    def rl(self,X):
        """
            Rectified linear
        """
        return X*(X>0.)

    def fprop(self,X,batch_size=64):
        return self.decoder(self.encoder(self.corrupt(X,batch_size)))

    def cost(self,X,batch_size=64):
        encoding = self.encoder(self.corrupt(X,batch_size))
        reconstr = self.decoder(encoding)

        L = T.sum(0.5 * ((X - reconstr)**2), axis=1)
        return T.mean(L)

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum 

    def get_params(self):
        return numpy.concatenate([p.get_value().flatten() for p in self.params])

    def save(self, filename):
        numpy.save(filename, self.get_params())

    def load(self, filename):
        self.updateparams(numpy.load(filename))

    def get_cost_updates(self,X,learning_rate):
        self.cost(X)
        cost = self._cost
        gparams = self._grads

        updates = []
        for param, gparam in zip(self.params, gparams):
            print param,learning_rate,gparam
            updates.append((param, param - learning_rate * gparam))

        return cost, updates


if __name__ == "__main__":
    import theano.tensor as T
    from theano.tensor.shared_randomstreams import RandomStreams
    import train as sgd

    numpy_rng  = np.random.RandomState(2355)
    theano_rng = RandomStreams(2355)

#    model = ccA( numpy_rng, theano_rng, 96*96, [96*96*2,96*96,96*96/30,30], 0.01)
    from pylearn2.datasets.mnist import MNIST
    
    strain = MNIST(which_set='train')
    valid  = MNIST(which_set='test')

    print strain.X.shape[1]
    model = ccA( numpy_rng, theano_rng, strain.X.shape[1], [5000,300,30], 0.01)

    sgd.train(strain.X,valid.X,model)
