import timer
import theano
import theano.tensor as T

def iterate(sets,fcts,batch_size):
    lists = [[] for i in range(len(fcts))]
    for i in xrange(sets[0].shape[0]/batch_size+1):
        start = i*batch_size
        end = min((i+1)*batch_size,sets[0].shape[0])
        if start==end:
            break

        for l, fct in zip(lists,fcts):
            l.append(fct(*[s[start:end] for s in sets]))

    return lists

def train(train,valid,model,
          train_labels=None,valid_labels=None,
          batch_size=1,
          learning_rate=0.01,
          epochs=100,
          wait_for=10,
          epsylon=0.01,
          aug = 1.1,
          dim = 0.65,
          x = None,
          y = None,
          transformer=None):

    print "train"
    print train.shape
    print "valid"
    print valid.shape

    min_c = 1e99
    patience = wait_for

    params = []

    import numpy as np
    lr = theano.shared(value=np.cast['float32'](learning_rate),name='lr')

    if x is None:
        x = T.matrix('x')
    if train_labels is not None:
        if y is None:
            y = T.matrix('y')
        cost = model.cost(x,y)
    else:
        cost = model.cost(x)

    if hasattr(model,'updates'):
        if train_labels is not None:
            updates = model.updates(x,y,lr)
        else:
            updates = model.updates(x,lr)
    else:
        print model.params
        gparams = T.grad(cost,model.params)

        updates = []
        for param, gparam in zip(model.params, gparams):
            updates.append((param, param-lr*gparam))

    if train_labels is not None:
        train_f = theano.function([x,y], cost, updates=updates)
    else:
        train_f = theano.function([x], cost, updates=updates)

    if valid_labels is not None:
        test_f = theano.function([x,y],model.errors(x,y))
    else:
        test_f = theano.function([x],model.errors(x))#,valid))

    if transformer is not None and train_labels is not None:
        tmp_f = train_f # redirect pointer
        train_f = lambda X, y: tmp_f(*transformer.perform(X,y))
    elif transformer is not None:
        tmp_f = train_f # redirect pointer
        train_f = lambda X: tmp_f(transformer.perform(X))
       

    t = timer.Timer(epochs*(train.shape[0]/batch_size),min_time=1)
    t.start()

    print "start training"
    print batch_size

    # go through training epochs
    for epoch in xrange(epochs):
        valid_costs = []
        train_costs = []

        # go through trainng set
        if train_labels is not None:
            costs, train_costs = iterate([train,train_labels],[train_f,test_f],batch_size)
            valid_costs = iterate([valid,valid_labels],[test_f],batch_size)
        else:
            costs, train_costs = iterate([train],[train_f,test_f],batch_size)
            valid_costs = iterate([valid],[test_f],batch_size)

        v_cost = np.array(valid_costs).mean()
        t_cost = np.array(train_costs).mean()

        if min_c - v_cost > epsylon:
            lr.set_value(lr.get_value()*np.cast[theano.config.floatX](aug))
        else:
            lr.set_value(lr.get_value()*np.cast[theano.config.floatX](dim))

        if min_c - v_cost > epsylon:
            patience = wait_for
            min_c = v_cost
            params = [param.get_value() for param in model.params]
        else:
            patience -= 1

        if patience < 0:
            break

        print 'Training epoch %03d, cost %.4f %.4f patience %d' % (epoch, t_cost, v_cost, patience),lr.get_value()

    for sparam, vparam in zip(model.params,params):
        sparam.set_value(vparam)

    print ('Training took %f minutes' % (t.over() / 60.))
