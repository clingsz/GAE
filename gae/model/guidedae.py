import numpy
import theano.tensor as T
import theano
from lasagne.updates import adam,adagrad,sgd,adadelta,rmsprop

class GAEOpts(object):
    def __init__(self,w=2,d=1,wr=1,act='tanh',
                 opt='adam',lam=0.5,noise_level=0.1,
                 guided=True,linear_codes=0,l2=1e-2,
                 learning_rate_ratio=1,
                 guide_layer_l2r=1,
                 verbose=1,batch_size=10,
                 learn_residual=False,max_patience=1,
                 decoder_l2r=0, train_on_enet=False,
                 check_frequency=100,epochs=5000,
                 validation_ratio=0.995,
                 robust_lambda=0,pr=None,
                 blind_epochs=500,corruption_level=0,
                 randseed=0):
                    self.w = w
                    self.d = d
                    self.wr = wr
                    self.act = act
                    self.opt = opt
                    self.lam = lam
                    self.guided = guided
                    self.linear_codes = linear_codes
                    self.noise_level = noise_level
                    self.verbose = verbose
                    self.batch_size = batch_size
                    self.l2 = l2
                    self.guide_layer_l2r = guide_layer_l2r
                    self.learn_residual=learn_residual
                    self.max_patience = max_patience
                    self.decoder_l2r = decoder_l2r
                    self.train_on_enet = train_on_enet
                    self.epochs = epochs
                    self.check_frequency = check_frequency
                    self.validation_ratio = validation_ratio
                    self.robust_lambda = robust_lambda
                    self.pr = pr
                    self.learning_rate_ratio = learning_rate_ratio
                    self.blind_epochs = blind_epochs
                    self.corruption_level = corruption_level
                    self.randseed = randseed
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_hidden,activation = T.tanh):
        self.input = input
        W,b = initialize_layer_params(rng,n_in,n_hidden,activation)
        self.W = W
        self.b = b
        if activation is None:
            self.output = (T.dot(input,self.W) + self.b)
        else:
            self.output = activation(T.dot(input,self.W) + self.b)
        self.params = [self.W, self.b]

def initialize_layer_params(rng,n_in,n_out,act):
    R = rng.uniform(
            low=-numpy.sqrt(6. / (n_in + n_out)),
            high=numpy.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)
        )
    if act == theano.tensor.nnet.sigmoid:
        R *= 4
    elif act == theano.tensor.nnet.relu:
        R = rng.rand(n_in,n_out)*numpy.sqrt(2/n_in)+0.01
    W_values = numpy.asarray(
        R,
        dtype=theano.config.floatX
    )
    W = theano.shared(value=W_values, name='W', borrow=True)
    b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
    b = theano.shared(value=b_values, name='b', borrow=True)
    return W,b

def GAE(rng,dim_in,dim_z,aeopt,r,rg,pr):
    w=aeopt.w
    depth=aeopt.d
    layer_w_ratio=aeopt.wr
    act=aeopt.act
    opt=aeopt.opt
    linear_codes=aeopt.linear_codes
    lam=aeopt.lam
    L2=aeopt.l2
    robust_lambda = aeopt.robust_lambda
    x = T.matrix("x")
    y = T.matrix("y")
    z = T.matrix("z")
    bids = T.vector("bids",dtype='int32')
#    r = T.matrix("r")
#    rg = T.matrix("rg")
    if act=='tanh':
        act = T.tanh
    elif act=='sigmoid':
        act = T.nnet.sigmoid
    elif act=='relu':
        act = T.nnet.relu
    else:
        print "No such activation ", act
    dim_out = dim_in
    
    dims = []
    for i in range(depth):
        dims.append(w*pow(layer_w_ratio,depth-i-1))
#    print dims
    params = []
    dec_params = []
    pen_sum = 0
    c = 0
    if w>0:
        hls = []
        D = len(dims)
        for i in range(D):
            if i==0:
                hls.append(HiddenLayer(rng,x,dim_in,dims[i],act))
            else:
                hls.append(HiddenLayer(rng,hls[-1].output,dims[i-1],dims[i],act))
        code_nl = hls[-1].output
        for i in range(D-1):
            hls.append(HiddenLayer(rng,hls[-1].output,dims[D-1-i],dims[D-2-i],act))
        hls.append(HiddenLayer(rng,hls[-1].output,dims[0],dim_out,act))
        hls.append(HiddenLayer(rng,hls[-1].output,dim_out,dim_out,None))
        for h in hls:
            c = c + 1
            params.extend(h.params)
            if c<D:
                pen_sum = pen_sum + T.sum(T.sqr(h.W))
            else:
                dec_params.extend(h.params)
                pen_sum = pen_sum + aeopt.decoder_l2r*T.sum(T.sqr(h.W))
        out_nl = hls[-1].output
   
    if linear_codes>0:
        h_in = HiddenLayer(rng,x,dim_in,linear_codes,None)
        code_ln = h_in.output
        h_out = HiddenLayer(rng,code_ln,linear_codes,dim_out,None)
        params.extend(h_in.params)
        params.extend(h_out.params)
        pen_sum = pen_sum + T.sum(T.sqr(h_in.W))
        out_ln = h_out.output
    
    lw = linear_codes
    if w>0 and lw>0:
        code = T.concatenate([code_nl,code_ln],axis=1)
        out = out_nl + out_ln
    if w>0 and lw==0:
        code = code_nl
        out = out_nl
    if w==0 and lw>0:
        code = code_ln
        out = out_ln
    if w==0 and lw==0:
        print "Nothing in the AE, please have at least something!"                    
    guide_layer = HiddenLayer(rng,code,w+lw,dim_z,None)
    train_guide_mse = T.mean(T.sqr(guide_layer.output-z-rg[bids,:]))
#    test_guide_mse = T.mean(T.sqr(guide_layer.output-z))
#    test_guide_mae = T.mean(T.abs_(guide_layer.output-z))
    params.extend(guide_layer.params)
    pen_sum = pen_sum + aeopt.guide_layer_l2r*T.sum(T.sqr(guide_layer.W))
    sumPr = T.sum(pr+1e-5)
    train_mse = T.mean(T.dot(T.sqr(out-x-r[bids,:]),pr))*1.0/sumPr
#    test_mse = T.mean(T.sqr(out-x))
#    test_mae = T.mean(T.abs_(out-x))
    if robust_lambda>0:
        cost = (1-lam)*train_mse + lam*train_guide_mse + L2*pen_sum + \
        1.0/robust_lambda*(T.mean(T.abs_(r))+T.mean(T.abs_(rg)))
        params.extend([r,rg])
    else:
        cost = (1-lam)*train_mse + lam*train_guide_mse + L2*pen_sum

#    guide_cost = train_guide_mse + L2*pen_sum
    reconstruct_cost = train_mse + L2*pen_sum
    
    learning_rate_ratio = aeopt.learning_rate_ratio
    
    if opt=='adam':
        updates = adam(cost, params, learning_rate=0.0002*learning_rate_ratio, beta1=0.9, beta2=0.999, epsilon=1e-08)    
    elif opt=='sgd':
        updates = sgd(cost, params, learning_rate=0.001)
    elif opt=='adagrad':
        updates = adagrad(cost, params, learning_rate=0.1, epsilon=1e-06)
    elif opt=='rmsprop':
        updates = rmsprop(cost, params, learning_rate=0.001, rho=0.9, epsilon=1e-06)
    elif opt=='adadelta':
        updates = adadelta(cost, params, learning_rate=0.1, rho=0.95, epsilon=1e-06)
    else:
        print 'no such optimizer!'

    dec_update = adam(reconstruct_cost, dec_params, learning_rate=0.0002*learning_rate_ratio, beta1=0.9, beta2=0.999, epsilon=1e-08)    
    
    train = theano.function(
        inputs=[x,y,z,bids],
        outputs=[cost],
        updates=updates,
        on_unused_input='ignore',allow_input_downcast=True)
#    train_guide = theano.function(
#        inputs=[x,y,z,bids],
#        outputs=[reconstruct_cost],
#        updates=updates,
#        on_unused_input='ignore',allow_input_downcast=True)
    train_decoder = theano.function(
        inputs=[x,y,z,bids],
        outputs=[reconstruct_cost],
        updates=dec_update,
        on_unused_input='ignore',allow_input_downcast=True)
    
    test = theano.function(
        inputs=[x,y,z,bids],
        outputs=[cost,train_mse,train_guide_mse],
        on_unused_input='ignore',allow_input_downcast=True)    
    pred = theano.function(
        inputs=[x],
        outputs=out,allow_input_downcast=True)
    predz = theano.function(
        inputs=[x],
        outputs=guide_layer.output,allow_input_downcast=True)

    encode = theano.function(
            inputs=[x],
            outputs=code,allow_input_downcast=True)
    decode = theano.function(
            inputs=[code],
            outputs=out,allow_input_downcast=True)
            
    return train,train_decoder,test,pred,encode,decode,predz

def fitGAE(x,z,vx,vy,aeopt):
    p = x.shape[1]
    n = x.shape[0]
    nv = vx.shape[0]
    x = x.astype(dtype=theano.config.floatX)
    z = z.astype(dtype=theano.config.floatX)
    R = theano.shared(value=numpy.zeros((n,p),dtype=theano.config.floatX),
                      name='R', borrow=True)
    RG = theano.shared(value=numpy.zeros((n,1),dtype=theano.config.floatX),
                      name='RG', borrow=True)
    aeopt.pr = numpy.ones([p,1])
    pr = theano.shared(value=aeopt.pr,name='PR',borrow=True)
            
    dim_z = z.shape[1]
    epochs = aeopt.epochs
    patience_now = patience = aeopt.max_patience
    rng = numpy.random.RandomState(aeopt.randseed)
    train,train_dec,test,pred,encode,decode,predz = GAE(rng,p,dim_z,aeopt,R,RG,pr)
    blst = numpy.arange(n)
    bestobj = 999999    
    batch_size = aeopt.batch_size    
    allids = numpy.arange(n)
    allvids = numpy.arange(nv)
    
    noise_level = aeopt.noise_level
    if batch_size>=n:
        batch_size = n
    n_batches = numpy.ceil(n/batch_size).astype('int')
    for i in range(epochs):
        rng.shuffle(blst)
        ns = noise_level*rng.randn(n,p)
#        mask = (rng.rand(n,p)>aeopt.corruption_level).astype('float')
#        ns = theano.shared(value=ns.astype(dtype=theano.config.floatX),name='NS')            
        for b in range(n_batches):
            bend = (b+1)*batch_size
            if bend>n:
                bend=n
            bids = blst[b*batch_size:bend]
            xi = x[bids,:]
            nsi = ns[bids,:]
            zi = z[bids,:]
            x_input = xi+nsi
            train(x_input,xi,zi,bids)
            train_dec(x_input,xi,zi,bids)
        if (i%aeopt.check_frequency==0):
            obj,vcost,gcost = test(x,x,z,allids)
            objv,dncost,prcost = test(vx,vx,vy,allvids)
            if obj<bestobj*aeopt.validation_ratio:
                bestobj = obj
                patience_now = patience
            else:
                if i>aeopt.blind_epochs:
                    patience_now = patience_now - 1
                    if patience_now<=0:
                        break
            if aeopt.verbose==1:
                print 'Ep:%d Obj:%.4f RE:%.4f PE:%.4f' % (i,obj,dncost,prcost)
    obj,vcost,gcost = test(x,x,z,allids)
    return pred,encode,decode,i,predz

#if __name__ == "__main__":
#    from trainer import TrainerOpts,make_trainer
#    from data_gen import DataOpts,load_data
#    n_codes = 7
#    dataopt = DataOpts(name='combined',test_folds=5)
#    data = load_data(dataopt)
#    ds = data.get_test()
#    pcaopt = TrainerOpts(name='PCA',n_components=n_codes)
#    pca = make_trainer(pcaopt)
#    pca.train(ds)
#    pca.test(ds)
#    aeopt = AEOpts(w=n_codes,d=2,wr=1,
#                   noise_level=0.0,lam=0.3,
#                    robust_lambda=0,l2=1e-2,decoder_l2r=0,
#                    verbose=1,top=1)
#    traineropt=TrainerOpts(name='AE',
#                       aeopt=aeopt)                                               
#    gae = make_trainer(traineropt)
#    gae.train(ds)
#    gae.test(ds)