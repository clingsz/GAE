import numpy
from sklearn.model_selection import KFold
from sklearn import linear_model

# a common space of data options #
class DataOpts(object):
    def __init__(self,name='combined',test_folds=5,
                 valid_folds=3,random_seed=0,
                 standardize_beforehand=False,
                 standardize_ontraining=True,
                 test_fold_id=0,valid_fold_id=0,
                 noise_level=0.5):
        self.test_folds = test_folds
        self.valid_folds = valid_folds
        self.random_seed = random_seed
        self.standardize_beforehand = standardize_beforehand
        self.standardize_ontraining = standardize_ontraining
        self.test_fold_id = test_fold_id
        self.valid_fold_id = valid_fold_id
        self.name = name
        self.noise_level = noise_level

####################################################
#   General data processing functions
####################################################

# extract raw data
def get_raw_combined_dataset():
    input_file = 'data/combine_031517.csv'
    with open(input_file) as f:
        ncols = len(f.readline().split(','))
    D = numpy.loadtxt(input_file,delimiter=',',skiprows=1, usecols=range(1,ncols))
    Ds = numpy.loadtxt(input_file,delimiter=',',dtype='string',skiprows=0)
    feature_names = Ds[0,1:] # first column is patient_id
    demo_names = ['mfs','source','cmv','ebv','gender','age','bmi']
    demo_ids = [[i] for i in range(len(demo_names))]
    demo = {}
    for did,name in zip(demo_ids,demo_names):
        demo[name] = D[:,did]
    demo['patient_id'] = Ds[1:,0]
    flow_ids = range(7,32)
    chex = D[:,33:37]
    chex4 = chex[:,-1:]
    cyto_ids = [32] + range(37,len(feature_names))
    D_flow = D[:,flow_ids]
    D_cyto = D[:,cyto_ids]
    cyto_names = feature_names[cyto_ids]
    flow_names = feature_names[flow_ids]
    data = {}
    demo['chex4'] = chex4
    data['demo'] = demo
    data['D_flow'] = D_flow
    data['D_cyto'] = D_cyto
    data['flow_names'] = flow_names
    data['cyto_names'] = cyto_names
    return data

# process the raw data        
def get_processed_data(correction=True):
    data = get_raw_combined_dataset()
    D_flow = data['D_flow']
    D_cyto = data['D_cyto']
    lD_flow = numpy.log(D_flow+1)
    lD_cyto = numpy.log(D_cyto+1)
    lD_cyto = remove_cyto_outliers(lD_cyto) 
    
    demo = data['demo']    
    demo_nms = ['source','chex4','age','gender']
    if correction:
        r_cyto = numpy.concatenate([demo[did] for did in demo_nms],axis=1)
        r_cyto[:,1] = numpy.log(r_cyto[:,1]+1)
        lD_cyto = regress_out(r_cyto,lD_cyto,[0,1])
    
        demo_nms = ['source','age','gender']
        r_flow = numpy.concatenate([demo[did] for did in demo_nms],axis=1)
        lD_flow = regress_out(r_flow,lD_flow,[0])
    data['flow'] = lD_flow
    data['cyto'] = lD_cyto
    return data

# get all the data with standardized parameters
def get_training_data():
    data = get_processed_data()
    x,_,_ = make_standardize(data['cyto'])
    y,y_mu,y_sig = make_standardize(data['demo']['age'])    
    train = {'X':x, 'Y':y, 'MU_Y':y_mu, 'SIG_Y':y_sig}
    return train

# load immune data
def load_immune(folds=0,fold_id=0):
    if folds==0:
        train = get_training_data()
        data = {'x_train':train['X'],
                'y_train':train['Y']}
    elif folds>0:                        
        dataopt = DataOpts()
        dataopt.test_folds = folds
        combine_dataset = Combine_dataset(dataopt)
        cv_task = combine_dataset.get_cv_task(fold_id).get_test()
        data = {'x_train':cv_task[0][0],
                'y_train':cv_task[0][1],
                'x_test': cv_task[1][0],
                'y_test': cv_task[1][1]}        
    return data

def load_data(dataopt=None):
    if dataopt is None:
        dataopt = DataOpts()
    task_id = dataopt.test_fold_id
    combine_dataset = Combine_dataset(dataopt)
    return combine_dataset.get_cv_task(task_id)

####################################################
#  internal used functions
####################################################

def regress_out(x,y,rmvid=None):
    if rmvid is None:
        rmvid = 0
    lst = numpy.where(~numpy.isnan(y[:,0]))[0]
    y_all = y.copy()
    x = x[lst,:]
    y = y[lst,:]
   
    regr = linear_model.LinearRegression()
    regr.fit(x,y)
    z = numpy.zeros(regr.coef_.shape)
    z[:,rmvid] = regr.coef_[:,rmvid]
    regr.coef_ = z
    y_res = y - regr.predict(x)
    y_all[lst,:] = y_res
    return y_all

class Dataset():
    def __init__(self,name,train_x,train_y,test_x,test_y,x_labels=None,y_labels=None,dataopt=None):
        if dataopt is None:
            dataopt = DataOpts()
        self.dataopt = dataopt
        self.name = name
        self.train_x = train_x.astype('float32')
        self.train_y = train_y.astype('float32')
        self.test_x = test_x.astype('float32')
        self.test_y = test_y.astype('float32')
        self.x_labels = x_labels
        self.y_labels = y_labels
        self.all_x = numpy.concatenate([train_x,test_x],axis=0)
        self.all_y = numpy.concatenate([train_y,test_y],axis=0)
        all_x,mus,all_stds = make_standardize(self.all_x)
        all_y,musy,all_stdsy = make_standardize(self.all_y)
        self.all_x_noise = add_noise_gaussian(self.all_x,all_stds,dataopt.noise_level)
        self.train_x_noise = add_noise_gaussian(train_x,all_stds,dataopt.noise_level)
        self.test_x_noise = add_noise_gaussian(test_x,all_stds,dataopt.noise_level)
        self.partition_training_data()
    def partition_training_data(self):
        folds=self.dataopt.valid_folds
        rand_state=self.dataopt.random_seed
        kf = KFold(n_splits=folds, shuffle=True, random_state=rand_state)
        kf.get_n_splits(self.train_x)
        self.train_ids = []
        self.valid_ids = []
        for train,valid in kf.split(self.train_x):
            self.train_ids.append(train)
            self.valid_ids.append(valid)
    def apply_training_standardization(self,train_data,valid_data):
        new_train = []
        new_valid = []
        self.transformation=[]
        for a,b in zip(train_data,valid_data):
            astd,ms,vs = make_standardize(a)
            self.transformation.append([ms,vs])
            bstd = apply_standardize(b,ms,vs)
            new_train.append(astd)
            new_valid.append(bstd)
        return new_train,new_valid
            
    def get_test(self):
        train_data = (self.train_x,self.train_y,self.train_x_noise)
        test_data = (self.test_x,self.test_y,self.test_x_noise)
        if self.dataopt.standardize_ontraining:
            train_data,test_data = self.apply_training_standardization(train_data,test_data)
        return train_data,test_data
        
    def get_validation_partition(self,fold=0):
        train_ids = self.train_ids
        valid_ids = self.valid_ids
        tids = train_ids[fold]
        vids = valid_ids[fold]
        train_data = (self.train_x[tids,:],self.train_y[tids,:],self.train_x_noise[tids,:])
        valid_data = (self.train_x[vids,:],self.train_y[vids,:],self.test_x_noise[tids,:])
        if self.dataopt.standardize_ontraining:
            train_data,valid_data = self.apply_training_standardization(train_data,valid_data)
        return train_data,valid_data
    def get_all(self):
        train_data = (self.all_x,self.all_y)
        train_data,valid_data = self.apply_training_standardization(train_data,train_data)
        return train_data,valid_data

def make_standardize(x):
    n,p = x.shape
    X = numpy.copy(x)
    mus = []
    stds = []
    for i in range(p):
        x = X[:,i]
        mu = numpy.mean(x)
        st = numpy.std(x)
        x = x - mu
        x = x / st
        X[:,i] = x
        mus.append(mu)
        stds.append(st)
    return X,mus,stds

def apply_standardize(x,mus,stds):
    n,p = x.shape
    X = numpy.copy(x)
    for i in range(p):
        x = X[:,i]
        mu = mus[i]
        st = stds[i]
        x = x - mu
        x = x / st
        X[:,i] = x
    return X

class Data_gen(object):
    def __init__(self,name,x,y,xlb,ylb,dataopt):
        self.name = name
        self.x = x
        self.y = y
        self.x_labels = xlb
        self.y_labels = ylb
        self.dataopt = dataopt
        self.partition_cv_ids()
        
    def partition_cv_ids(self):
        random_state = self.dataopt.random_seed
        folds = self.dataopt.test_folds
        kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        kf.get_n_splits(self.y)
        self.train_ids = []
        self.valid_ids = []
        for train,valid in kf.split(self.y):
            self.train_ids.append(train)
            self.valid_ids.append(valid)        
            
    def get_task_use_ids(self,task_name,train_ids,test_ids):
        return Dataset(name=task_name,
                       train_x = self.x[train_ids,:],
                        train_y = self.y[train_ids,:],
                        test_x = self.x[test_ids,:],
                        test_y = self.y[test_ids,:],
                        x_labels = self.x_labels,
                        y_labels = self.y_labels,dataopt=self.dataopt)
                        
    def get_cv_task(self,fold=0):
        train_ids = self.train_ids
        valid_ids = self.valid_ids
        tids = train_ids[fold]
        vids = valid_ids[fold]
        return self.get_task_use_ids(self.name+'-cv-'+str(fold),tids,vids)

def identify_binary(D):
    p = D.shape[1]
    binary = numpy.zeros([p,])
    for i in range(p):
#            print i,f
        if len(numpy.unique(D[:,i]))==2:
            binary[i] = 1
    return binary

def display_features(feature_names):
    for i,f in zip(range(len(feature_names)),feature_names):
        print i,f

def remove_cyto_outliers(lD_cyto):
        X = lD_cyto
        Xs,mus,stds = make_standardize(X)
        upper = numpy.asarray(mus)+numpy.asarray(stds)*3
        lower = numpy.asarray(mus)-numpy.asarray(stds)*3
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i,j]>upper[j]: X[i,j] = upper[j]
                if X[i,j]<lower[j]: X[i,j] = lower[j]
        return X       
            
class Combine_dataset(Data_gen):
    def __init__(self,dataopt):
        self.dataopt = dataopt
        data = get_processed_data()
        self.x = data['cyto']
        self.y = data['demo']['age']
        self.x_labels = data['cyto_names']
        self.y_labels = ['age']
        self.name = 'combined'
        self.partition_cv_ids()
        
def add_noise_gaussian(x,sigma=0.1,noise_level=0.5):
    rng = numpy.random.RandomState(0)
    if len(sigma)>1:
        sv = numpy.asarray(sigma).reshape([1,len(sigma)])
        sigma = numpy.repeat(sv,x.shape[0],axis=0)
    x_noisy = x + rng.randn(x.shape[0],x.shape[1])*sigma*noise_level
    return x_noisy

def test():
    data = get_processed_data()
    print data['demo']['patient_id']

if __name__=='__main__':
    data = get_training_data()

