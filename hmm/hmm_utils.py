import sys
sys.path.append(r'/home/songbird/Dropbox/Work/MDGAN_paper/code/Gagan/Train')
sys.path.append(r'/home/songbird/Dropbox/Work/MDGAN_paper/code/Gagan/encode_decode')
from hmmlearn.hmm import GaussianHMM
from utils import *
from encoder_utils import *
#from nets_16col_selu import _netE, _netG, weights_init
from networks_16col_v2 import _netG, _netE, _netD, weights_init, GANLoss
import argparse
import joblib
from joblib import Parallel, delayed
from time import time


def tempered_sampling(model, beta=3., timesteps=64, sample_obs=True, start_state_max=False, sample_var = 0):
    # start probability sample
    if start_state_max:
        row = np.argmax(model.startprob_)
    else:
        # choose a start state
        p = np.exp(beta * np.log(model.startprob_))
        p /= np.sum(p)
        s0 = np.random.multinomial(1,p)
        row = np.where(s0==1.)[0][0]
    s0 = row
    states = np.zeros((timesteps),dtype='int64')
    obs = np.zeros((timesteps, model.means_.shape[-1]))
    for i in range(timesteps):
        # extract the correct row from the transition matrix
        a = model.transmat_[row,:]
        # make the gibbs probability vector
        p = np.exp(beta * np.log(a))
        p /= np.sum(p)
        # sample from it 
        s = np.random.multinomial(1,p)
        row = np.where(s==1.)[0][0]
        states[i] = row
        # sample from the corresponding distribution in the model
        mean_i = model.means_[row]
        sigma_i = model.covars_[row]
        if sample_obs:
            # sample an observation 
            if sample_var == 0:
                obs[i] = np.random.multivariate_normal(mean_i,sigma_i,size=1)
            else:
                sigma_in = sample_var*np.eye(mean_i.shape[0])
                obs[i] = np.random.multivariate_normal(mean_i,sigma_in,size=1)
        else:
            obs[i] = mean_i
    return obs, states, s0


class Latent_loader(object):
    def __init__(self, merged_list, external_file_path = [], train_val_split=0.8, val_test_split=0.5):
        # split every element of merge_list into train, val and test
        self.id_lists_per_day = []
        for m in merged_list:
            id_list_train, id_list_valtest = split_list(m, train_val_split)
            id_list_val, id_list_test = split_list(id_list_valtest, val_test_split)
            self.id_lists_per_day.append([id_list_train, id_list_val, id_list_test])
        self.days = np.arange(len(merged_list))
        self.external_file_path = external_file_path
        
    def __getitem__(self, index):
        if self.external_file_path:
            birdname = os.path.basename(index['filepath'])
            f = h5py.File(os.path.join(self.external_file_path, birdname),'r')
        else:
            f = h5py.File(index['filepath'], 'r') 
        z = np.array(f.get(index['within_file']))
        f.close()
        return z
    
    def get_whole_day_sequences(self, day):
        id_list_train, id_list_val, id_list_test = self.id_lists_per_day[day]
        Ztrain = [None for i in range(len(id_list_train))]
        Zval = [None for i in range(len(id_list_val))]
        Ztest = [None for i in range(len(id_list_test))]
        for i in range(len(id_list_train)):
            Ztrain[i] = self.__getitem__(id_list_train[i])
        for i in range(len(id_list_test)):
            Ztest[i] = self.__getitem__(id_list_test[i])
        for i in range(len(id_list_val)):
            Zval[i] = self.__getitem__(id_list_val[i])
        return Ztrain, Ztest, Zval
    
    def get_N_random_training_samples(self, day, N=1):
        id_list_train, _ = self.id_lists_per_day[day]
        inds = np.random.choice(np.arange(len(id_list_train)), size=N, replace=False)
        id_list_out = [id_list_train[i] for i in inds]
        Zsample = [self.__getitem__(i) for i in id_list_out]
        return Zsample

    
class songbird_data_sample(object):
    def __init__(self, path2idlist, external_file_path):
        with open(path2idlist, 'rb') as f:
            self.id_list = pickle.load(f)
        self.external_file_path = external_file_path
    def __len__(self):
        # total number of samples
        return len(self.id_list)
    
    def __getitem__(self, index):
        # load one wav file and get a sample chunk from it
        ID = self.id_list[index]
        age_weight = ID['age_weight']
        # this 'ID' is a dictionary containing several fields,
        # use field 'filepath' and 'within_file' to get data
        if self.external_file_path:
            birdname = os.path.basename(ID['filepath'])
            f = h5py.File(os.path.join(self.external_file_path, birdname),'r')
        else:
            f = h5py.File(ID['filepath'], 'r') 
        x = np.array(f.get(ID['within_file']))
        f.close()
        return x, age_weight
    
    def get_contiguous_minibatch(self, start_idx, mbatchsize=64):
        ids = np.arange(start=start_idx, stop=start_idx+mbatchsize)
        X = [self.__getitem__(i)[0] for i in ids]
        return X

    
def load_netG(netG_file_path, ngpu = 1, nz = 16, ngf = 128, nc = 1, cuda = False):
    netG = _netG(ngpu, nz, ngf, nc)
    netG.apply(weights_init)
    netG.load_state_dict(torch.load(netG_file_path))

    if cuda:
        netG.cuda()
    netG.mode(reconstruction=True)
    return netG


def decode_by_batch(zhat, netG,  batch_size = 64, imageH=129, imageW=16, cuda = False, get_audio=False):
    if type(zhat)==np.ndarray:
        zhat = torch.from_numpy(zhat).float()
        zhat = zhat.resize_(zhat.shape[0], zhat.shape[1], 1, 1)
    if cuda:
        zhat = zhat.cuda()
    out_shape = [imageH, imageW]
    reconstructed_samples = []
    recon_audio = []
    # do inference in batches
    nbatches = round(zhat.size(0)/batch_size)
    i = 0
    for n in range(nbatches):
        reconstruction = netG(zhat[i:i+batch_size])
        i += batch_size
        for k in range(reconstruction.data.cpu().numpy().shape[0]):
            reconstructed_samples.append(reconstruction.data[k,:,:,:].cpu().numpy().reshape(out_shape))
            if get_audio:
                recon_audio.append(inverse_transform(reconstruction.data[k,:,:,:].cpu().numpy().reshape(out_shape), N=500))
    reconstructed_samples = np.concatenate(reconstructed_samples, axis=1)
    if get_audio:
        recon_audio = np.concatenate(recon_audio, axis=1)
        recon_audio = lc.istft(reconstructed_samples)*2
    return rescale_spectrogram(reconstructed_samples), recon_audio

from scipy.stats import entropy
def average_entropy(T):
    E = []
    for i in range(T.shape[0]):
        tmp = entropy(T[i], base = 2)
        #tmp = 0.
        #for j in range(T.shape[1]):
            #if T[i,j] > 0.:
            #    tmp += -T[i,j]*np.log(T[i,j])
        E.append(tmp)
    E = np.array(E)
    return E.mean()


def load_multiple_models(birdpath, days):
    models = [None for _ in range(len(days))]
    for i,d in enumerate(days):
        dirpath = os.path.join(birdpath,'day_'+str(d))
        fls = glob.glob(dirpath+'/model*')[0]
        m = joblib.load(fls)
        models[i] = m['model']
    return models


def KLdiv_bw_2multGaussians(p, q):
    '''
    Here p and q are dictionaries
    '''
    mu_p = np.array(p[0])
    mu_q = np.array(q[0])
    var_p = np.array(p[1])
    var_q = np.array(q[1])
    
    var_q_inv = np.linalg.inv(var_q)
    det_var_p = np.linalg.det(var_p)
    det_var_q = np.linalg.det(var_q)
    varprod = np.matmul(var_q_inv, var_p)
    
    term1 = np.dot((mu_p - mu_q).T, np.dot(var_q_inv, (mu_p - mu_q)))
    term2 = np.trace(varprod)
    term3 = np.log(det_var_p / det_var_q)
    term4 = len(mu_p)
    
    return term1+term2-term3-term4