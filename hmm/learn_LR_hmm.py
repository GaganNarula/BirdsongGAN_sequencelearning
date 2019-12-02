import sys
sys.path.append(r'/home/songbird/Dropbox/Work/MDGAN_paper/code/Gagan/Train')
sys.path.append(r'/home/songbird/Dropbox/Work/MDGAN_paper/code/Gagan/encode_decode')
from hmmlearn.hmm import GaussianHMM
from utils import *
from encoder_utils import *
import argparse
import joblib
from joblib import Parallel, delayed
from time import time
from nets_16col_selu import _netE, _netG, weights_init
import pdb
import warnings
warnings.filterwarnings("ignore")

hmm_opts = {'hidden_state_size' : 100,
           'covariance_type' : 'diag', 
           'fitted_params' : 'stmc',
           'transmat_prior' : 1.,
           'n_iter' : 300,
           'tolerance' : 0.01,
           'nz' : 16, 
           'ngf' : 128,
           'nc' : 1,
            'imageH': 129,
           'imageW': 16,
           'batchsize' : 128,
           'nsamplesteps' : 128}


def create_LR_transmat(K=10):
    T = 0.5 * np.ones(K-1)
    T = np.diag(T, k = 1)
    T = 0.5 * np.eye(K,K) + T
    T[-1,-1] = 1.
    return T

def learn_single_LR_hmm_gauss(data, lengths = [], K = 10, covtype='spherical', 
                        transmat_prior=1, n_iter=1000, tol = 0.01, fit_params = 'stmc', covarweight=1.):
    model = GaussianHMM(n_components=K, covariance_type=covtype, transmat_prior=transmat_prior, \
                       random_state=0, n_iter = n_iter, covars_weight=covarweight, init_params = 'mc', \
                        params=fit_params, verbose=False, tol=tol)
    # initialize starting state as state 1 
    S = np.zeros(K)
    S[0] = 1.
    model.startprob_ = S
    # initialize transition matrix to Left Right
    model.transmat_ = create_LR_transmat(K)
    model.fit(data, lengths)
    return model


    
    
parser = argparse.ArgumentParser()
parser.add_argument('--path2mergedlist')
parser.add_argument('--path2gen')
parser.add_argument('--outpath')
parser.add_argument('--hidden_state_size', default = 100)
parser.add_argument('--covariance_type', type = str, default = 'diag')
parser.add_argument('--fit_params', type = str, default = 'tmc')
parser.add_argument('--transmat_prior', type = float, default = 1.)
parser.add_argument('--n_iter', type = int, default = 300)
parser.add_argument('--tolerance', type = float, default = 0.01)




if __name__ == '__main__':
    args = parser.parse_args()
    results = train_models(args)
    joblib.dump({'models_and_scores': results, 'opts': hmm_opts}, args.outpath+'models_and_scores.pkl')
    

    
        
        
