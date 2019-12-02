import sys
sys.path.append(r'/home/songbird/Dropbox/Work/MDGAN_paper/code/Gagan/Train')
sys.path.append(r'/home/songbird/Dropbox/Work/MDGAN_paper/code/Gagan/encode_decode')
import numpy as np
from hmmlearn.hmm import GaussianHMM
import argparse
import joblib
from joblib import Parallel, delayed
from time import time
import pdb
import warnings
warnings.filterwarnings("ignore")

def get_samples(model, nsteps = 100, nsamps = 10000, random_state = 0):
    #samples = Parallel(n_jobs=-2)(delayed(
    #    model.sample)(nsteps, random_state) for i in range(nsamps))
    #samples = [s[0] for s in samples]
    samples = Parallel(n_jobs=-2)(delayed(tempered_sampling)(model, 1., nsteps, False, True) for i in range(nsamps))
    samples = [s[0] for s in samples]
    return samples

def get_scores(model, samples):
    s = Parallel(n_jobs=-2)(delayed(model.score)(s) for s in samples)
    return np.mean(s)

def tempered_sampling(model, beta=3., timesteps=64, sample_obs=True, start_state_max=False):
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
            obs[i] = np.random.multivariate_normal(mean_i,sigma_i,size=1)
        else:
            obs[i] = mean_i
    return obs, states, s0

def compute_JR_dists(args):
    # load merged_list for the bird
    #merged_list = joblib.load(args.path2mergedlist)
    # create data object
    #latent_loader = Latent_loader(merged_list)
    # load models and opts
    models_and_scores = joblib.load(args.path2models)
    MS = models_and_scores['models_and_scores']
    OPTS = models_and_scores['opts']
    # how many models?
    L = len(MS)
    # which one is tutor model? (last one or an another one)
    tutmodel = MS[-1][0]
    # load data of this model 
    #ztrain_tut, zval_tut, ztest_tut = latent_loader.get_whole_day_sequences(L)
    #Lval_tut  = [z.shape[0] for z in zval_tut]
    #Ltest_tut  = [z.shape[0] for z in zval_test]
    
    # score of this models data under itself
    #tutscore_val = [(1/l)*tutmodel.score(x) for x,l in zip(zval_tut, Lval_tut)]
    #tutscore_test = [(1/l)*tutmodel.score(x) for x,l in zip(zval_tut, Lval_tut)]
    #tutscore_val = np.mean(tutscore_val)
    
    # get several sample trajectories from tutormodel
    tutsamples = get_samples(tutmodel, args.nsteps, args.nsamps, random_state = 0)
    # compute tutor model score
    tuttutscore = get_scores(tutmodel, tutsamples)
    
    KLD_tut_pup = np.zeros(L)
    KLD_pup_tut = np.zeros(L)
    JSD = np.zeros(L)
    
    for i in range(L):
        # get the lmodel
        pupmodel = MS[i][0]
        # get pup samples
        pupsamples = get_samples(pupmodel, args.nsteps, args.nsamps, random_state = 0)
        # get score of tut data under this model Q(x|pup_model)
        puptutscore = get_scores(pupmodel, tutsamples)
        # get score of pup samples under pup model
        puppupscore = get_scores(pupmodel, pupsamples)
        # get score of pup data under tutor model 
        tutpupscore = get_scores(tutmodel, pupsamples)
        
        # compute divergences
        # DKL(tut | pup)
        KLD_tut_pup[i] = tuttutscore - puptutscore
        # DKL (pup | tut)
        KLD_pup_tut[i] = puppupscore - tutpupscore
        # Jenson Shannon
        JSD[i] = 0.5*(KLD_tut_pup[i]) + 0.5*(KLD_pup_tut[i])
        
        print('\n .... %d/%d models evaluated .... '%(i,L))
        print('...... KLD_tut_pup: %f  , KLD_pup_tut: %f, JSD: %f'%(KLD_tut_pup[i], KLD_pup_tut[i], JSD[i]))
    return KLD_tut_pup, KLD_pup_tut, JSD

parser = argparse.ArgumentParser()
parser.add_argument('--path2models')
parser.add_argument('--nsteps', type = int, default = 100)
parser.add_argument('--outpath')
parser.add_argument('--nsamps', type = int, default = 10000)

if __name__ == '__main__':
    args = parser.parse_args()
    results = compute_JR_dists(args)
    joblib.dump(results,args.outpath + 'KLD_JSD_divergences.pkl')


        