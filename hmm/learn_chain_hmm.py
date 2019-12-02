from hmm_utils import *
import pdb
import warnings
warnings.filterwarnings("ignore")

REDUCING_K = [200, 200, 200, 200, 200, 200, 200, 200, 190, 190, 190, 180, 180, 180, 170, 
              170, 160, 160, 150, 150, 140, 140, 130, 120, 110, 100, 90, 80, 70, 60, 60, 
              60, 60, 60, 60, 60, 60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50 ]

hmm_opts = {'hidden_state_size' : 100,
           'covariance_type' : 'spherical', 
           'fitted_params' : 'stm',
           'transmat_prior' : 1.,
           'n_iter' : 300,
           'tolerance' : 0.01,
           'nz' : 32, 
           'ngf' : 256,
           'nc' : 1,
            'imageH': 129,
           'imageW': 16,
           'batchsize' : 128,
           'nsamplesteps' : 128,
           'nsamps': 10,
           'sample_var': 0.1
           }


def learn_single_hmm_gauss_with_initialization(data, lastmodel = [], lengths = [], K=10, covtype='spherical', \
                           transmat_prior=1, n_iter=1000, tol = 0.01, fit_params = 'stmc', covarweight=1.):
    if not lastmodel:
        model = GaussianHMM(n_components=K, covariance_type=covtype, transmat_prior=transmat_prior, \
                       random_state=0, n_iter = n_iter, covars_weight=covarweight, params=fit_params, 
                        init_params = 'stmc', verbose=False, tol=tol)
        model.fit(data, lengths)
        return model
    model = GaussianHMM(n_components=K, covariance_type=covtype, transmat_prior=transmat_prior, \
                       random_state=0, n_iter = n_iter, covars_weight=covarweight, params=fit_params, 
                        init_params = 'c', verbose=False, tol=tol)
    # initiliaze parameters to last model
    model.transmat_ = lastmodel.transmat_
    model.startprob_ = lastmodel.startprob_
    model.means_ = lastmodel.means_
    model.fit(data, lengths)
    return model


def create_output(model, outpath, idx, hmm_opts, netG, sequence=[], nsamps=10):
    if len(sequence)==0:
        # create samples
        seqs = [tempered_sampling(model, beta = 1., timesteps=hmm_opts['nsamplesteps'], 
                                 sample_obs=False, start_state_max=True, 
                                 sample_var = hmm_opts['sample_var']) for _ in range(nsamps)]
        seqs = [s[0] for s in seqs]
    else:
        # seqs is a single numpy array of shape [timesteps x latent_dim]
        seqs = []
        seqs.append(sequence)
        
    # create spectrogram
    spect_out = [None for _ in range(len(seqs))]
    for i in range(len(seqs)):
        spect_out[i] = decode_by_batch(seqs[i], netG,  batch_size = hmm_opts['batchsize'], \
                                 imageH=hmm_opts['imageH'], imageW=hmm_opts['imageW'], 
                                 cuda = True, get_audio = hmm_opts['get_audio'])
    audio_out = [a[1] for a in spect_out]
    spect_out = [s[0] for s in spect_out]
    # create output folder
    outputfolder = os.path.join(outpath, 'day_'+str(idx))
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    # save spectrograms
    if type(hmm_opts['hidden_state_size'])==list:
        K = hmm_opts['hidden_state_size'][idx]
    else:
        K = hmm_opts['hidden_state_size']
        
    if len(spect_out)==1:
        plt.figure(figsize=(50,10))
        plt.imshow(spect_out[0], origin='lower', cmap = 'gray')
        # real sequence
        plt.savefig(os.path.join(outputfolder, 
                             'real_sequence.eps'), dpi = 50, format='eps')
        plt.close()
    else:
        for j in range(nsamps):
            plt.figure(figsize=(50,10))
            plt.imshow(spect_out[j], origin='lower', cmap = 'gray')
            plt.savefig(os.path.join(outputfolder, 
                                 'hiddenstatesize_'+str(K)+'_sample_'+str(j)+'.eps'), 
                                     dpi = 50, format='eps')
            plt.close()
    # if audio is computed, save that 
    if hmm_opts['get_audio']:
        if len(audio_out)==1:
            save_audio_sample(audio_out[0], 
                              os.path.join(outputfolder,'real_sequence.wav'), 16000)
        else:
            for j in range(nsamps):
                save_audio_sample(audio_out[j], 
                              os.path.join(outputfolder,
                                           'hiddenstatesize_'+str(K)+'_sample_'+str(j)+'.wav'), 16000)
            
    
    
def load_z_data_and_learn(latent_loader, lastmodel, idx, netG, outpath, hmm_opts):
    '''
        Loads data for one (merge) day and learns one HMM for all the training sequences
    '''
    if type(hmm_opts['hidden_state_size'])==list:
        K = hmm_opts['hidden_state_size'][idx]
    else:
        K = hmm_opts['hidden_state_size']
    # get the latent space data
    ztrain, zval, ztest = latent_loader.get_whole_day_sequences(idx)
    # choose some ztrain for saving
    inds_to_save = np.random.choice(len(ztrain), size=hmm_opts['nsamps'])
    ztosave = [ztrain[i] for i in inds_to_save]
    # get lengths of sequences
    Ltrain = [z.shape[0] for z in ztrain]
    # train HMM
    ztrain = np.concatenate(ztrain, axis=0)
    model = learn_single_hmm_gauss_with_initialization(ztrain, lastmodel, Ltrain, K, hmm_opts['covariance_type'], 
                                                       hmm_opts['transmat_prior'], hmm_opts['n_iter'], 
                                                       hmm_opts['tolerance'], hmm_opts['fitted_params']) 
    # compute validation log likelihood 
    Lval  = [z.shape[0] for z in zval]
    zval = np.concatenate(zval, axis=0)
    val_scores = model.score(zval, Lval)
    # compute test log likelihood
    Ltest = [z.shape[0] for z in ztest]
    ztest = np.concatenate(ztest, axis=0)
    test_scores = model.score(ztest, Ltest)
    # compute train log likelihood
    train_scores = model.score(ztrain, Ltrain)
    
    # create 10 samples
    # concatenate the sequences because otherwise they are usually shorter than batch_size
    ztosave = np.concatenate(ztosave, axis=0)
    create_output(model, outpath, idx, hmm_opts, netG, [], nsamps=hmm_opts['nsamps'])
    
    # save 10 real files
    create_output(model, outpath, idx, hmm_opts, netG, ztosave, nsamps=hmm_opts['nsamps'])
    print('# trained and sampled from a model! #')
    # save interim model
    outputfolder = os.path.join(outpath, 'day_'+str(idx))
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    joblib.dump({'model':model, 'val_score': val_scores, 'test_score': test_scores},
               os.path.join(outputfolder, 'model_and_scores_day_'+str(idx)+'.pkl'))
    return model, val_scores, test_scores, train_scores




parser = argparse.ArgumentParser()
parser.add_argument('--path2mergedlist')
parser.add_argument('--path2gen')
parser.add_argument('--outpath')
parser.add_argument('--do_chaining', action = 'store_true')
parser.add_argument('--hidden_state_size', type = int, default = 100)
parser.add_argument('--covariance_type', type = str, default = 'diag')
parser.add_argument('--fit_params', type = str, default = 'stm')
parser.add_argument('--transmat_prior', type = float, default = 1.)
parser.add_argument('--n_iter', type = int, default = 300)
parser.add_argument('--tolerance', type = float, default = 0.01)
parser.add_argument('--get_audio', action = 'store_true')
parser.add_argument('--start_from', type = int, default = 0)

def train_models(args):
    
    K = args.hidden_state_size
    # load merged_list for the bird
    merged_list = joblib.load(args.path2mergedlist)
    extpath = os.path.split(args.path2mergedlist)[0]
    # create data object
    latent_loader = Latent_loader(merged_list, external_file_path = extpath)
    # load generator
    netG = load_netG(args.path2gen, ngpu = 1, nz = hmm_opts['nz'], ngf = hmm_opts['ngf'], nc = hmm_opts['nc'], cuda = True)
    netG.eval()
    Nmodels = len(merged_list) - args.start_from 
    # update hmm_opts
    if K != 0:
        hmm_opts['hidden_state_size'] = K
    elif K==0:
        hmm_opts['hidden_state_size'] = REDUCING_K
        print('\n .... REDUCING K CHOSEN! .... ')
    hmm_opts['covariance_type'] = args.covariance_type
    hmm_opts['fitted_params'] = args.fit_params
    hmm_opts['transmat_prior'] = args.transmat_prior
    hmm_opts['n_iter'] = args.n_iter
    hmm_opts['tolerance'] = args.tolerance
    hmm_opts['get_audio'] = args.get_audio
    hmm_opts['do_chaining'] = args.do_chaining
    # train models
    print('\n ..... training HMMs ..... \n')
    # check if there is a model from the previous day
    if args.do_chaining:
        oldmodelpath = os.path.join(args.outpath, 'day_'+str(args.start_from-1),'models_and_scores.pkl')
        if os.path.exists(oldmodelpath):
            m = joblib.load(oldmodelpath)
            lastmodel = m['model']
        else:
            lastmodel = []
    else:
        lastmodel = []
        
    
    results = [None for _ in range(Nmodels)]
    for k in range(args.start_from, Nmodels):
        
        if k>args.start_from and args.do_chaining:
            lastmodel = results[k-1][0]
        start = time()
        results[k] = load_z_data_and_learn(latent_loader, lastmodel, k, netG, 
                                           args.outpath,  hmm_opts)
        end = time()
        print('\n ..... %d/%d models trained, train lls: %f, val lls: %f, test: %f ..... '%(k,Nmodels,
                                                                                            results[k][3], 
                                                                                            results[k][1],
                                                                                           results[k][2]))
        print('..... time taken %5d secs .....'%(end-start))
    
    print('\n ...... finished training! ......')
    return results

if __name__ == '__main__':
    args = parser.parse_args()
    results = train_models(args)
    joblib.dump({'models_and_scores': results, 'opts': hmm_opts}, args.outpath+'models_and_scores.pkl')
    

    
        
        
