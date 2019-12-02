import numpy as np
import os
import glob
from scipy.io import wavfile
import h5py
import torch
from torch.utils import data
from random import shuffle

def save_image(image,save_path,save_idx,amplitude_only=False):
    if amplitude_only:
        np.save(join(save_path, str(save_idx) + '.npy'), image[:,:,0])
    else:
        np.save(join(save_path, str(save_idx) + '.npy'), image)
        
        
def extract_pseudoage_from_folder_names(folders):
    if len(folders)<5: 
        age = [100 for i in range(len(folders))]
        return age
    
    flds = sorted(folders)
    age = []
    for (i,f) in enumerate(flds):
        if 'tutor' in f:
            age.append(100)
        else:
            age.append(i)
    return age


def save_image_and_age(image, save_path, age, save_idx, amplitude_only=False):
    if amplitude_only:
        np.save(join(save_path, str(save_idx) + '_age_' + str(age) + '.npy'), image[:,:,0])
    else:
        np.save(join(save_path, str(save_idx) + '_age_' + str(age) + '.npy'), image)
        
        
def generate_dataset(base_path, out_path, nfft=256, max_num=None, downsample_factor=None, shuffle_folds = False, \
                     standardize=False, save_together=False, add_age = True, save_idx=0):
    """
    :param base_path: Path to the directory containing directories that contain raw audio files
    :param out_path: Where to save the genearted dataset
    :param nfft: parameter of STFT
    :param max_num: Max number of samples to genearte (if None, then genearet with entire set)
    :param downsample_factor: The original data is sampled at 32kHz, if  for exampel, downsample_factor is 2, then the
                              STFT is computed on the audio after resampling to 16kHz
    :param shuffle_folds: shuffle folder order
    :param standardize: divide wav file trace by its standard deviation
    :param save_together: save all files in this folder as one long image
    :param add_age: whether to add an integer between [0, 100] expressing Pseudo age (0: start of recordings, 100: adult)
    :param save_idx: additional number added to make sure there are no naming conflicts with other files
    """
    
    folders = os.listdir(base_path)
    age = extract_pseudoage_from_folder_names(folders)
    
    if shuffle_folds:
        # to make sure juvenile and adult orders are shuffled
        folders = shuffle(folders) 
        
    for (k,folder) in enumerate(folders):
        songs, fs, filenames = load_from_folder(base_path, folder)
        if songs:
            if downsample_factor:
                songs = [downsample(i, downsample_factor) for i in songs]
                if standardize:
                    songs = [s/np.std(s) for s in songs]
            ims = [to_image(i,nfft) for i in songs]
            if save_together:
                print('')
            else:
                for (j,im) in enumerate(ims):
                    if add_age:
                        save_image_and_age(im, out_path, filenames[j]+str(save_idx), age[k], amplitude_only=False)
                    else:
                        save_image(im, out_path, filenames[j]+str(save_idx), amplitude_only = False)
                    save_idx+=1
                    if max_num:
                        if save_idx>=max_num:
                            return
                        
                        
def load_from_folder(base_path,folder_path):
    dd = os.path.join(base_path,folder_path,'songs')
    if os.path.exists(dd):
        files = os.listdir(dd)
        foldnam = dd
    else:
        files = os.listdir(join(base_path,folder_path))
        foldnam = join(base_path,folder_path)
    files = [i for i in files if '.wav' in i.lower()]
    data = []
    fs = 0
    for i in files:
        samples, fs = load_wav_file(join(base_path,folder_path,'songs',i))
        data.append(samples)
    return data,fs,files



def load_wav_file(path):
    fs,wf = wavfile.read(path)
    if wf.dtype == 'int16':
        nb_bits = 16 # -> 16-bit wav files
    elif wf.dtype == 'int32':
        nb_bits = 32 # -> 32-bit wav files
    max_nb_bit = float(2 ** (nb_bits - 1))
    samples = wf / (max_nb_bit + 1.0) 
    return samples, fs
    
    

def generate_dataset_by_day(base_path, out_path, nfft, amplitude_only=False, downsample_factor=None, normalize_amplitude=False,
                            train_val_split=False, name='', save_single_files = True, hdf = True):
    
    save_idx=0
    folders = os.listdir(base_path)
    days = []
    for f in folders:
        try:
            day = dt.strptime('-'.join(f.split('-')[:3]), '%Y-%m-%d')
            days.append(day)
        except:
            continue
    days.sort()
    j = 0
    
    os.makedirs(out_path, exist_ok = True)
    if hdf:
        h5f = h5py.File(join(out_path, name+'_all.h5'), 'w')
    for i in range(len(folders)):
        try:
            day = dt.strptime('-'.join(folders[i].split('-')[:3]), '%Y-%m-%d')
            num = (day-days[0]).days
            out_path_full = join(out_path,"day%d_%s"%(num,name),"images")
        except:
            out_path_full = join(out_path,folders[i]+'_'+name,"images")
        if not hdf:
            os.makedirs(out_path_full, exist_ok=True)
            
        songs, fs, _ = load_from_folder(base_path, folders[i])
        if not songs:
            continue
        if downsample_factor:
            songs = [downsample(i, downsample_factor) for i in songs]
        if normalize_amplitude:
            songs = [s/np.max(s) for s in songs]
        ims = [to_image(i,nfft) for i in songs]
        for k in range(len(ims)):
            if len(ims[k].shape) < 3:
                del ims[k]
                
        if train_val_split:
            
            im_train,im_test = train_test_split(ims,test_size=0.1)
            im_val,im_test = train_test_split(im_test,test_size=0.5)
            if not hdf:
                out_path_test = out_path_full.replace(out_path,out_path+'_test/')
                out_path_val = out_path_full.replace(out_path,out_path+'_val/')
                os.makedirs(out_path_test, exist_ok=True)
                os.makedirs(out_path_val, exist_ok=True)
                if save_single_files:
                    for im in im_train:
                        save_image(im, out_path_full, save_idx, amplitude_only=amplitude_only)
                        save_idx+=1
                    for im in im_val:
                        save_image(im, out_path_val, save_idx, amplitude_only=amplitude_only)
                        save_idx+=1
                    for im in im_test:
                        save_image(im, out_path_test, save_idx, amplitude_only=amplitude_only)
                        save_idx+=1
                else:
                    # save all images as list 
                    tf = join(out_path_full, 'all_files' + '.xz')
                    joblib.dump({'meta': meta_data, 'data': im_train}, tf, compress=9)
                    tf = join(out_path_val, 'all_files' + '.xz')
                    joblib.dump({'meta': meta_data, 'data': im_train}, tf, compress=9)
                    tf = join(out_path_test, 'all_files' + '.xz')
                    joblib.dump({'meta': meta_data, 'data': im_train}, tf, compress=9)

                    #np.save(join(out_path_full, 'all_files' + '.npy'), im_train)
                    #np.save(join(out_path_train, 'all_files' + '.npy'), im_val)
                    #np.save(join(out_path_test, 'all_files' + '.npy'), im_test)
                    # save as hdf
            else:
                h5f.create_dataset('day'+str(num)+'_train', data=im_train, compression='gzip', compression_opts = 9)
                h5f.create_dataset('day'+str(num)+'_val', data=im_val, compression='gzip', compression_opts = 9)
                h5f.create_dataset('day'+str(num)+'_test', data=im_test, compression='gzip', compression_opts = 9)
        else:
            if not hdf:
                if save_single_files:
                    for im in ims:
                        save_image(im,out_path_full,save_idx,amplitude_only=amplitude_only)
                        save_idx+=1
                else:
                    # save all images as list 
                    #np.save(join(out_path_full, 'all_files' + '.npy'), im)
                    tf = join(out_path_full, 'all_files' + '.xz')
                    joblib.dump({'meta': meta_data, 'data': ims}, tf, compress=6)
            else:
                h5f.create_dataset('day'+str(num)+'_all', data=ims, compression='gzip', compression_opts = 9)
        j+=1
        if hdf:
            h5f.close()
        print('\n .... Folder %s done .... %d/%d left ...\n'%(folders[i],j,len(folders)))


def create_bird_spectrogram_hdf(birdname, birddatapath, outpath, downsample_factor=2, nfft=256, standardize=False, 
                                compress_type='gzip', compress_idx=9):
    '''
    Creates HDF file for this bird. Each day of recording is a Group, 
    and the spectrogram of each wav file is a Dataset in a Group.
    Attributes are added for each group.
    
    Params
    ------
    
    '''
    start = time()
    # create hdf file for this bird in the outpath folder
    birdfile = h5py.File(os.path.join(outpath, birdname), 'w')
    try:
        # get all folder names
        folders = os.listdir(birddatapath)
        # determine pseudo age
        ages = extract_pseudoage_from_folder_names(folders)
        if len(ages)==1:
            ages = [ages[0] for i in range(len(folders))]
        # go through each folder, load files, downsample (optional), standardize (optional) and create STFTs
        save_idx = 0
        for (k,fold) in enumerate(folders):
            # create group
            d = birdfile.create_group(fold)
            d.attrs['CLASS'] = 'STFT_with_Magnitude_and_Phase(2nd index in last dimension)'
            d.attrs['DTYPE'] = 'float32'
            d.attrs['PSEUDO_AGE'] = ages[k]
            d.attrs['nfft'] = nfft
            d.attrs['downsamplerate'] = downsample_factor
            d.attrs['standardized'] = standardize
            songs, fs, filenames = load_from_folder(birddatapath, fold)
            if songs:
                if downsample_factor:
                    songs = [downsample(i, downsample_factor) for i in songs]
                    if standardize:
                        songs = [s/np.std(s) for s in songs]
                ims = [to_image(i,nfft) for i in songs]
                for (i,im) in enumerate(ims):
                    d.create_dataset(filenames[i].split('.')[0]+'_'+str(save_idx)+'_'+str(ages[k]), data=im, \
                                     compression = compress_type, compression_opts = compress_idx)
        birdfile.close()
    except:
        birdfile.close()
    end = time()
    print('..... bird %s finished in %5d secs.....'%(birdname, end-start)) 
    
    
def make_IDs(birdhdfpath, birdname, id_list, age_weight_list, cnt):
    ''' 
    Makes a list of metadata for each wav file spectrogram in a single
    bird's hdf dataset
    '''
    birdfile = h5py.File(birdhdfpath, 'r')
    all_grps = list(birdfile.items())
    # cycle over days for this bird
    for g in all_grps:
        day = birdfile.get(g[0])
        day_wav_list = list(day.items())
        # cycle over files
        for f in day_wav_list:
            duration = np.array(f[1]).shape[1]
            age_weight = 1 - float(f[0].split('_')[-1])/100
            if age_weight==0.0:
                # for adults age is automaticaly 100, so weight 
                # be 0 which is undesirable
                age_weight = 0.1
            age_weight_list.append(age_weight)
            id_list.append({'id':cnt, 'birdname': birdname, 'filepath': birdhdfpath, \
                            'within_file': '/'+g[0]+'/'+f[0], 'age_weight': age_weight,
                           'duration':duration})
            cnt += 1
    birdfile.close()
    return id_list, age_weight_list, cnt


def make_ID_list(path2birds):
    birds = os.listdir(path2birds)
    id_list = []
    age_weight_list = []
    cnt = 0
    for (i,b) in enumerate(birds):
        id_list, age_weight_list, cnt = make_IDs(os.path.join(path2birds, b), b, id_list, age_weight_list, cnt)
        print('..... %d of %d birds indexed .....'%(i, len(birds)))
    return id_list, age_weight_list, cnt


def get_random_spectrogram_sample_from_bird(birdhdfpath, folder_names, Nsamps = 1):
    birdfile = h5py.File(birdhdfpath, 'r')
    base_items = list(birdfile.items())
    directory_names = [b[0] for b in base_items]
    # search for folder name and get random samples
    out = [None for k in range(len(folder_names))]
    for j,f in enumerate(folder_names):
        for i,d in enumerate(directory_names):
            if f in d:
                sequencelist = list(birdfile.get(base_items[i][0]).items())
                idx = np.random.choice(len(sequencelist),size=Nsamps,replace=False)
                out[j] = [None for m in range(Nsamps)]
                for k in range(len(idx)):
                    out[j][k] = np.array(birdfile.get('/'+d+'/'+sequencelist[idx[k]][0]))
    birdfile.close()
    return out



def create_bird_latentspace_hdf(birdname, birdspecpath, outpath, netE, transform_sample=True, batch_size=64, \
                                imageH=129, imageW=16, return_tensor=False, cuda=True, compress_type='gzip', compress_idx=9):
    '''
    Creates HDF file for this bird. Each day of recording is a Group, 
    and the latent space of each spectrogram is a Dataset in a Group.
    Attributes are added for each group
    '''
    start = time()
    # create hdf file for this bird in the outpath folder
    birdoutfile = h5py.File(os.path.join(outpath, birdname), 'w')
    # read this birds spectrogram file
    birdinfile = h5py.File(os.path.join(birdspecpath, birdname), 'r')
    
    # get all groups (folders)
    grps  = list(birdinfile.items())
    folds = [g[0] for g in grps]
    for f in folds:
        foldata = birdinfile.get(f)
        pseudo_age = foldata.attrs['PSEUDO_AGE']
        # create a group in the latent file with the same name
        fout = birdoutfile.create_group(f)
        fout.attrs['PSEUDO_AGE'] = pseudo_age
        # get list of spectrogram items for this folder
        spectrogram_list = list(foldata.items())
        # get only the names
        spectrogram_list = [s[0] for s in spectrogram_list]
        for s in spectrogram_list:
            spect = np.array(birdinfile.get('/'+f+'/'+s))
            # encode the spect
            z = encode(spect, netE, transform_sample, batch_size, imageH, imageW, return_tensor, cuda)
            # create a dataset in the latent file with same name
            fout.create_dataset('Zvec_'+s, data=z, \
                                              compression = compress_type, compression_opts = compress_idx) 
    birdinfile.close()
    birdoutfile.close()
    end = time()
    print('..... bird %s finished in %5d secs.....'%(birdname, end-start)) 
    
    
    