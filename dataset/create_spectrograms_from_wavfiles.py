from utils.utils import *
from scipy.io import wavfile
from time import time


def read_wav_file(path):
    fs, wf = wavfile.read(path)
    if wf.dtype == 'int16':
        nb_bits = 16  # -> 16-bit wav files
    elif wf.dtype == 'int32':
        nb_bits = 32  # -> 32-bit wav files
    max_nb_bit = float(2 ** (nb_bits - 1))
    samples = wf / (max_nb_bit + 1.0)
    return samples, fs


def load_wavfiles_from_folder(base_path ,folder_path):
    pth = join(base_path, folder_path, 'songs')
    if os.path.exists(pth):
        pp = pth
    else:
        pp = join(base_path ,folder_path)
    files = os.listdir(pp)
    files = [i for i in files if '.wav' in i.lower()]
    data = []
    fs = 0
    for i in files:
        samples, fs = read_wav_file(join(base_path ,folder_path ,'songs' ,i))
        data.append(samples)
    return data, fs


def generate_spectrogram_dataset_by_day(base_path, out_path, name='', nfft=256, amplitude_only=False, downsample_factor=2,
                                        normalize_amplitude=False, train_val_split=True):
    '''
    Go through all folders in a bird's dataset, load wav files, generate spectrograms and save them

    :param base_path: path to bird's individual wav file day folders
    :param out_path:  path to save spectrogram day folders
    :param name: birdname (required!) e.g. 'b3r16'
    :param nfft: number of nfft bins to use for short time Fourier transform
    :param amplitude_only:  whether to save only the magnitude of the spectrogram and not the phase
    :param downsample_factor:  how much to downsample learning rate
    :param normalize_amplitude:  whether to normalize amplitude (maximum value = 1)
    :param train_val_split: split data into train and test sets?

    :return:
    '''
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
    for i in range(len(folders)):
        try:
            day = dt.strptime('-'.join(folders[i].split('-')[:3]), '%Y-%m-%d')
            num = (day-days[0]).days
            out_path_full = join(out_path,"day%d_%s"%(num,name),"images")
        except:
            out_path_full = join(out_path,folders[i]+'_'+name,"images")

        os.makedirs(out_path_full, exist_ok=True)
        songs, fs = load_from_folder(base_path, folders[i])
        if not songs:
            continue
        if downsample_factor:
            songs = [downsample(i, downsample_factor) for i in songs]
        if normalize_amplitude:
            songs = [s/np.max(s) for s in songs]
        ims = [to_image(i,nfft) for i in songs]
        if train_val_split:
            out_path_test = out_path_full.replace(out_path,out_path+'_test/')
            out_path_val = out_path_full.replace(out_path,out_path+'_val/')
            os.makedirs(out_path_test, exist_ok=True)
            os.makedirs(out_path_val, exist_ok=True)
            im_train,im_test = train_test_split(ims,test_size=0.1)
            im_val,im_test = train_test_split(im_test,test_size=0.5)
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
            for im in ims:
                save_image(im,out_path_full,save_idx,amplitude_only=amplitude_only)
                save_idx+=1
        j+=1
        print('\n .... Folder %s done .... %d/%d left ...\n'%(folders[i],j,len(folders)))


def generate_dataset_per_bird(base_inpath, base_outpath, birds, names=''):
    '''
    Go through a folder contain folders named by "bird-name_wav_files". Each bird-name folder contains folders
    named "20XX-XX-XX" which is one day's recordings. Please check the basepath below for your personal directory setup.

    :param base_inpath: base path to wav file folders of each bird
    :param base_outpath: base path to save spectrogram folders of each bird
    :param birds: list of bird-name folders to process e.g. ['k3r16_wav_files', 'p20r16_wav_files'],
    :param names: list of bird names (alternate names/codes)
    :return:
    '''
    if names=='':
        names = birds
    dsetstart = time()
    for k in range(len(birds)):
        bird = birds[k]
        name = names[k]
        print('\n\n ############ bird %s ############ '%(name))
        path = os.path.join(base_inpath, bird, 'SAP')
        start = time()
        generate_dataset_by_day(path, base_outpath,
                                nfft=256, amplitude_only=False, downsample_factor=2,
                                normalize_amplitude=False, train_val_split=True, name=name)
        end = time()
        print('\n ........... This took %f seconds ............ \n'%(end - start))
    dsetend = time()
    print('\n\n ################# DONE CREATING DATASET, TOTAL TIME ELAPSED %f ######################'%(dsetend - dsetstart))