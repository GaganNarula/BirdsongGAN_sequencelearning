import torch
from torch.autograd import Variable
import sys, os
import numpy as np
import soundfile as sf
import wave
from scipy.signal import resample
import librosa.core as lc
from librosa.util import fix_length
from datetime import datetime as dt
from PIL import Image
import librosa
import sounddevice as sd
import pickle
from os.path import join
from collections import OrderedDict

if not sys.platform=='linux':
    import librosa.display
    import matplotlib.pyplot as plt


def random_crop(im,width=8):
    ceil = im.shape[1]-width
    ind = np.random.randint(ceil)
	return im[:,ind:ind+width]


def segment_image(im,width=8):
    segments = [im[:,i*width:(i+1)*width] for i in range(im.shape[1]//width)]
    return segments


def segment_image_withoverlap(im,width=8,nonoverlap=4):
    segments = []
    nbins = im.shape[1]//nonoverlap
    idx = 0
    for n in range(nbins):
        segments.append(im[:,idx:idx+width])
        idx += nonoverlap
    return segments


def to_batches(segments,batch_size):
    n_batches = int(np.ceil(len(segments)/batch_size))
    batches = [np.zeros(shape=(batch_size,)+tuple(segments[0].shape)) for i in range(n_batches)]
    for i in range(len(segments)):
        batch_idx = i//batch_size
        idx = i%batch_size
        batches[batch_idx][idx] = segments[i]
    return np.array(batches), len(segments)


def get_random_sample(directory):
    try:
        files = os.listdir(join(directory,'images'))
        randint = np.random.randint(len(files))
        return join(directory,'images',files[randint])
    except:
        dirs = [i for i in os.listdir(directory) if not len(os.listdir(join(directory,i)))==0]
        rand_dir = dirs[np.random.randint(len(dirs))]
        files = os.listdir(join(directory,rand_dir))
        return join(directory,rand_dir,files[np.random.randint(len(files))])


def downsample(x,down_factor):
    n = x.shape[0]
    y = np.floor(np.log2(n))
    nextpow2 = int(np.power(2, y + 1))
    x = np.concatenate((np.zeros((nextpow2-n), dtype=x.dtype), x))
    x = resample(x,len(x)//down_factor)
    return x[(nextpow2-n)//down_factor:]


def play_clip(data, fs=44100):
    sd.play(data, fs)


def play_file(file_path):
    data,fs = sf.read(file_path)
    print("playing from file: ",file_path)
    sd.play(data, fs)


def load_from_folder(base_path,folder_path):
    files = os.listdir(join(base_path,folder_path,'songs'))
    files = [i for i in files if '.wav' in i.lower()]
    data = [sf.read(join(base_path,folder_path,'songs',i)) for i in files]
    rates = [i[1] for i in data]
    data = [i[0] for i in data]
    if len(data)==0:
        return None,None
    if len(set(rates))>1:
        print("Sample rates are not the same")
        print("Sample rates are: ")
        print(rates)
    else:
        print("Sample rate is : ",rates[0])
    return data,rates[0]


def play(f):
    import pyaudio
    CHUNK = 1024
    print('playing file: ' + f.split('\\')[-1])
    wf = wave.open(f, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(CHUNK)
    try:
        while data != '':
            stream.write(data)
            data = wf.readframes(CHUNK)
    except KeyboardInterrupt:
        pass
    stream.stop_stream()
    stream.close()
    p.terminate()


def normalize_image(image):
    return image / np.std(image)


def update_progress(progress):
    print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(progress * 50),progress * 100),end='')


def phase_restore(mag, random_phases,n_fft, N=50):
    p = np.exp(1j * (random_phases))

    for i in range(N):
        _, p = librosa.magphase(librosa.stft(
            librosa.istft(mag * p), n_fft=n_fft))
    #    update_progress(float(i) / N)
    return p


def to_image(seq,nfft):
    nfft_padlen = int(len(seq) + nfft / 2)
    stft = lc.stft(fix_length(seq, nfft_padlen), n_fft=nfft)
    return np.array([np.abs(stft), np.angle(stft)]).transpose(1, 2, 0)


def from_polar(image):
    return image[:, :, 0]*np.cos(image[:, :, 1]) + 1j*image[:,:,0]*np.sin(image[:,:,1])


def from_image(image,clip_len=None):
    if clip_len:
        return fix_length(lc.istft(from_polar(image)), clip_len)
    else:
        return lc.istft(from_polar(image))


def save_image(image,save_path,save_idx,amplitude_only=False):
    if amplitude_only:
        np.save(join(save_path, str(save_idx) + '.npy'), image[:,:,0])
    else:
        np.save(join(save_path, str(save_idx) + '.npy'), image)


def get_spectrogram(data,log_scale=False,show=False,polar_form_input=False):
    if polar_form_input:
        image = from_polar(data)
    elif len(data.shape)==2:
        image=data
    else:
        image = lc.stft(data)
    D = librosa.amplitude_to_db(image, ref=np.max)
    if show:
        plt.figure()
        if log_scale:
            librosa.display.specshow(D, y_axis='log')
        else:
            librosa.display.specshow(D, y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Frequency power spectrogram')
        plt.show()
    return D


def save_spectrogram(filename,D):
    if D.min() < 0:
        D = D-D.min()
    D = D/D.max()
    I = Image.fromarray(np.uint8(D*255))
    I.save(filename)


def save_audio_sample(sample,path,samplerate):
    sf.write(path,sample,samplerate=int(samplerate))


def normalize_spectrogram(image,threshold):
    if threshold>1.0:
        image = image/threshold
    image = np.minimum(image,np.ones(shape=image.shape))
    image = np.maximum(image,-np.ones(shape=image.shape))
    return image


def generate_dataset(base_path,out_path,nfft,max_num=None,amplitude_only=False,downsample_factor=None):
    save_idx=0
    folders = os.listdir(base_path)
    out_path = join(out_path,'images')
    os.makedirs(out_path,exist_ok=True)
    for folder in folders:
        songs, fs = load_from_folder(base_path, folder)
        if songs:
            if downsample_factor:
                songs = [downsample(i, downsample_factor) for i in songs]
            ims = [to_image(i,nfft) for i in songs]
            for im in ims:
                save_image(im,out_path,save_idx,amplitude_only=amplitude_only)
                save_idx+=1
                if max_num:
                    if save_idx>=max_num:
                        return


def transform(im):
    """
    This function should be used to transform data into the desired format for the network.
    inverse transoform should be an inverse of this function
    """
    im = from_polar(im)
    im, phase = lc.magphase(im)
    im = np.log1p(im)
    return im


def inverse_transform(im):
    """
    Inverse (or at least almost) of transofrm()
    """
    random_phase = im.copy()
    np.random.shuffle(random_phase)
    p = phase_restore((np.exp(im) - 1), random_phase, 256, N=50)
    return (np.exp(im) - 1) * p


def im_loader(path):
    """
    This is the function between data on disk and the network.
    """
    im = np.load(path)
    im = random_crop(im, width=opt.imageW)
    im = transform(im)
    return im


def encode_sample(path,sample,epoch=None):
    with open(join(path, 'opt.pkl'), 'rb') as f:
        opt = pickle.load(f)
    ngpu = opt.ngpu
    nz = opt.nz
    ngf = opt.ngf
    nc = opt.nc
    input = torch.FloatTensor(opt.batchSize, nc, opt.imageH, opt.imageW)
    input = Variable(input)

    if epoch is not None:
        net_E_file = join(path,'netE_epoch_%d.pth'%(epoch))
    else:
        E_files = [i for i in os.listdir(path) if 'netE' in i]
        net_E_file = join(path,'netE_epoch_%d.pth'%(len(E_files)-1))
    try:
        from networks_1d import _netG, _netE, _netD, weights_init, GANLoss
        netE = _netE(ngpu, nz, ngf, nc)
        netE.apply(weights_init)
        netE.load_state_dict(torch.load(net_E_file))
        opt.imageH=opt.nc
    except:
        from networks_audio_nophase import _netG, _netE, _netD, weights_init, GANLoss
        netE = _netE(ngpu, nz, ngf, nc)
        netE.apply(weights_init)
        netE.load_state_dict(torch.load(net_E_file))

    netE.cuda()
    encoded = []
    sample_segments = segment_image(sample, width=opt.imageW)
    sample_segments = [transform(k) for k in sample_segments]
    sample_batches, num_segments = to_batches(sample_segments, opt.batchSize)
    cnt = 0
    sequence=[]
    for j in range(len(sample_batches)):
        input.data.copy_(torch.from_numpy(sample_batches[j]))
        encoding = netE(input)
        for k in range(opt.batchSize):
            if cnt >= num_segments:
                sequence.append(np.zeros(shape=sequence[-1].shape))
            else:
                sequence.append(encoding.data[k].cpu().numpy())
                cnt += 1
    return np.array([i for i in sequence if not np.sum(np.abs(i))==0])


def save_spectrogram_matplotlib(filepath, spectrogram, formatt):
    fig, axarr = plt.subplots(2)
    # flip the axis because we always show higher frequencies on top (while imshow flips it)
    axarr[0].imshow(np.flip(spectrogram, axis=0))
    plt.savefig(filepath, dpi=300, format=formatt)
    plt.clf()


def reconstruct_spect_and_audio_from_z(zseq, netG, opt, savepath):
    #Given a latent z-vector sequence and a GAN generator (netG), decode spectrogram as well as audio file
    zseq = zseq.reshape((zseq.shape[0], opt.nz, 1, 1))
    # sometimes z-vectors are double type, need to be float for pytorch
    zseq = Variable(torch.from_numpy(zseq).float())
    if opt.cuda:
        zseq = zseq.cuda()

    # decoding step
    reconstruction = netG(zseq)
    nsteps = reconstruction.size()[0]
    reconstructed_samples = []
    spectrogram = []
    for k in range(nsteps):
        tmp = reconstruction.data[k, :, :, :].cpu().numpy().reshape((opt.imageH, opt.imageW))
        reconstructed_samples.append(inverse_transform(tmp))
        spectrogram.append(tmp)

    reconstructed_samples = np.concatenate(reconstructed_samples, axis=1)
    reconstructed_audio = lc.istft(reconstructed_samples)
    del reconstructed_samples, reconstruction
    # save generated audio
    save_audio_sample(reconstructed_audio, savepath + 'rec_audio.wav', 16000)
    # return spectrogram from z
    spectrogram = np.concatenate(spectrograms, axis=1)  # here columns are time, rows = freq
    return spectrogram


def fix_state_dict(loaded_dict):
    new_state_dict = OrderedDict()
    for k,v in loaded_dict.items():
        name = k[7:] # remove 'module'
        new_state_dict[name] = v
    return new_state_dict


def renormalize_spectrogram(s):
    s = s - np.min(s)
    s = s / np.max(s)
    return 10*np.log(s + 0.01)


def load_and_concatenate_z(path):
    #Load and concatenate all z-vectors of a single day of songs into one np.array
    z_vecs=[]
    for f in os.listdir(path):
        pth2fil = os.path.join(path,f)
        if pth2fil.endswith('.npy'):
            s = np.load(pth2fil)
            for k in range(s.shape[0]):
                # this is done to remove zeros (masking added by Svenni)
                if np.sum(np.abs(s[k,:])) != 0:
                    z_vecs.append(s)
    z_vecs = np.concatenate(z_vecs,axis=0) #rows are time steps, cols are features
    return np.array(z_vecs)



def load_z_no_concatenate(path):
    #Returns a list containing individual elements as individual z-vec
    #np.arrays from one day of singing
    z_vecs=[]
    for f in os.listdir(path):
        pth2fil = os.path.join(path,f)
        if pth2fil.endswith('.npy'):
            s = np.load(pth2fil)
            for k in range(s.shape[0]):
                # this is done to remove zeros (masking added by Svenni)
                if np.sum(np.abs(s[k,:])) != 0:
                    z_vecs.append(s)
    return z_vecs