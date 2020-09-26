from moviepy.editor import *
import pdb
import librosa
import librosa.display
# from fastai.vision import *
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
def to_mp3():
    video = VideoFileClip('/home/share/Highlight/proDataset/DomainSpecific/video/skating/1n7Afe5AZCM.mp4')
    audio = video.audio
    # audio1 = audio.subclip(0,0.5)
    # pdb.set_trace()
    # audio.write_audiofile('tl
    
    # est.mp3')
    frames = int(video.duration*video.fps)
    step = int(frames/16)
    print(video.duration)
    for i in range(step):
        start = i*16/video.fps
        end = (i+1)*16/video.fps
        print(start,end)
        subvideo = video.subclip(start,end)
        subaudio = audio.subclip(start,end)
        # subvideo.write_videofile('samples/'+str(i)+'.mp4')
        subaudio.write_audiofile('samples/'+str(i)+'.mp3')
to_mp3()
def to_signal(path):
    fs, signal = wav.read(path)
    pdb.set_trace()
    print(a)

def to_spec(path):
 
    filename =path
    # 2. Load the audio as a waveform `y`
    #    Store the sampling rate as `sr`
    y, sr = librosa.load(filename,sr=None)
    
    # plt.figure(figsize=(12, 8))
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    # plt.subplot(4, 2, 1)
    librosa.display.specshow(D, y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    plt.show()


def to_mfcc(pathname):
    y, sr = librosa.load(pathname)  # 将音频文件加载为浮点时​​间系列。
    mfcc = librosa.feature.mfcc(y,sr)
    print(mfcc)
    print(mfcc.shape)

# y, sr = librosa.load(librosa.util.example_audio_file())
# librosa.feature.mfcc(y=y, sr=sr)
    return mfcc

to_spec('samples/16.wav')
to_mfcc('samples/16.mp3')



class conf:
    sampling_rate = 44100
    duration = 1  # sec
    hop_length = 347 * duration  # to make time steps 128
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    padmode = 'constant'
    samples = sampling_rate * duration
def read_audio(conf, pathname, trim_long_data):
    """
    librosa 是音频处理库，conf.sampling_rate 为采样率 44100
    :param conf:
    :param pathname:
    :param trim_long_data:
    :return:
    """
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)  # 将音频文件加载为浮点时​​间系列。
    # trim silence
    if 0 < len(y):  # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y)  # trim, top_db=default(60)
    # make it unified length to conf.samples
    if len(y) > conf.samples:  # long enough 88200
        if trim_long_data:
            y = y[0:0 + conf.samples]
    else:  # pad blank
        padding = conf.samples - len(y)  # add padding at both ends 不够的话就补充。
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), conf.padmode)
    return y,sr
 
def audio_to_melspectrogram(conf, audio):
    """
    计算一个梅尔频谱系数图
    :param conf:
    :param audio:
    :return:
    """
    spectrogram = librosa.feature.melspectrogram(audio,
                                                 sr=conf.sampling_rate,
                                                 n_mels=conf.n_mels,
                                                 hop_length=conf.hop_length,
                                                 n_fft=conf.n_fft,
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)  # 转化频谱系数单位
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram
 
def show_melspectrogram(conf, mels, title='Log-frequency power spectrogram'):
    """
    
    :param conf: 
    :param mels: 
    :param title: 
    :return: 
    """
    librosa.display.specshow(mels, x_axis='time', y_axis='mel',
                             sr=conf.sampling_rate, hop_length=conf.hop_length,
                             fmin=conf.fmin, fmax=conf.fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    
    # plt.show()
def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):
    """
    :param conf:
    :param pathname:
    :param trim_long_data:
    :param debug_display:
    :return:
    """
    x = read_audio(conf, pathname, trim_long_data)
    mels = audio_to_melspectrogram(conf, x)
    if debug_display:
        IPython.display.display(IPython.display.Audio(x, rate=conf.sampling_rate))
        show_melspectrogram(conf, mels)
    return mels
def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    """
    
    :param X: 
    :param mean: 
    :param std: 
    :param norm_max: 
    :param norm_min: 
    :param eps: 
    :return: 
    """
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)
 
    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V
def get_default_conf():
    return conf
# conf = get_default_conf()
# for i in range(180):
#     to_mfcc('samples/'+str(i)+'.mp3')
#     print('samples/'+str(i)+'.mp3')
    # x = read_as_melspectrogram(conf, 'samples/'+str(i)+'.mp3', trim_long_data=False)
    # show_melspectrogram(conf,x)
    # plt.savefig('samples/'+str(i)+'.jpg')
    # plt.figure()
    # print('samples/'+str(i)+'.jpg')
# to_mp3()