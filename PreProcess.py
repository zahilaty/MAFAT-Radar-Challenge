from All_Imports import *

# Functions for preprocessing and preprocess function
def fft(iq, axis=0):
    """
    Computes the log of discrete Fourier Transform (DFT).

    Arguments:
      iq_burst -- {ndarray} -- 'iq_sweep_burst' array
      axis -- {int} -- axis to perform fft in (Default = 0)

    Returns:
      log of DFT on iq_burst array
    """
    iq = np.log(np.abs(np.fft.fft(hann(iq), axis=axis)))
    #iq = np.log(np.abs(np.fft.fftshift(np.fft.fft(hann(iq), axis=axis), axes=axis)))
    return iq


def hann(iq, window=None):
    """
    Preformes Hann smoothing of 'iq_sweep_burst'.

    Arguments:
      iq {ndarray} -- 'iq_sweep_burst' array
      window -{range} -- range of hann window indices (Default=None)
               if None the whole column is taken

    Returns:
      Regulazied iq in shape - (window[1] - window[0] - 2, iq.shape[1])
    """
    if window is None:
        window = [0, len(iq)]

    N = window[1] - window[0] - 1
    n = np.arange(window[0], window[1])
    n = n.reshape(len(n), 1)
    hannCol = 0.5 * (1 - np.cos(2 * np.pi * (n / N)))
    return (hannCol * iq[window[0]:window[1]])[1:-1]


def max_value_on_doppler(iq, doppler_burst):
    """
    Set max value on I/Q matrix using doppler burst vector.

    Arguments:
      iq_burst -- {ndarray} -- 'iq_sweep_burst' array
      doppler_burst -- {ndarray} -- 'doppler_burst' array (center of mass)

    Returns:
      I/Q matrix with the max value instead of the original values
      The doppler burst marks the matrix values to change by max value
    """
    iq_max_value = np.max(iq)
    for i in range(iq.shape[1]):
        if doppler_burst[i] >= len(iq):
            continue
        iq[doppler_burst[i], i] = iq_max_value
    return iq


def normalize(iq):
    """
    Calculates normalized values for iq_sweep_burst matrix:
    (vlaue-mean)/std.
    """
    m = iq.mean()
    s = iq.std()
    return (iq - m) / s


def data_preprocess(data,axis=0):
    """
    Preforms data preprocessing.
    Change target_type lables from string to integer:
    'human'  --> 1
    'animal' --> 0

    Arguments:
      data -- {ndarray} -- the data set

    Returns:
      processed data (max values by doppler burst, DFT, normalization)
    """
    X = []
    for i in range(len(data['iq_sweep_burst'])):
        iq = fft(data['iq_sweep_burst'][i])
        #iq = np.fft.fft(hann(data['iq_sweep_burst'][i]), axis=axis)
        iq = max_value_on_doppler(iq,data['doppler_burst'][i])
        iq = normalize(iq)
        X.append(iq)
    #X = np.array(X)
    #data['iq_sweep_burst'] = X.reshape(X.shape[0],1,X.shape[1],X.shape[2])
    data['iq_sweep_burst'] = np.array(X)
    if 'target_type' in data:
        data['target_type'][data['target_type'] == 'animal'] = 0
        data['target_type'][data['target_type'] == 'human'] = 1
        data['target_type'][data['target_type'] == 'empty'] = 2
    return data

def CalcMedianVal(train_x,background):
    #train_x shape is ~6600,126,32
    #background shape is ~50000,126,32
    train_x_mid_val = np.median(np.abs(train_x),axis=(1,2))
    back_mid_val = np.median(np.abs(background),axis=(1,2))
    return train_x_mid_val , back_mid_val

def in_loop_process(x,background,x_mid_val,back_mid_val,Augmantaion = 'Train'):
    if Augmantaion == 'Train':
        #x shape is batch,126,32
        #background shape is 50000,126,32
        NumOfbackForPic = 2
        NumOfIdxToSample = x.shape[0]*NumOfbackForPic #3 background noises foreach picture
        rand_idx = random.choices(range(background.shape[0]),k=NumOfIdxToSample)
        back_norm_pic = background[rand_idx,:,:]/np.reshape(back_mid_val[rand_idx],(-1,1,1))
        back_norm_pic = back_norm_pic.reshape(x.shape[0],NumOfbackForPic,back_norm_pic.shape[1],back_norm_pic.shape[2])
        #validate with np.median(np.abs(back_norm_pic),axis=(2,3))
        #now back_norm_pic shape is x.shape[0],2,126,32
        x = x + np.sum(NumOfbackForPic*back_norm_pic*x_mid_val.reshape(-1,1,1,1),axis=1)
        #x = x + np.sum(10*back_norm_pic*x_mid_val.reshape(-1,1,1,1),axis=1)

        FlipVertical_idx = np.random.choice(a=[False, True], size=(x.shape[0],), p=[0.5, 1 - 0.5])
        x[FlipVertical_idx,:,:] = np.flip(x[FlipVertical_idx,:,:],axis=1)
        FlipHorizontal_idx = np.random.choice(a=[False, True], size=(x.shape[0],), p=[0.5, 1 - 0.5])
        x[FlipHorizontal_idx,:,:] = np.flip(x[FlipHorizontal_idx,:,:],axis=2)

    # if Augmantaion == 'Exp':
    #     NumOfbackForPic = 1
    #     NumOfIdxToSample = x.shape[0] * NumOfbackForPic  # 3 background noises foreach picture
    #     rand_idx = random.choices(range(background.shape[0]), k=NumOfIdxToSample)
    #     back_norm_pic = background[rand_idx, :, :] / np.reshape(back_mid_val[rand_idx], (-1, 1, 1))
    #     back_norm_pic = back_norm_pic.reshape(x.shape[0], NumOfbackForPic, back_norm_pic.shape[1], back_norm_pic.shape[2])
    #     # validate with np.median(np.abs(back_norm_pic),axis=(2,3))
    #     # now back_norm_pic shape is x.shape[0],2,126,32
    #     x = x + np.sum(10 * back_norm_pic * x_mid_val.reshape(-1, 1, 1, 1), axis=1)
    #     # Do nothing, we will just randomize the selection..

    x = np.log10(np.abs(x))
    x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
    return x

def CV_process(x,y):
    x = np.log10(np.abs(x))
    x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

    x = torch.from_numpy(x.astype('float32'))
    y = torch.from_numpy(y.astype('long')).type(torch.LongTensor)
    return x,y

def Base_process(x,y):
    x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
    x = torch.from_numpy(x.astype('float32'))
    y = torch.from_numpy(y.astype('float')).type(torch.LongTensor).reshape(-1,1)
    return x,y

def Base_process_improved(x1,x2,y):
    x1 = x1.reshape(x1.shape[0], 1, x1.shape[1], x1.shape[2])
    x1 = torch.from_numpy(x1.astype('float32'))
    x2 = x2.reshape(x2.shape[0], 1, x2.shape[1], x2.shape[2])
    x2 = torch.from_numpy(x2.astype('float32'))
    y = torch.from_numpy(y.astype('float')).type(torch.LongTensor).reshape(-1,1)
    return x1,x2,y


def MagAngleProcess(data):

    X = []
    for i in range(len(data['iq_sweep_burst'])):
        iq = fft(data['iq_sweep_burst'][i])
        #iq = np.fft.fft(hann(data['iq_sweep_burst'][i]), axis=axis)
        iq = max_value_on_doppler(iq,data['doppler_burst'][i])
        iq = normalize(iq)
        X.append(iq)
    data['MagFFT'] = np.array(X)

    X = []
    for i in range(len(data['iq_sweep_burst'])):
        Mat = data['iq_sweep_burst'][i]
        MatAngleProcessed = np.angle(Mat * np.conj(Mat[0, :]))
        MatAngleProcessed = np.unwrap(MatAngleProcessed, axis=0)
        scipy.signal.detrend(MatAngleProcessed, axis=0, type='linear', bp=0, overwrite_data=True)
        MatAngleProcessed = normalize(MatAngleProcessed)
        X.append(MatAngleProcessed)
    data['ResidualPhase'] = np.array(X)

    if 'target_type' in data:
        data['target_type'][data['target_type'] == 'animal'] = 0
        data['target_type'][data['target_type'] == 'human'] = 1
        data['target_type'][data['target_type'] == 'empty'] = 2

    return data