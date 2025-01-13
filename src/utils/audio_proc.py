import os
import torch
import numpy as np
import torch.nn.functional as F

# preprocessing for raw KWS
def fix_audio_length(audio, t=1.0, sr=16000):
    '''
    audio (Tensor): Tensor of audio of dimension (1, Time)
    '''
    # Padding if needed
    req_len = int(t*sr)
    if audio.size(1) < req_len:
        p = req_len - audio.size(1)
        audio = F.pad(audio, (0, p), 'constant', 0)
    elif audio.size(1) > req_len: # Random crop
        start = np.random.randint(0, audio.size(1)-req_len)
        audio = audio[:, start:start+req_len]
    return audio

def trim_or_pad_audio(audio, t=1.0, fs=16000):
    '''
    audio (Tensor): Tensor of audio of dimension (Time)
    '''
    max_len = int(t*fs)
    shape = audio.shape
    if shape[0] >= max_len:
        audio = audio[:max_len]
    else:
        n_pad = max_len - shape[0]
        zero_shape = (n_pad,)
        audio = torch.cat((audio, torch.zeros(zero_shape)), axis=0)
    return audio
