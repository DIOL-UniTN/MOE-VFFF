import torch
from torchaudio.transforms import MFCC as TorchMFCC
from torchaudio.transforms import FrequencyMasking, TimeMasking
from typing import List

class MFCCLog2(TorchMFCC):
    def __init__(self, sample_rate:int, n_mfcc:int, n_mels:int, window_hop:List[int], center:bool, 
                 log_mels:bool=False, norm:bool=1, ws_hl=None):#t_size:int, norm:bool=1)#, f_mask:float=0.0, t_mask:float=0.0):
        norm = "slaney" if norm else None 
        n_fft, hop_length = window_hop
        super().__init__(n_mfcc=n_mfcc, 
                         melkwargs={"n_mels": n_mels, "n_fft": n_fft, "norm": norm,
                                    "hop_length": hop_length, "center": center, "norm": norm},
                         log_mels=log_mels
                         )

    def forward(self, waveform):
        mel_specgram = self.MelSpectrogram(waveform)
        if self.log_mels:
            log_offset = 1e-6
            mel_specgram = torch.log2(mel_specgram + log_offset)
        else:
            mel_specgram = self.amplitude_to_DB(mel_specgram)

        # (..., time, n_mels) dot (n_mels, n_mfcc) -> (..., n_nfcc, time)
        mfcc = torch.matmul(mel_specgram.transpose(-1, -2), self.dct_mat).transpose(-1, -2)
        return mfcc
