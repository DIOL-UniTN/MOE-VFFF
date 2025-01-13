import torch
from torchaudio.transforms import MFCC as TorchMFCC
from torchaudio.transforms import FrequencyMasking, TimeMasking
from typing import List

class MFCC(torch.nn.Module):
    def __init__(self, sample_rate:int, n_mfcc:int, n_mels:int, window_hop:List[int], center:bool, 
                 log_mels:bool=False, norm:bool=1, ws_hl=None):#t_size:int, norm:bool=1)#, f_mask:float=0.0, t_mask:float=0.0):
        super(MFCC, self).__init__()
        norm = "slaney" if norm else None 
        n_fft, hop_length = window_hop
        self.mfcc = TorchMFCC(n_mfcc=n_mfcc, sample_rate=sample_rate,
                              melkwargs={"n_mels": n_mels, "n_fft": n_fft, "norm": norm,
                                         "hop_length": hop_length, "center": center, "norm": norm},
                              log_mels=log_mels
                              )
        # self.fmasker = FrequencyMasking(int(f_mask*n_mfcc), iid_mask=True)
        # self.tmasker = TimeMasking(int(t_mask*n_mfcc), iid_mask=True)

    def forward(self, x):
        return self.mfcc(x)
