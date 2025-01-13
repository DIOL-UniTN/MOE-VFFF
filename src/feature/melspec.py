import torch
from torchaudio.transforms import MelSpectrogram as TorchMelSpectrogram
from typing import List

class MelSpectrogram(torch.nn.Module):
    def __init__(self, sample_rate:int, n_mels:int, window_hop:List[int], center:bool, 
                 log_mels:bool=False, norm:str|None="slaney"):
        super(MelSpectrogram, self).__init__()
        norm = "slaney" if norm else None 
        n_fft, hop_length = window_hop
        self.melspec = TorchMelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft,
                                           hop_length=hop_length, center=center,
                                           norm=norm)
        self.log = log_mels

    def forward(self, x):
        x = self.melspec(x)
        x = torch.log(x) if self.log else x
        return x
