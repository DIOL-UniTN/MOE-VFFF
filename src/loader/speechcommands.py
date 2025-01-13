import shutil
import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS 
from torch.utils.data import Subset, DataLoader
from utils.audio_proc import fix_audio_length
from torchaudio.datasets.utils import _load_waveform
from tqdm import tqdm
from pathlib import Path
from glob import glob
from tqdm import tqdm
import numpy as np
import logging

# TODO: uknown -> shuffle(walker) -> picke the ones not in label -> take 4000
# TODO: noise -> picke random noise -> scale randomly -> create dir and files 4000
# LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown']
LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'noise']
#TODO: noise?

class SCLoader():
    def __init__(self, batch_size:int, feature, sample_rate:int, debug:bool):
        self.batch_size = batch_size
        self.labels = LABELS #TODO
        self.duration = 1.0 # TODO
        self.sr = sample_rate # To be resampled
        self.out_dim = 10

        feature = feature(sample_rate=self.sr)
        trainset = SubsetSC('./data', 'training', feature, sample_rate, debug)
        testset = SubsetSC('./data', 'testing', feature, sample_rate, debug)
        validset = SubsetSC('./data', 'validation', feature, sample_rate, debug)

        num_workers = 0 if debug else 8
        self.train = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers)
        self.valid = DataLoader(validset, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers)
        self.test = DataLoader(testset, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers)
        # TODO: is valid shuffled then loaded? cHECK!

        self.batch_size = batch_size
        self.labels = LABELS
        self.out_dim = 10
        # self.out_dim = 11
        if feature:
            out_shape = feature(torch.randn(1, int(self.sr*self.duration))).shape
            self.in_chan, self.in_size = out_shape[0], tuple(out_shape[1:])
            logging.info(f"Feature size: {self.in_size}")
        else:
            self.in_chan, self.in_size = 1, (1, int(self.sr*self.duration))
            logging.info("Raw data")

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, dataset_dir: str='./data', 
                 subset:str='training', feature=None, 
                 sample_rate=16000,
                 debug: bool=False):
        super().__init__(dataset_dir, download=True, subset=subset)
        self.labels = LABELS
        self.debug = debug
        self.sr = sample_rate
        self.duration = 1.0
        self.resampler = torchaudio.transforms.Resample(16000, self.sr)
        self._walker = [file for file in self._walker if file.split('/')[-2] in LABELS] 
        self.subset=subset
        # np.random.shuffle(self._walker)
        self.feature = feature

    def __len__(self):
        if self.debug:
            return 2
        return int(len(self._walker))

    def label_to_target(self, word):
        return torch.tensor(LABELS.index(word))

    def __getitem__(self, idx):
        metadata = self.get_metadata(idx)
        waveform = _load_waveform(self._archive, metadata[0], metadata[1])
        waveform = self.resampler(waveform)
        waveform = fix_audio_length(waveform, t=self.duration, sr=self.sr) # TODO Handle randomness for test? How to handle?
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-10)
        feat = self.feature(waveform)
        target = self.label_to_target(metadata[2])
        return feat, target

# class SubsetSCOld(SPEECHCOMMANDS):
#     def __init__(self, dataset_dir: str='./data', 
#                  setname:str='train', feature=None, 
#                  sample_rate=16000,
#                  debug: bool=False):
#         super().__init__(dataset_dir, download=True)
#         self.labels = LABELS
#         self.debug = debug
#         self.sr = sample_rate
#         self.duration = 1.0
#         self.resampler = torchaudio.transforms.Resample(16000, self.sr)
#         ppath = Path(self._path)
#         pwalker = [Path(wavf) for wavf in self._walker]
#
#         def handle_unknown_class():
#             unknown_dir = ppath/'unknown'
#             if unknown_dir.is_dir():
#                 return
#             unknown_dir.mkdir()
#             logging.info("Generating the UNKNOWN class...")
#             unknown_wavs = [wavf for wavf in self._walker if wavf.parent.name not in LABELS]
#             np.random.shuffle(unknown_wavs)
#             for i in tqdm(range(4000)):
#                 wav = unknown_wavs[i]
#                 shutil.copyfile(wav, unknown_dir/(wav.parent.name+'_'+wav.name))
#
#         def handle_subset():
#             if ((ppath/'testing_list_subset.txt').is_file() and
#                 (ppath/'training_list_subset.txt').is_file()):
#                 return
#             logging.info("Generating subsets...")
#             test_list = load_list('testing_list.txt')
#             # unknown_list = [Path(wavf) for wavf in glob(str(ppath/"unknown/*wav"))]
#             # train_list = [wavf for wavf in tqdm(pwalker, desc='Generating training list') if wavf not in test_list] #and wavf not in unknown_list)]
#             train_list = [wavf for wavf in tqdm(self._walker, desc='Generating training list') if wavf not in test_list] #and wavf not in unknown_list)]
#             test_list = [Path(wavf) for wavf in test_list]
#             train_list = [Path(wavf) for wavf in train_list]
#             with (open(ppath/'testing_list_subset.txt', 'w') as testf, 
#                   open(ppath/'training_list_subset.txt', 'w') as trainf):
#                 [trainf.write(str(wavf.parent.name/Path(wavf.name+'\n'))) for wavf in train_list if wavf.parent.name in LABELS]
#                 [testf.write(str(wavf.parent.name/Path(wavf.name+'\n'))) for wavf in test_list if wavf.parent.name in LABELS]
#                 # [trainf.write(str(wavf.parent.name/Path(wavf.name+'\n'))) for wavf in unknown_list[:3900]]
#                 # [testf.write(str(wavf.parent.name/Path(wavf.name+'\n'))) for wavf in unknown_list[3600:]]
#
#         def load_list(filename):
#             filepath = ppath/filename
#             with open(filepath) as f:
#                 return [str(ppath/line.strip()) for line in f]
#
#         # handle_unknown_class()
#         handle_subset()
#         self.walker = load_list(setname+"_list_subset.txt")
#         np.random.shuffle(self.walker)
#         self.feature = feature
#         self.train = True if setname == 'training'
#
#     def __len__(self):
#         if self.debug:
#             return 2
#         return int(len(self.walker))
#
#     def label_to_target(self, word):
#         return torch.tensor(LABELS.index(word))
#
#     def __getitem__(self, idx):
#         audio, sr = torchaudio.load(self.walker[idx])
#         audio = self.resampler(audio)
#         audio = fix_audio_length(audio, t=self.duration, sr=self.sr) # TODO Handle randomness for test? How to handle?
#         audio = (audio - audio.mean()) / (audio.std() + 1e-10)
#         feat = self.feature(audio)
#         label = Path(self.walker[idx]).parent.name
#         target = self.label_to_target(label)
#         return feat, target
