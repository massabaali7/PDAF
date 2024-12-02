# -*- coding: utf-8 -*-
# !pip install torch torchvision torchaudio
# !pip install datasets transformers
# !pip install g2p_en praatio librosa
import os 
import sys 

os.chdir('./charsiu')
sys.path.insert(0,'src')

from Charsiu import charsiu_forced_aligner,charsiu_predictive_aligner
import sys
import torch
from itertools import groupby
from datasets import load_dataset
import matplotlib.pyplot as plt
import librosa
import pandas as pd 
charsiu_pa = charsiu_predictive_aligner(aligner='charsiu/en_w2v2_fc_10ms')

def align(waveform):
    alignment = charsiu_pa.align(
    audio = waveform) 
    phonemeAlignment = pd.DataFrame(
    data    = alignment,
    columns = ['start', 'end', 'phoneme'])
    return phonemeAlignment 

