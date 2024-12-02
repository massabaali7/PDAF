import torchaudio
from chairsu_align import align
import os 
os.chdir('../')
from util_stats.local_stats import local_extract_phn_frame_probs
import torch
import torchaudio.functional as F
from preprocessing.ast_processor import ast
from torch.nn import CosineSimilarity
from encoder.self_attn import TransformerSelfAttention
import yaml 
import numpy as np 

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config("config.yaml")
config_inf = config['inference']
config_data = config['data']

def preprocess_attn_model(wav_path,target_sample_rate, emb_dim):
  feature_extractor = ast()
  audio_arrays = torchaudio.load(wav_path)
  audio_arrays_resampled = F.resample(audio_arrays[0], audio_arrays[1], target_sample_rate)
  phonemeAlignment = align(wav_path)
  input_values = feature_extractor(audio_arrays_resampled[0], sampling_rate=target_sample_rate,return_tensors="pt").input_values #[0] 
  mask_specific_phoneme = False
  phone_vector , frame_vector, mask_vector = local_extract_phn_frame_probs(phonemeAlignment,mask_specific_phoneme,emb_dim)
  return input_values, phone_vector, frame_vector, mask_vector

def attn_model(audio):  
  model = TransformerSelfAttention(config_inf['d_model'], config_inf['heads'], config_inf['d_model'],config_inf['numSpks'])
  snapshot = torch.load(config_data['checkpoint_path'],map_location='cpu')
  model.load_state_dict(snapshot["MODEL_STATE"], strict= False) 
  model.eval()
  input_values, phone_vector, frame_vector, mask_vector = preprocess_attn_model(audio,config_data['sample_rate'],config_inf['emb_dim'])
  
  phone_vector = np.array(phone_vector)
  mask_vector = np.array(mask_vector)
  phone_vector = torch.tensor(phone_vector).unsqueeze(0)
  mask_vector = torch.tensor(mask_vector).unsqueeze(0)
  input_values = np.array(input_values)
  input_values = torch.from_numpy(input_values)
  y_pred,emb = model(input_values, prob_phn=phone_vector, mask=mask_vector, lambda_val = config_inf['lambda_value'])
  return emb[0]
def verify(audio1, audio2):
    cosine_sim = CosineSimilarity(dim=-1)
    score = cosine_sim(audio1, audio2)
    if score > config_inf['threshold']:
       print("matched")
    else:
       print("unmatched") 
    return score


wav_file1 = "./samples/00003_female.wav"

y_pred_1 = attn_model(wav_file1)
wav_file2 = "./samples/00005_female.wav"

y_pred_2 = attn_model(wav_file2)

score = verify(y_pred_1, y_pred_2)
print(score)