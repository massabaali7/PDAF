import numpy as np 
import pandas as pd 
import math 

def local_extract_phn_frame_probs(data,mask_x,emb_dim): # per utterance
  data = pd.DataFrame(
    data    = data,
    columns = ['start', 'end', 'phoneme'])
  seq_len = emb_dim 
  mask_vector = np.zeros((emb_dim, emb_dim)) #input_features
  frame_vector = np.zeros((emb_dim, emb_dim)) #input_features
  phone_vector = np.zeros((emb_dim, emb_dim)) #input_features
  #print(data['phoneme'])
  a = sorted(data['phoneme'].unique()) 
  a = [value for value in a if value != "[SIL]"]
  dic_phn = {}
  dic_frame_timing = {} 
  dic_frame_timing['total'] = data[data['phoneme'] != '[SIL]']['end'].sum()
  dic_phn['total'] = data[data['phoneme'] != '[SIL]'].shape[0]
  ind = 0
  for p in data['phoneme']:
      dic_phn[p] = data['phoneme'].value_counts()[p]
      dic_frame_timing[p] = data[data['phoneme'] == p]['end'].sum()
      phn_prob = dic_phn[p] / dic_phn['total']
      frame_prob = (dic_frame_timing[p]*100) / (dic_frame_timing['total']*100)
      phn_prob = math.log(phn_prob)
      frame_prob = math.log(frame_prob)
      replace_list_phn = [phn_prob] * seq_len
      start = int(data['start'][ind] * 100)
      end = int(data['end'][ind] * 100)
      phone_vector[start:end] = replace_list_phn
      replace_list_frame = [frame_prob] * seq_len
      frame_vector[start:end] = replace_list_frame
      if p == mask_x or p == "[SIL]":
          replace_list_mask = [0] * seq_len
          mask_vector[start:end] = replace_list_mask
      else: 
          replace_list_mask = [1] * seq_len
          mask_vector[start:end] = replace_list_mask
      ind = ind + 1
  return phone_vector , frame_vector, mask_vector
