import numpy as np 
import pandas as pd
import math 
def sum_except_key(data, skip_key):
  """Calculates the sum of values in a dictionary excluding a specific key.

  Args:
      data: The dictionary containing key-value pairs.
      skip_key: The key to exclude from the sum.

  Returns:
      The sum of all values in the dictionary except for the skip_key.
  """
  total = 0
  for key, value in data.items():
    if key != skip_key:
      total += float(value)
  return total

# check the seq_length
def global_extract_phn_frame_probs(data,mask_x,dic_phn_freq, dic_frame_freq, emb_dim): 
  data = pd.DataFrame(
    data    = data,
    columns = ['start', 'end', 'phoneme'])
  seq_len = emb_dim 
  mask_vector = np.zeros((emb_dim, emb_dim)) #input_features
  mask_vector_raw = np.zeros((emb_dim, 1)) #input_features
  frame_vector = np.zeros((emb_dim, emb_dim)) #input_features
  phone_vector = np.zeros((emb_dim, emb_dim)) #input_features
  a = sorted(data['phoneme'].unique())
  a = [value for value in a if value != "[SIL]"]
  dic_phn = {}
  dic_frame_timing = {}
  dic_frame_timing['total'] = sum_except_key(dic_frame_freq, '[SIL]') 
  dic_phn['total'] = sum_except_key(dic_phn_freq, '[SIL]')
  ind = 0
  count_sil = 0
  for ind, row in data.iterrows():
      p = row['phoneme']
      dic_phn[p] = dic_phn_freq[p] 
      dic_frame_timing[p] = dic_frame_freq[p]
      phn_prob = dic_phn[p] / dic_phn['total']
      frame_prob = (dic_frame_timing[p]*100) / (dic_frame_timing['total']*100)
      phn_prob = math.log(phn_prob)
      frame_prob = math.log(frame_prob)
      replace_list_phn = [phn_prob] * seq_len
      start = int(row['start'] * 100)
      end = int(row['end'] * 100)
      phone_vector[start:end] = replace_list_phn
      replace_list_frame = [frame_prob] * seq_len
      frame_vector[start:end] = replace_list_frame
      if p == "[SIL]": #p == mask_x or p == "[SIL]":
          replace_list_mask = [0] * seq_len
          mask_vector[start:end] = replace_list_mask
      else:
          replace_list_mask = [1] * seq_len
          mask_vector[start:end] = replace_list_mask
        # ind = ind + 1
  return phone_vector , frame_vector, mask_vector
