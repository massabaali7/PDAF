import pickle

def create_freq_phn(dic_phone_alignments,filename):
    frequency_dict = {}
    for k in dic_phone_alignments:
        x_list = dic_phone_alignments[k]['phoneme']
        for symbol in x_list:
            if symbol in frequency_dict:
                frequency_dict[symbol] += 1
            else:
                frequency_dict[symbol] = 1
    # Iterate over the list and count the frequency of each symbol
    with open(filename, 'wb') as handle:
        pickle.dump(frequency_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_freq_frame(dic_phone_alignments,filename):
    frequency_dict = {}
    for k in dic_phone_alignments:
        x_list = dic_phone_alignments[k]['phoneme']
        count_test = 0
        for symbol in x_list:
            if symbol in frequency_dict:
                frequency_dict[symbol] += dic_phone_alignments[k]['end'][count_test]
            else:
                frequency_dict[symbol] = dic_phone_alignments[k]['end'][count_test]
            count_test = count_test + 1 
    # Iterate over the list and count the frequency of each symbol
    with open(filename, 'wb') as handle:
        pickle.dump(frequency_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)