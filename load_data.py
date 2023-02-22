from All_Imports import *

def load_data(file_path,NumToLoad=-1):
    """
    Reads all data files (metadata and signal matrix data) as python dictionary,
    the pkl and csv files must have the same file name.

    Arguments:
      file_path -- {str} -- path to the iq_matrix file and metadata file

    Returns:
      Python dictionary
    """
    pkl = load_pkl_data(file_path)
    meta = load_csv_metadata(file_path)
    data_dictionary = {**meta, **pkl}
    if NumToLoad == -1:
        for key in data_dictionary.keys():
            data_dictionary[key] = np.array(data_dictionary[key])
    else:
        load_idx = random.choices(range(data_dictionary['segment_id'].shape[0]), k=NumToLoad)
        #load_idx = np.array(load_idx).reshape(-1)
        for key in data_dictionary.keys():
            data_dictionary[key] = np.array(data_dictionary[key])[load_idx,...]
    return data_dictionary


def load_pkl_data(file_path):
    """
    Reads pickle file as a python dictionary (only Signal data).

    Arguments:
      file_path -- {str} -- path to pickle iq_matrix file

    Returns:
      Python dictionary
    """
    # path = os.path.join(mount_path, competition_path, file_path + '.pkl')
    path = os.path.join('Data', file_path + '.pkl')
    print(path)
    with open(path, 'rb') as data:
        output = pickle.load(data)
    return output


def load_csv_metadata(file_path):
    """
    Reads csv as pandas DataFrame (only Metadata).

    Arguments:
      file_path -- {str} -- path to csv metadata file

    Returns:
      Pandas DataFarme
    """
    # path = os.path.join(mount_path, competition_path, file_path + '.csv')
    path = os.path.join('Data', file_path + '.csv')
    with open(path, 'rb') as data:
        output = pd.read_csv(data)
    return output

def append_dict(dict1, dict2):
  for key in dict1:
    dict1[key] = np.concatenate([dict1[key], dict2[key]], axis=0)
  return dict1