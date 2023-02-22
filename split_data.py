from All_Imports import *

# Function for splitting the data to training and validation
# and function for selecting samples of segments from the Auxiliary dataset
def split_train_val_4(data):

    idx = data['geolocation_id'] == 4
    training_x = data['iq_sweep_burst'][np.logical_not(idx)]
    training_y = data['target_type'][np.logical_not(idx)]
    validation_x = data['iq_sweep_burst'][idx]
    validation_y = data['target_type'][idx]
    return training_x, training_y, validation_x, validation_y

def split_train_val_3(data):

    idx_human_1 = random.choices(
        np.where((data['geolocation_id'] == 1) & (data['target_type'] == 1) & (data['segment_id'] < 6656))[0], k=40)
    idx_human_4 = random.choices(
        np.where((data['geolocation_id'] == 4) & (data['target_type'] == 1) & (data['segment_id'] < 6656))[0], k=40)
    idx_animal_1 = random.choices(
        np.where((data['geolocation_id'] == 1) & (data['target_type'] == 0) & (data['segment_id'] < 6656))[0], k=20)
    idx_animal_2 = random.choices(
        np.where((data['geolocation_id'] == 2) & (data['target_type'] == 0) & (data['segment_id'] < 6656))[0], k=20)
    idx_animal_3 = random.choices(
        np.where((data['geolocation_id'] == 3) & (data['target_type'] == 0) & (data['segment_id'] < 6656))[0], k=20)
    idx_animal_4 = random.choices(
        np.where((data['geolocation_id'] == 4) & (data['target_type'] == 0) & (data['segment_id'] < 6656))[0], k=20)
    idx_list = idx_human_1 + idx_human_4 + idx_animal_1 + idx_animal_2 + idx_animal_3 + idx_animal_4

    mask = np.ones(data['iq_sweep_burst'].shape[0], dtype=bool)  # np.ones_like(a,dtype=bool)
    mask[idx_list] = False
    training_x = data['iq_sweep_burst'][mask]
    training_y = data['target_type'][mask]
    validation_x = data['iq_sweep_burst'][~mask]
    validation_y = data['target_type'][~mask]
    return training_x, training_y, validation_x, validation_y



def split_train_val_2(data):
    """
    I want about 200 samples in cv
    100 human - 25 from 1 , 25 from 4 - both from training set , and then 50 randomly from the experiment
    100 animals - 25 from 1 to 4
    """
    # I will need to change to data['target_type'] == 'human') if labels are not change in the pre-process
    idx_human_1 = random.choices(np.where( (data['geolocation_id'] == 1) & (data['target_type'] == 1) & (data['segment_id'] < 6656) )[0]  , k=25)
    idx_human_4 = random.choices(np.where( (data['geolocation_id'] == 4) & (data['target_type'] == 1) & (data['segment_id'] < 6656) )[0]  , k=25)
    idx_human_exp = random.choices(np.where( (data['target_type'] == 1) & (data['segment_id'] > 6656) )[0]  , k=50)
    idx_animal_1 = random.choices(np.where( (data['geolocation_id'] == 1) & (data['target_type'] == 0) & (data['segment_id'] < 6656) )[0]  , k=25)
    idx_animal_2 = random.choices(np.where( (data['geolocation_id'] == 2) & (data['target_type'] == 0) & (data['segment_id'] < 6656) )[0]  , k=25)
    idx_animal_3 = random.choices(np.where( (data['geolocation_id'] == 3) & (data['target_type'] == 0) & (data['segment_id'] < 6656) )[0]  , k=25)
    idx_animal_4 = random.choices(np.where( (data['geolocation_id'] == 4) & (data['target_type'] == 0) & (data['segment_id'] < 6656) )[0]  , k=25)
    idx_list = idx_human_1+idx_human_4+idx_human_exp+idx_animal_1+idx_animal_2+idx_animal_3+idx_animal_4
    
    mask = np.ones(data['iq_sweep_burst'].shape[0], dtype=bool)  # np.ones_like(a,dtype=bool)
    mask[idx_list] = False
    training_x = data['iq_sweep_burst'][mask]
    training_y = data['target_type'][mask]
    validation_x = data['iq_sweep_burst'][~mask]
    validation_y = data['target_type'][~mask]
    return training_x, training_y, validation_x, validation_y



def split_train_val(data):
    """
    Split the data to train and validation set.
    The validation set is built from training set segments of
    geolocation_id 1 and 4.
    Use the function only after the training set is complete and preprocessed.

    Arguments:
      data -- {ndarray} -- the data set to split

    Returns:
      iq_sweep_burst ndarray matrices
      target_type vector
      for training and validation sets
    """
    idx = ((data['geolocation_id'] == 4) | (data['geolocation_id'] == 1)) \
          & (data['segment_id'] % 6 == 0)
    training_x = data['iq_sweep_burst'][np.logical_not(idx)]
    training_y = data['target_type'][np.logical_not(idx)]
    validation_x = data['iq_sweep_burst'][idx]
    validation_y = data['target_type'][idx]
    return training_x, training_y, validation_x, validation_y

def split_train_val_new(data):
    """
    Split the data to train and validation set.
    The validation set is built from training set segments of
    geolocation_id 1 and 4.
    Use the function only after the training set is complete and preprocessed.

    Arguments:
      data -- {ndarray} -- the data set to split

    Returns:
      iq_sweep_burst ndarray matrices
      target_type vector
      for training and validation sets
    """
    # idx = ((data['geolocation_id'] == 4) | (data['geolocation_id'] == 1)) \
    #       & (data['segment_id'] % 16 == 0) #16 rather than 6 because I want periodicty (it will see the segment in the synthesis..) + %6 is way to much!
    idx = ((data['geolocation_id'] == 4) | (data['geolocation_id'] == 1)) \
          & (data['segment_id'] % 16 == 0) & ((data['segment_id'] < 6656) | (data['segment_id'] > 6939)) #16 rather than 6 because I want periodicty (it will see the segment in the synthesis..) + %6 is way to much!
    training_x = data['iq_sweep_burst'][np.logical_not(idx)]
    training_y = data['target_type'][np.logical_not(idx)]
    validation_x = data['iq_sweep_burst'][idx]
    validation_y = data['target_type'][idx]
    return training_x, training_y, validation_x, validation_y

def split_train_val_new_improve(data):
    """
    """
    idx = ((data['geolocation_id'] == 4) | (data['geolocation_id'] == 1)) \
          & (data['segment_id'] % 16 == 0) #16 rather than 6 because I want periodicty (it will see the segment in the synthesis..) + %6 is way to much!
    training_x_MagFFT = data['MagFFT'][np.logical_not(idx)]
    training_x_ResidualPhase = data['ResidualPhase'][np.logical_not(idx)]
    validation_x_MagFFT = data['MagFFT'][idx]
    validation_x_ResidualPhase = data['ResidualPhase'][idx]
    training_y = data['target_type'][np.logical_not(idx)]
    validation_y = data['target_type'][idx]
    return training_x_MagFFT,training_x_ResidualPhase,training_y,validation_x_MagFFT,validation_x_ResidualPhase,validation_y


def aux_split(data):
    """
    Selects segments from the auxilary set for training set.
    Takes the first 3 segments (or less) from each track.

    Arguments:
      data {dataframe} -- the auxilary data

    Returns:
      The auxilary data for the training
    """
    idx = (data['segment_id'] % 4 == 0)
    for key in data:
        data[key] = data[key][idx]
    return data

# I think there is bug here:
#   idx = np.bool_(np.zeros(len(data['track_id'])))
#   #221 - print(np.unique(data['track_id']).size)#zahi
#   for track in np.unique(data['track_id']):
#     idx |= data['segment_id']==(data['segment_id'][data['track_id'] == track][:3])

#   for key in data:
#     data[key] = data[key][idx]
#   return data