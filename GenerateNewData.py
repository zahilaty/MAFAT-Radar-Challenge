import pickle
from load_data import *
#from My_models import *
from PreProcess import *

######### FFT ONLY #############
# train_df = load_data(train_path)
# train_df = data_preprocess(train_df)
# with open("Data\\train_processed.pkl", 'wb') as data:
#     pickle.dump(train_df, data)
# del train_df
#
# syn_df = load_data(syn_auxiliary)
# syn_df = data_preprocess(syn_df)
# with open("Data\\syn_processed.pkl", 'wb') as data:
#     pickle.dump(syn_df, data)
# del syn_df
#
# exp_df = load_data(experiment_auxiliary)
# exp_df = data_preprocess(exp_df)
# with open("Data\\exp_processed.pkl", 'wb') as data:
#     pickle.dump(exp_df, data)
# del exp_df
#
# back_df = load_data(background_path)
# back_df = data_preprocess(back_df)
# with open("Data\\back_processed.pkl", 'wb') as data:
#     pickle.dump(back_df, data)
# del back_df
#
# test_df = load_data(test_path)
# test_df = data_preprocess(test_df)
# with open("Data\\test_processed.pkl", 'wb') as data:
#     pickle.dump(test_df, data)
# del test_df

private_df = load_data(private_path)
private_df = data_preprocess(private_df)
with open("Data\\private_processed.pkl", 'wb') as data:
    pickle.dump(private_df, data)
del private_df

FULL_df = load_data(FULL_path)
FULL_df = data_preprocess(FULL_df)
with open("Data\\FULL_processed.pkl", 'wb') as data:
    pickle.dump(FULL_df, data)
del FULL_df

########## MAG AND PHASE ##########
# train_df = load_data(train_path)
# train_df = MagAngleProcess(train_df)
# train_df.pop('iq_sweep_burst', None)
# with open("Data\\train_processed_improve.pkl", 'wb') as data:
#     pickle.dump(train_df, data)
# del train_df
#
# syn_df = load_data(syn_auxiliary)
# syn_df = MagAngleProcess(syn_df)
# syn_df.pop('iq_sweep_burst', None)
# with open("Data\\syn_processed_improve.pkl", 'wb') as data:
#     pickle.dump(syn_df, data)
# del syn_df
#
# exp_df = load_data(experiment_auxiliary)
# exp_df = MagAngleProcess(exp_df)
# exp_df.pop('iq_sweep_burst', None)
# with open("Data\\exp_processed_improve.pkl", 'wb') as data:
#     pickle.dump(exp_df, data)
# del exp_df
#
# back_df = load_data(background_path)
# back_df = MagAngleProcess(back_df)
# back_df.pop('iq_sweep_burst', None)
# with open("Data\\back_processed_improve.pkl", 'wb') as data:
#     pickle.dump(back_df, data)
# del back_df
#
# test_df = load_data(test_path)
# test_df = MagAngleProcess(test_df)
# test_df.pop('iq_sweep_burst', None)
# with open("Data\\test_processed_improve.pkl", 'wb') as data:
#     pickle.dump(test_df, data)
# del test_df
# #favorite_color = pickle.load(open("save.p", "rb"))
#
#

