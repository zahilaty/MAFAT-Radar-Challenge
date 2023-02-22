from All_Imports import *
from Metrics import *
from load_data import append_dict
from PreProcess import Base_process
from split_data import split_train_val,split_train_val_new,aux_split
import My_models
resnet = My_models.Adjust_ResNet18();resnet.load_state_dict(torch.load('Backups\\Resnet18_correctCV_0.9072164948453608_epochs_10.pth'))
googlenet = My_models.Adjust_googLeNet();googlenet.load_state_dict(torch.load("Backups\\GoogleNet_correctCV_0.8814432989690721_epochs_6.pth"))
net1 = My_models.Net1();net1.load_state_dict(torch.load("Backups\\Net1_correctCV_0.8917525773195877_epochs_21.pth"))
phasenet = My_models.MagPhaseNet();phasenet.load_state_dict(torch.load("Backups\\PhaseNet_0.8865979381443299_epochs_6.pth"))

NetsList = [resnet,googlenet,net1,phasenet]

####### Loading  data #########
with open("Data\\train_processed.pkl", 'rb') as data:
    train_df = pickle.load(data)
with open("Data\\syn_processed.pkl", 'rb') as data:
    syn_df = pickle.load(data)
with open("Data\\exp_processed.pkl", 'rb') as data:
    exp_df = aux_split(pickle.load(data))
with open("Data\\test_processed.pkl", 'rb') as data:
    test_df = pickle.load(data)
print('----- Done loading -----')
train_x, train_y, val_x, val_y = split_train_val_new(  append_dict(append_dict(train_df,syn_df) ,exp_df)   )
del train_df,syn_df,exp_df,train_x,train_y

val_x, val_y = Base_process(val_x,val_y)
Mcv = val_x.shape[0]
net1.eval().cpu()
resnet.eval().cpu()
googlenet.eval().cpu()
phasenet.eval().cpu()

val_y = val_y.detach().numpy()
for net in NetsList:
    pred_cv = net(val_x).detach().numpy()