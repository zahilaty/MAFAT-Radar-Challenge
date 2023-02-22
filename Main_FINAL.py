### Same as main 1, but for resnet

from All_Imports import *
from Metrics import *
from load_data import append_dict
from PreProcess import Base_process
from split_data import split_train_val,split_train_val_new,aux_split
import My_models

####### Hyper-params #########
Lr = 0.001#0.001#0.5
#Weight_decay = 0.00 #Regularization ("lambda" in most books)
alpha = 0.15
gamma = 3
Betas = (0.9,0.999) #momentum and RMSprop of adam optimizer
NumOfEpoch = 0 #Num of itertions on all training set
AugmantionFlag = True
AddCVToTrain = False
Th = 0.89
net = My_models.Adjust_ResNet18();net.load_state_dict(torch.load('Backups\\NewRes\\NetW_correctCV_0.9072164948453608_epochs_13.pth'))
#net = My_models.Adjust_ResNet18();net.load_state_dict(torch.load('Backups\\NewRes2\\NetW_correctCV_0.9020618556701031_epochs_10.pth'))
#net = My_models.MyEfficientNet();net.load_state_dict(torch.load('Backups\\LR_5e-4_adam\\NetW_correctCV_0.9020618556701031_epochs_8.pth'))
#net = My_models.MyEfficientNet();net.load_state_dict(torch.load('Backups\\Lr_1e-3\\NetW_correctCV_0.8969072164948454_epochs_10.pth'))
BatchSize = 256

weight_conv, bias_conv, weight_fc, bias_fc = GetWeightBiasParams(net)
# bias_p weight decay was determined according to keras l2 regulizer, altoug it is not the same thing when using adam
#optimizer = torch.optim.Adam([{'params': weight_conv, 'weight_decay':0},{'params': bias_conv, 'weight_decay':1e-3},{'params': weight_fc, 'weight_decay':1e-3},{'params': bias_fc, 'weight_decay':0}], lr=Lr, betas=Betas)
optimizer = torch.optim.Adam([{'params': weight_conv, 'weight_decay':0},{'params': bias_conv, 'weight_decay':0},{'params': weight_fc, 'weight_decay':0},{'params': bias_fc, 'weight_decay':0}], lr=Lr, betas=Betas)
#optimizer = torch.optim.SGD([{'params': weight_conv, 'weight_decay':0},{'params': bias_conv, 'weight_decay':0},{'params': weight_fc, 'weight_decay':0},{'params': bias_fc, 'weight_decay':0}], lr=Lr,momentum=0.6)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,2,4,8], gamma=0.75)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,15], gamma=0.5)

####### Loading  data #########
with open("Data\\train_processed.pkl", 'rb') as data:
    train_df = pickle.load(data)
with open("Data\\syn_processed.pkl", 'rb') as data:
    syn_df = pickle.load(data)
with open("Data\\exp_processed.pkl", 'rb') as data:
    exp_df = aux_split(pickle.load(data))
# with open("Data\\test_processed.pkl", 'rb') as data:
#     test_df = pickle.load(data)
with open("Data\\FULL_processed.pkl", 'rb') as data:
    FULL_df = pickle.load(data)
with open("Data\\private_processed.pkl", 'rb') as data:
    private_df = pickle.load(data)

print('----- Done loading -----')
train_x, train_y, val_x, val_y = split_train_val_new( append_dict( FULL_df ,  append_dict(append_dict(train_df,syn_df) ,exp_df)  ) )
del train_df,syn_df,exp_df

####### Run Only For Submission #########
if AddCVToTrain is True:
    train_x = np.concatenate((train_x, val_x), axis=0, out=None)
    train_y = np.concatenate((train_y, val_y), axis=0, out=None)

####### Add Augmanted pictures for animals #########
if AugmantionFlag is True:
    VelocityFlip = np.flip(train_x[train_y==0,:,:],axis=1)
    TimeFlip = np.flip(train_x[train_y==0,:,:],axis=2)
    print("Adding augmantaion of %d VelocityFlip and %d TimeFlip" % (VelocityFlip.shape[0], TimeFlip.shape[0]))
    train_x = np.concatenate((train_x, VelocityFlip, TimeFlip), axis=0, out=None)
    train_y = np.concatenate((train_y, np.zeros(VelocityFlip.shape[0]+TimeFlip.shape[0],)), axis=0, out=None)
    # train_x = np.concatenate((train_x, VelocityFlip), axis=0, out=None)
    # train_y = np.concatenate((train_y, np.zeros(VelocityFlip.shape[0],)), axis=0, out=None)

####### shuffle for good luck #########
randperm = np.random.permutation(train_x.shape[0])
train_x = train_x[randperm,:,:]
train_y = train_y[randperm]
train_x, train_y = Base_process(train_x, train_y)
val_x, val_y = Base_process(val_x,val_y)

###### same for the test ######
test_x = private_df['iq_sweep_burst']
test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1], test_x.shape[2])
test_x = torch.from_numpy(test_x.astype('float32'))
submission = pd.DataFrame()
submission['segment_id'] = private_df['segment_id']

####### Extract some usefull params #########
print("train_x shape is " + str(train_x.shape))
print("train_y shape is " + str(train_y.shape))
Mtr = train_x.shape[0]
Mcv = val_x.shape[0]
print("There are " + str(Mtr) + " Training examples and " + str(Mcv) + " CV examples and ")

print('----- Start Training -----')
NumOfBatches = int(np.ceil(Mtr / BatchSize))
Costs = np.array([])
Corrects = np.array([])
Corrects_cv = np.array([])
t0 = time.time()
MaxCV = -1
for Epoch in range(NumOfEpoch):
    randperm = np.random.permutation(train_x.shape[0])
    train_x = train_x[randperm, :, :]
    train_y = train_y[randperm]
    for Batch in range(NumOfBatches):
        st_ind = Batch * BatchSize
        end_ind = min(st_ind + BatchSize, Mtr)
        x = train_x[st_ind:end_ind, :, :]
        y = train_y[st_ind:end_ind]

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            net = net.cuda()
        # if Half:
        #     x = x.half()
        #     net.half()

        optimizer.zero_grad()
        prediction = net(x)

        loss = torch.mean(-alpha * ((1 - prediction) ** gamma) * torch.log(prediction+torch.tensor(1e-10)) * y - (1 - alpha) * (prediction ** gamma) * torch.log(1 - prediction+torch.tensor(1e-10)) * (1 - y))
        correct = ((torch.round(prediction) == y).sum().item()) / (y.shape[0])  # dont forget to consider the actual batch size
        Costs = np.append(Costs, loss.cpu().detach().numpy())
        Corrects = np.append(Corrects, correct)

        # Backward and optimizer step
        loss.backward()
        optimizer.step()

        net = net.eval()
        # if True: #run on cpu to avoid memory issues
        #     net.cpu()
        #     pred_cv = net(val_x)
        #     correct_cv = (torch.round(pred_cv) == val_y).sum().item() / Mcv
        #     net.cuda()
        # else:
        pred_cv = net(val_x.cuda()) #.half()
        correct_cv = (torch.round(pred_cv) == val_y.cuda()).sum().item() / Mcv
        Corrects_cv = np.append(Corrects_cv,correct_cv)

        print('Epoch num: %d , Batch num: %d , loss: %.4f , corrects_tr: %.3f, corrects_cv: %.3f' % (Epoch, Batch, loss.item(), correct,correct_cv))
        if (correct_cv > MaxCV) & (correct_cv > Th):
            print('starting fine tuning')
            MaxCV = correct_cv
            Th = Th + 0.002
            #optimizer.betas = (0.3, 0.999)
            #optimizer.lr = 1e-4
            fname = "./NetW_correctCV_" + str(correct_cv)+ "_epochs_" + str(Epoch + 1) + ".pth"
            torch.save(net.state_dict(), fname)
            net.eval().cpu()
            submission['prediction'] = net(test_x).detach().numpy()
            submission['prediction'] = submission['prediction'].astype('float')
            submission.to_csv('submission.csv', index=False)
            with ZipFile('submission.zip', 'w') as myzip:
                myzip.write('submission.csv')
            net.train().cuda()

        net = net.train()

    scheduler.step()
    fname = "./SomeBigNet_" + str(Epoch + 1) + "_epochs.pth"
    torch.save(net.state_dict(), fname)

elapsed = time.time() - t0
print('----- finish Training -----')
print('Ellapsed = ', elapsed, '[sec]')
# if Half:
#     net.float()

net.eval().cpu()
pred_cv = net(val_x)
stats_single(pred_cv.detach().numpy(), val_y.detach().numpy())

####### Figures #########
plt.figure()
plt.plot(Costs, color='tab:blue',label = 'Loss')
plt.legend();plt.xlabel('Iterations[#]');plt.ylabel('Cost');plt.title('Loss function');plt.grid()
#plt.savefig("cost.png")
plt.figure()
plt.plot(Corrects, color='tab:green',label = 'Corrects')
plt.plot(Corrects_cv, color='tab:red',label = 'Corrects_cv')
plt.legend();plt.xlabel('Iterations[#]');plt.ylabel('Accuracy');plt.title('Accuracy');plt.grid()
#plt.savefig("Accuracy.png")