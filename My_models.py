import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

####################################################################################################################################################
def Adjust_ResNet18():

    # ptrblck explain:
    # model = models.resnet50(pretrained=True)
    # weight = model.conv1.weight.clone()
    # model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # with torch.no_grad():
    #     model.conv1.weight[:, :3] = weight
    #     model.conv1.weight[:, 3] = model.conv1.weight[:, 0]
    #
    # x = torch.randn(10, 4, 224, 224)
    # output = model(x)

    MyResNet18 = torchvision.models.resnet18(pretrained=True)
    weight = MyResNet18.conv1.weight[:,1,:,:].clone()
    MyResNet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 3), stride=(2, 2), padding=(3, 3), bias=False) #changed number of channels from 3 to 1. The rest is the same
    with torch.no_grad():
        ##MyResNet18.conv1.weight = weight #weight.reshape(weight.shape[0],1,weight.shape[1],weight.shape[2])
        MyResNet18.conv1.weight = torch.nn.Parameter(weight.reshape(weight.shape[0], 1, weight.shape[1], weight.shape[2]))
    #MyResNet18.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    #MyResNet18.fc = F.hardsigmoid(nn.Linear(in_features=512, out_features=2, bias=True))
    #MyResNet18.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    MyResNet18.fc = nn.Sequential(nn.Dropout(p=0.25), nn.Linear(in_features=512, out_features=64, bias=True),nn.Dropout(p=0.0),nn.Linear(in_features=64, out_features=1, bias=True), nn.Sigmoid())
    MyResNet18 = MyResNet18.float()
    return MyResNet18

####################################################################################################################################################
def Adjust_squeezenet1_1():
    #https://discuss.pytorch.org/t/how-to-replace-a-layer-or-module-in-a-pretrained-network/60068
    Mysqueezenet1_1 = torchvision.models.squeezenet1_1(pretrained=True)
    weight = Mysqueezenet1_1.features[0].weight[:,1,:,:].clone()
    Mysqueezenet1_1.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 3), stride=(2, 2), padding=(3, 3), bias=False) #changed number of channels from 3 to 1. kernel from 3x3 to 5x5
    with torch.no_grad():
        Mysqueezenet1_1.features[0].weight = torch.nn.Parameter(weight.reshape(weight.shape[0], 1, weight.shape[1], weight.shape[2]))
    Mysqueezenet1_1.classifier = nn.Sequential(nn.Dropout(p=0.0, inplace=False), nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1)),nn.Sigmoid(),nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    Mysqueezenet1_1 = Mysqueezenet1_1.float()
    return Mysqueezenet1_1

####################################################################################################################################################
def Adjust_googLeNet():
    #https://discuss.pytorch.org/t/how-to-replace-a-layer-or-module-in-a-pretrained-network/60068
    MyGoogleNet = torchvision.models.googlenet(pretrained=True,transform_input=False) #transform_input=False is extimliy important here because 3 channels are hard coded there!!
    weight = MyGoogleNet.conv1.conv.weight[:,1,:,:].clone()
    MyGoogleNet.conv1.conv = nn.Conv2d(1, 64, kernel_size=(7, 3), stride=(2, 2), padding=(3, 3), bias=False) #changed number of channels from 3 to 1. kernel from 3x3 to 5x5
    with torch.no_grad():
        MyGoogleNet.conv1.conv.weight = torch.nn.Parameter(weight.reshape(weight.shape[0], 1, weight.shape[1], weight.shape[2]))
    MyGoogleNet.fc = nn.Sequential(nn.Dropout(p=0.75),nn.Linear(in_features=1024, out_features=64, bias=True),nn.Dropout(p=0.2),nn.Linear(in_features=64, out_features=1, bias=True), nn.Sigmoid())
    MyGoogleNet = MyGoogleNet.float()
    return MyGoogleNet

####################################################################################################################################################
def Adjust_densenet121():

    MyDensenet121 = torchvision.models.densenet121(pretrained=True)
    weight = MyDensenet121.features.conv0.weight[:,1,:,:].clone()
    MyDensenet121.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) #changed number of channels from 3 to 1. The rest is the same
    with torch.no_grad():
        MyDensenet121.features.conv0.weight = torch.nn.Parameter(weight.reshape(weight.shape[0], 1, weight.shape[1], weight.shape[2]))
    MyDensenet121.classifier = nn.Sequential(nn.Dropout(p=0.0), nn.Linear(in_features=1024, out_features=64, bias=True),nn.Dropout(p=0.0),nn.Linear(in_features=64, out_features=1, bias=True), nn.Sigmoid())
    MyDensenet121 = MyDensenet121.float()
    return MyDensenet121

####################################################################################################################################################
def Adjust_vgg11_bn():

    Myvgg11 = torchvision.models.vgg11_bn(pretrained=True)
    weight = Myvgg11.features[0].weight[:,1,:,:].clone()
    Myvgg11.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 3), stride=(2, 2), padding=(3, 3), bias=False)
    with torch.no_grad():
        Myvgg11.features[0].weight = torch.nn.Parameter(weight.reshape(weight.shape[0], 1, weight.shape[1], weight.shape[2]))
    Myvgg11.classifier[6] = nn.Sequential(nn.Dropout(p=0.25), nn.Linear(in_features=4096, out_features=64, bias=True),nn.Dropout(p=0.0),nn.Linear(in_features=64, out_features=1, bias=True), nn.Sigmoid())
    Myvgg11.features[28].stride = 1
    Myvgg11.features[28].kernel_size = 1
    Myvgg11 = Myvgg11.float()
    return Myvgg11

####################################################################################################################################################
def Adjust_mobilenet_v2():

    Adjust_mobilenet_v2 = torchvision.models.mobilenet_v2(pretrained=True)
    weight = Adjust_mobilenet_v2.features[0][0].weight[:,1,:,:].clone()
    Adjust_mobilenet_v2.features[0][0] = nn.Conv2d(1, 32, kernel_size=(7, 3), stride=(2, 2), padding=(3, 3), bias=False)
    with torch.no_grad():
        Adjust_mobilenet_v2.features[0][0].weight = torch.nn.Parameter(weight.reshape(weight.shape[0], 1, weight.shape[1], weight.shape[2]))
    Adjust_mobilenet_v2.classifier[1] = nn.Sequential(nn.Linear(in_features=1280, out_features=64, bias=True),nn.Dropout(p=0.0),nn.Linear(in_features=64, out_features=1, bias=True), nn.Sigmoid())
    Adjust_mobilenet_v2 = Adjust_mobilenet_v2.float()
    return Adjust_mobilenet_v2

####################################################################################################################################################

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet,self).__init__()
        #torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(3,3),stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=(1,1))
        #torch.nn.Linear(in_features, out_features, bias=True)
        tmp_num = self.num_flat_features(F.max_pool2d(self.conv2(F.max_pool2d(self.conv1(torch.zeros((1,1,126,32))) ,kernel_size = (2,2) )),kernel_size = (2,2)))
        print('during constructor of this object the FC size was calculated to be: ' ,tmp_num)
        self.fc1 = nn.Linear(tmp_num, 128)
        self.fc2 = nn.Linear(128,32)
        self.fc3 = nn.Linear(32,1)

    def forward(self, x):
        #torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(x),kernel_size = (2,2))
        x = self.conv2(x)
        x = F.max_pool2d(F.relu(x),kernel_size = (2,2))
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

    def num_flat_features(self,x):
        size = x.size()[1:] #all dimensions except the batch dimention
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

####################################################################################################################################################

class Net1(nn.Module):
    def __init__(self):
        super(Net1,self).__init__()
        #torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(7,5),stride=(1,1)) # chainging from 3x3 to 5x5 add 5% on validation!!
        self.bn_1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(7,5),stride=(1,1)) # chainging from 3x3 to 5x5 add 5% on validation!!
        #self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(3,3),stride=(1,1))
        #self.conv2 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=(3,3),stride=(1,1))
        #torch.nn.Linear(in_features, out_features, bias=True)
        self.bn_2 = nn.BatchNorm2d(num_features=32)
        self.drop_1 = nn.Dropout(p=0.3)#0.3
        tmp_num = self.num_flat_features(F.max_pool2d(self.conv2(F.max_pool2d(self.conv1(torch.zeros((1,1,126,32))) ,kernel_size = (2,2) )),kernel_size = (2,2)))
        print('during constructor of this object the FC size was calculated to be: ' ,tmp_num)
        #self.fc1 = nn.Linear(tmp_num, 1024)
        self.fc1 = nn.Linear(tmp_num,128)
        self.drop_2 = nn.Dropout(p=0.1)#0.1
        #self.fc1_b = nn.Linear(1024, 128)
        #self.drop_2b = nn.Dropout(p=0.00)#wasnt
        self.fc2 = nn.Linear(128,32)
        self.drop_3 = nn.Dropout(p=0.0)
        self.fc3 = nn.Linear(32,1)

    def forward_to_one_before_last(self,x):
        #torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        x = self.conv1(x)
        x = self.bn_1(x)
        x = F.max_pool2d(F.relu(x),kernel_size = (2,2))
        x = self.conv2(x)
        x = self.bn_2(x)
        x = F.max_pool2d(F.relu(x),kernel_size = (2,2))
        x = self.drop_1(x)
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.drop_2(x)
        #x = F.relu(self.fc1_b(x))
        #x = self.drop_2b(x)
        x = F.tanh(self.fc2(x))#x = F.tanh(self.fc2(x))relu
        return x

    def forward(self, x,embed=False,ebmed_noise=0):
        x = self.forward_to_one_before_last(x)
        if embed:
            return x
        x = x + ebmed_noise
        x = self.drop_3(x)
        x = F.sigmoid(self.fc3(x))
        return x

    def num_flat_features(self,x):
        size = x.size()[1:] #all dimensions except the batch dimention
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

####################################################################################################################################################
class AngularPenNet(nn.Module):
    def __init__(self, num_classes=1, loss_type='arcface'):
        super(AngularPenNet, self).__init__()
        self.convlayers = RepresentNet()
        self.adms_loss = AngularPenaltySMLoss(32, 2, loss_type=loss_type)

    def forward(self, x, labels=None, mode='training'):
        assert mode in ['training', 'get_embed', 'get_pred']
        x = self.convlayers(x)
        if mode == 'get_embed':
            return x
        if mode == 'training':
            L = self.adms_loss(x, labels,mode='training')
            return L
        if mode == 'get_pred':
            x = self.adms_loss(x,labels=None,mode='get_pred')
            return x

class RepresentNet(nn.Module):
    def __init__(self):
        super(RepresentNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(3,3),stride=(1,1))
        self.bn_1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=(1,1))
        self.bn_2 = nn.BatchNorm2d(num_features=32)
        self.drop_1 = nn.Dropout(p=0.3)#0.3
        self.fc1 = nn.Linear(5760,512)
        self.drop_2 = nn.Dropout(p=0.3)#0.3
        self.fc2 = nn.Linear(512,32)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn_1(x)
        x = F.max_pool2d(F.relu(x),kernel_size = (2,2))
        x = self.conv2(x)
        x = self.bn_2(x)
        x = F.max_pool2d(F.relu(x),kernel_size = (2,2))
        x = self.drop_1(x)
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.drop_2(x)
        x = F.tanh(self.fc2(x))
        return x

    def num_flat_features(self,x):
        size = x.size()[1:] #all dimensions except the batch dimention
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss

        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599

        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels,mode='training'):
        '''
        input shape (N, in_features)
        '''
        if mode == 'training':
            assert len(x) == len(labels)
            assert torch.min(labels) >= 0
            assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)

        if mode == 'get_pred':
            return wf
        #else - this is a training mode...
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


####################################################################################################################################################

class MagPhaseNet(nn.Module):
    def __init__(self):
        super(MagPhaseNet,self).__init__()

        ### Mag ###
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(7,5),stride=(1,1)) # chainging from 3x3 to 5x5 add 5% on validation!!
        self.bn_1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(7,5),stride=(1,1)) # chainging from 3x3 to 5x5 add 5% on validation!!
        self.bn_2 = nn.BatchNorm2d(num_features=32)
        self.drop_1 = nn.Dropout(p=0.3)
        tmp_num = self.num_flat_features(F.max_pool2d(self.conv2(F.max_pool2d(self.conv1(torch.zeros((1,1,126,32))) ,kernel_size = (2,2) )),kernel_size = (2,2)))
        print('during constructor of this object the FC size was calculated to be: ' ,tmp_num)
        self.fc1 = nn.Linear(tmp_num,128)
        self.drop_2 = nn.Dropout(p=0.1)#0.1
        self.fc2 = nn.Linear(128,32)
        #self.drop_3 = nn.Dropout(p=0.0)
        #self.fc3 = nn.Linear(32,1)

        ### Ang ###
        self.ang_conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 1),stride=(1, 1))  # chainging from 3x3 to 5x5 add 5% on validation!!
        self.ang_bn_1 = nn.BatchNorm2d(num_features=8)
        self.ang_conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 1),stride=(1, 1))  # chainging from 3x3 to 5x5 add 5% on validation!!
        self.ang_bn_2 = nn.BatchNorm2d(num_features=16)
        self.ang_drop_1 = nn.Dropout(p=0.5)
        tmp_num = self.num_flat_features(F.max_pool2d(self.ang_conv2(F.max_pool2d(self.ang_conv1(torch.zeros((1, 1, 128, 32))), kernel_size=(3, 1))),kernel_size=(3, 1)))
        print('during constructor of this object the FC size was calculated to be: ', tmp_num)
        self.ang_fc1 = nn.Linear(tmp_num, 128)
        self.ang_drop_2 = nn.Dropout(p=0.1)  # 0.1
        self.ang_fc2 = nn.Linear(128, 32)
        #self.ang_drop_3 = nn.Dropout(p=0.0)
        #self.ang_fc3 = nn.Linear(32, 1)

        ### combined ###
        self.drop_final = nn.Dropout(p=0.1)
        self.fc_final = nn.Linear(64,1)

    def forward_to_one_before_last(self,x1,x2):
        #torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        ### Mag ###
        x1 = self.conv1(x1)
        x1 = self.bn_1(x1)
        x1 = F.max_pool2d(F.relu(x1),kernel_size = (2,2))
        x1 = self.conv2(x1)
        x1 = self.bn_2(x1)
        x1 = F.max_pool2d(F.relu(x1),kernel_size = (2,2))
        x1 = self.drop_1(x1)
        x1 = x1.view(-1,self.num_flat_features(x1))
        x1 = F.relu(self.fc1(x1))
        x1 = self.drop_2(x1)
        x1 = F.relu(self.fc2(x1))

        ### Ang ###
        x2 = self.ang_conv1(x2)
        x2 = self.ang_bn_1(x2)
        x2 = F.max_pool2d(F.relu(x2),kernel_size = (3,1))
        x2 = self.ang_conv2(x2)
        x2 = self.ang_bn_2(x2)
        x2 = F.max_pool2d(F.relu(x2),kernel_size = (3,1))
        x2 = self.ang_drop_1(x2)
        x2 = x2.view(-1,self.num_flat_features(x2))
        x2 = F.relu(self.ang_fc1(x2))
        x2 = self.ang_drop_2(x2)
        x2 = F.relu(self.ang_fc2(x2))

        return torch.cat((x1,x2), dim=1)
        # return x2

    def forward(self, x1,x2):
        x = self.forward_to_one_before_last(x1,x2)
        x = self.drop_final(x)
        x = F.sigmoid(self.fc_final(x))
        return x

    def num_flat_features(self,x):
        size = x.size()[1:] #all dimensions except the batch dimention
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

####################################################################################################################################################
def Adjust_EfficientNet():
    model = EfficientNet.from_pretrained('efficientnet-b0')
    weight = model._conv_stem.weight[:, 1, :, :].clone()
    model._conv_stem.in_channels = 1
    with torch.no_grad():
        model._conv_stem.weight = torch.nn.Parameter(weight.reshape(weight.shape[0], 1, weight.shape[1], weight.shape[2]))
    # I wanted to reduce the size here,  but there is some issues with the gpu-effient-memory module
    # this is not working:
    # weight = model._fc.weight[:64,:].clone()
    # model._fc.out_features = 64
    # with torch.no_grad():
    #     model._fc.weight = torch.nn.Parameter(weight)
    return model

class MyEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.eff = Adjust_EfficientNet() #now we have a net with 1 channel input and outputs size = 1000
        self.classi = nn.Sequential(nn.Dropout(p=0.25), nn.Linear(in_features=1000, out_features=64, bias=True),nn.Tanh(),nn.Linear(in_features=64, out_features=1, bias=True),nn.Sigmoid())
        #self.classi = nn.Sequential(nn.Tanh(),nn.Linear(in_features=64, out_features=1, bias=True),nn.Sigmoid()) ## this is not working
    def forward(self,x):
        x = self.eff(x)
        #print(x.size()) just for debug
        x = self.classi(x)
        return x



