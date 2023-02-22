from All_Imports import *

# Function for calculating the final ROC-AUC score and plot the ROC curve,
# used in the "Results" section
def stats(pred, actual):
  """
  Computes the model ROC-AUC score and plots the ROC curve.

  Arguments:
    pred -- {ndarray} -- model's probability predictions
    actual -- the true lables

  Returns:
    ROC curve graph and ROC-AUC score
  """
  plt.figure(figsize=(20, 10))
  fpr1, tpr1, _ = roc_curve(actual[0], pred[0])
  fpr2, tpr2, _ = roc_curve(actual[1], pred[1])
  roc_auc = [auc(fpr1, tpr1), auc(fpr2, tpr2)]
  lw = 2
  plt.plot(fpr1, tpr1, lw=lw, label='Training set (ROC-AUC = %0.2f)' % roc_auc[0])
  plt.plot(fpr2, tpr2, lw=lw, label='Validation set (ROC-AUC = %0.2f)' % roc_auc[1])
  plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--', label = 'Random guess')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate', fontsize=18)
  plt.ylabel('True Positive Rate', fontsize=18)
  plt.title('Training set vs. Validation set ROC curves')
  plt.legend(loc="lower right", prop = {'size': 20})
  plt.show()

#Metrics.stats_single(predicted_cv,val_actual)
def stats_single(pred, actual):
  """
  Computes the model ROC-AUC score and plots the ROC curve.

  Arguments:
    pred -- {ndarray} -- model's probability predictions
    actual -- the true lables

  Returns:
    ROC curve graph and ROC-AUC score
  """
  plt.figure(figsize=(20, 10))
  fpr1, tpr1, _ = roc_curve(actual, pred)
  roc_auc = auc(fpr1, tpr1)
  lw = 2
  plt.plot(fpr1, tpr1, lw=lw, label='ROC-AUC = %0.2f' % roc_auc)
  plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--', label = 'Random guess')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate', fontsize=18)
  plt.ylabel('True Positive Rate', fontsize=18)
  plt.title('Training set vs. Validation set ROC curves')
  plt.legend(loc="lower right", prop = {'size': 20})
  plt.show()

def SimilarityMetric(preds1,preds2,Norm = 'L1',ChoosenPairs = 'All'):
    # A function for triplet harvesting
    # dims of preds are num_of_examples X low_dim_ embedding. example: 2048 x 8
    if Norm == 'L2':
        Pairs = preds1 - preds2.reshape(preds2.shape[0],1,preds2.shape[1])
        PairsDist = torch.sum(torch.abs(Pairs) ** 2 , dim=2)

    if Norm == 'L1':
        Pairs = preds1 - preds2.reshape(preds2.shape[0],1,preds2.shape[1])
        PairsDist = torch.sum(torch.abs(Pairs) ** 1 , dim=2)

    if Norm == 'Cosine':
        #must use torch transpose, not np

        preds1_norm = preds1 / torch.sqrt(torch.sum(preds1**2,dim=1,keepdim=True))
        preds2_norm = preds2 / torch.sqrt(torch.sum(preds2**2,dim=1,keepdim=True))
        corr = torch.sum( preds1_norm * preds2_norm.reshape(preds2_norm.shape[0],1,preds2_norm.shape[1])  , dim = 2)
        PairsDist = 1 - corr

    #After this, PairsDist contain all pairs, and have a shape of MxM

    if ChoosenPairs == 'All':
        Similarity = torch.mean(PairsDist)

    if ChoosenPairs == 'Max10':
        PairsDist = torch.reshape(PairsDist,(-1,))
        PairsDist = torch.sort(PairsDist, descending = True)[0]
        Similarity = torch.sum(PairsDist[0:9]/10)

    if ChoosenPairs == 'Min10':
        PairsDist = torch.reshape(PairsDist,(-1,))
        PairsDist = torch.sort(PairsDist, descending = False)[0]
        Similarity = torch.sum(PairsDist[0:9]/10)

    return Similarity


def EarlyStopping(Costs,NumOfBatches,EarlyStoppingCost):
    BreakFlag = False
    if (Costs.size > NumOfBatches):
        if np.mean(Costs[-NumOfBatches:-1]) < EarlyStoppingCost:
            BreakFlag = True
    return BreakFlag

# Got it from here: https://discuss.pytorch.org/t/l2-regularization-with-only-weight-parameters/56853
# def GetWeightBiasParams(model):
#     weight_p, bias_p = [], []
#     for name, p in model.named_parameters():
#         if 'bias' in name:
#             bias_p += [p]
#         else:
#             weight_p += [p]
#     return weight_p,bias_p
###call it with this line  weight_p, bias_p =  GetWeightBiasParams(net)


def GetWeightBiasParams(model):
    weight_conv, bias_conv , weight_fc, bias_fc = [],[],[],[]
    for name, p in model.named_parameters():
        if 'bias' in name:
            if 'conv' in name:
                bias_conv += [p]
            else:
                bias_fc += [p]
        else:
            if 'conv' in name:
                weight_conv += [p]
            else:
                weight_fc += [p]
    return weight_conv, bias_conv , weight_fc, bias_fc


def MyRocScore(pred, actual):
    th_vec = torch.tensor(np.linspace(1.0, 0.0, num=100))
    FPR = torch.zeros(th_vec.shape)
    TPR = torch.zeros(th_vec.shape)
    Pos_inds = (actual==1)
    Fal_num = (actual==0).sum().type(torch.FloatTensor)
    Pos_num = (actual==1).sum().type(torch.FloatTensor)

    for ind,th in enumerate(th_vec):
        #tmp_logical = pred>
        FPR[ind] = (pred[~Pos_inds] > th).sum() / Fal_num
        TPR[ind] = (pred[Pos_inds] > th).sum() / Pos_num

    FPR_diff = FPR[1:] - FPR[:-1]
    return (FPR_diff*TPR[1:]).sum()

def MyF1_loss(pred, actual):
    epsilon = 1e-7

    # tp = (actual * pred).sum()
    # tn = ((1 - actual) * (1 - pred)).sum()
    # fp = ((1 - actual) * pred).sum()
    # fn = (actual * (1 - pred)).sum()
    tp = ((1-actual) * (1-pred)).sum()
    tn = ((actual) * (pred)).sum()
    fp = ((actual) * (1-pred)).sum()
    fn = ((1-actual) * (pred)).sum()
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    #print(tp,tn,fp,fn)
    #print(precision,recall)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return 1-f1

    # Fal_num = (actual==0).sum().type(torch.FloatTensor)
    # Pos_num = (actual==1).sum().type(torch.FloatTensor)
    # tp = (actual * pred).sum()
    # fp = ((1 - actual) * pred).sum()
    # return (fp*Pos_num)/(tp*Fal_num+epsilon)

def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1

