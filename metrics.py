from math import log10
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from scipy.special import softmax

def PSNR(mse, peak=1.):
	return 10 * log10((peak ** 2) / mse)

class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def accuracy(preds, labels):
    """Accuracy, auc with masking.Acc of the masked samples"""
    correct_prediction = np.equal(np.argmax(preds, 1), labels).astype(np.float32)
    return np.sum(correct_prediction), np.mean(correct_prediction)

def auc(preds, labels, is_logit=True):
    ''' input: logits, labels  ''' 
    if is_logit:
        pos_probs = softmax(preds, axis=1)[:, 1]
    else:
        pos_probs = preds[:,1]
    try:
        auc_out = roc_auc_score(labels, pos_probs)
    except:
        auc_out = 0
    return auc_out

def prf(preds, labels, is_logit=True):
    ''' input: logits, labels  ''' 
    pred_lab= np.argmax(preds, 1)
    p, r, f, s = precision_recall_fscore_support(labels, pred_lab, average='macro', zero_division=1)
    return [p,r,f]


def numeric_score(preds, labels):
    FP = np.float(np.sum((preds == 1) & (labels == 0)))
    FN = np.float(np.sum((preds == 0) & (labels == 1)))
    TP = np.float(np.sum((preds == 1) & (labels == 1)))
    TN = np.float(np.sum((preds == 0) & (labels == 0)))
    return FP, FN, TP, TN

def metrics(preds, labels):
    preds = np.argmax(preds, 1) 
    FP, FN, TP, TN = numeric_score(preds, labels)
    sen = TP / (TP + FN + 1e-10)
    spe = TN / (TN + FP + 1e-10)


    return sen, spe

