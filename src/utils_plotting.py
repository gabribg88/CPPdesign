import os
import re
import sys
import random
#import colors
import warnings
import matplotlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import t
from math import factorial
#from texttable import Texttable
from itertools import combinations
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, \
                            average_precision_score,precision_score, recall_score, f1_score, matthews_corrcoef, fbeta_score

import matplotlib.pyplot as plt


def plot_cv_roc_curve(preds_list, name_list, ci=2):
    fig, ax = plt.subplots(figsize=(7,7))
    mean_auroc_list = []
    std_auroc_list = []
    mean_list = []
    sigma_list = []
    for i, (preds, name) in enumerate(zip(preds_list, name_list)):
        tpr_list = []
        auroc_list = []
        base_fpr = np.linspace(0, 1, 101)

        for n_fold, (true_fold, preds_fold) in enumerate(preds):
            fpr, tpr, thresholds = roc_curve(true_fold, preds_fold, pos_label=1)
            auroc = auc(fpr, tpr) # same as roc_auc_score(true_fold, preds_fold)
            interp_tpr = np.interp(base_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tpr_list.append(interp_tpr)
            auroc_list.append(auroc)

        mean_tpr = np.mean(tpr_list, axis=0)
        mean_tpr[-1] = 1.0
        mean_auroc = auc(base_fpr, mean_tpr)
        std_auroc = np.std(auroc_list)
        mean_auroc_list.append(mean_auroc), std_auroc_list.append(std_auroc)
        std_tpr = np.std(tpr_list, axis=0)
        tpr_upper = np.minimum(mean_tpr + ci*std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - ci*std_tpr, 0)

        mean = ax.plot(base_fpr, mean_tpr, color=f'C{i}', linewidth=3)
        sigma = ax.fill_between(base_fpr, tpr_lower, tpr_upper, color=f'C{i}', alpha=0.2)
        mean_list.append(mean), sigma_list.append(sigma)

    ax.legend([(mean[0], sigma) for mean, sigma in zip(mean_list, sigma_list)],
              [r'Mean $\pm$ {ci} std. dev. ROC {name} (AUROC = {mean:.2f} $\pm$ {std:.2f})'.format(ci=ci, name=name, mean=mean_auroc, std=ci*std_auroc)
               for name, mean_auroc, std_auroc in zip(name_list, mean_auroc_list, std_auroc_list)], loc='lower right')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.01])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid()
    
def plot_cv_pr_curve(preds_list, name_list, ci=2):
    fig, ax = plt.subplots(figsize=(7,7))
    mean_auprc_list = []
    std_auprc_list = []
    mean_list = []
    sigma_list = []
    for i, (preds, name) in enumerate(zip(preds_list, name_list)):
        precision_list = []
        auprc_list = []
        base_recall = np.linspace(0, 1, 101)

        for n_fold, (true_fold, preds_fold) in enumerate(preds):
            precision, recall, thresholds = precision_recall_curve(true_fold, preds_fold, pos_label=1)
            auprc = average_precision_score(true_fold, preds_fold) # invece di auc(recall, precision) si veda:
            # https://stackoverflow.com/questions/29656550/how-to-plot-pr-curve-over-10-folds-of-cross-validation-in-scikit-learn
            precision, recall, thresholds = precision[::-1], recall[::-1], thresholds[::-1]
            interp_precision = np.interp(base_recall, recall, precision)
            precision_list.append(interp_precision)
            auprc_list.append(auprc)

        mean_precision = np.mean(precision_list, axis=0)
        mean_auprc = np.mean(auprc_list)#auc(base_recall, mean_precision)
        std_auprc = np.std(auprc_list)
        mean_auprc_list.append(mean_auprc), std_auprc_list.append(std_auprc)
        std_precision = np.std(precision_list, axis=0)
        precision_upper = np.minimum(mean_precision + ci*std_precision, 1)
        precision_lower = np.maximum(mean_precision - ci*std_precision, 0)

        mean = ax.plot(base_recall, mean_precision, color=f'C{i}', linewidth=3)
        sigma = ax.fill_between(base_recall, precision_lower, precision_upper, color=f'C{i}', alpha=0.2)
        mean_list.append(mean), sigma_list.append(sigma)

    ax.legend([(mean[0], sigma) for mean, sigma in zip(mean_list, sigma_list)],
              [r'Mean $\pm$ {ci} std. dev. PR {name} (AUPRC = {mean:.2f} $\pm$ {std:.2f})'.format(ci=ci, name=name, mean=mean_auprc, std=ci*std_auprc)
               for name, mean_auprc, std_auprc in zip(name_list, mean_auprc_list, std_auprc_list)], loc='lower right')
    #ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlim([0.0, 1.01])
    ax.set_ylim([0.0, 1.01])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.grid()
    
    
    
    
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from sklearn.metrics import confusion_matrix as cm_sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#def plot_discrimination_threshold(clf, X_test, y_test, argmax='f1', title='Metrics vs Discriminant Threshold', fig_size=(10, 8), dpi=100, save_fig_path=None):
def plot_discrimination_threshold(y_test_probs, y_test, beta=1.0, argmax='f1', avg='macro', title='Metrics vs Discriminant Threshold', fig_size=(10, 8), dpi=100, save_fig_path=None):
    """
    Plot precision, recall and f1-score vs discriminant threshold for the given pipeline model
    Parameters
    ----------
    clf : estimator instance (either sklearn.Pipeline, imblearn.Pipeline or a classifier)
        PRE-FITTED classifier or a PRE-FITTED Pipeline in which the last estimator is a classifier.
    X_test : pandas.DataFrame of shape (n_samples, n_features)
        Test features.
    y_test : pandas.Series of shape (n_samples,)
        Target values.
    argmax : str, default: 'f1'
        Annotate the threshold maximized by the supplied metric. Options: 'f1', 'precision', 'recall'
    title : str, default ='FPR and FNR vs Discriminant Threshold'
        Plot title.
    fig_size : tuple, default = (10, 8)
        Size (inches) of the plot.
    dpi : int, default = 100
        Image DPI.
    save_fig_path : str, defaut=None
        Full path where to save the plot. Will generate the folders if they don't exist already.
    Returns
    -------
        fig : Matplotlib.pyplot.Figure
            Figure from matplotlib
        ax : Matplotlib.pyplot.Axe
            Axe object from matplotlib
    """
        
    thresholds = np.linspace(0, 1, 1000)
    
    precision_ls = []
    recall_ls = []
    f1_ls = []
    fpr_ls = []
    fnr_ls = []
    mcc_ls = []
    fbeta_ls = []
    
    # obtain probabilities
    probs = y_test_probs#[:,1]#clf.predict_proba(X_test)[:,1]

    for threshold in thresholds:   
    
        # obtain class prediction based on threshold
        y_predictions = np.where(probs>=threshold, 1, 0) 
        
        # obtain confusion matrix
        tn, fp, fn, tp = cm_sklearn(y_test, y_predictions).ravel()
        
        # obtain FRP and FNR
        FPR = fp / (tn + fp)
        FNR = fn / (tp + fn)
        
        # obtain precision, recall and f1 scores
        precision = precision_score(y_test, y_predictions, average=avg)
        recall = recall_score(y_test, y_predictions, average=avg)
        f1 = f1_score(y_test, y_predictions, average=avg)
        mcc = matthews_corrcoef(y_test, y_predictions)
        fbeta = fbeta_score(y_test, y_predictions, beta=beta, average=avg)
         
        precision_ls.append(precision)
        recall_ls.append(recall)
        f1_ls.append(f1)
        fpr_ls.append(FPR)
        fnr_ls.append(FNR)
        mcc_ls.append(mcc)
        fbeta_ls.append(fbeta)
              
    metrics = pd.concat([
        pd.Series(precision_ls),
        pd.Series(recall_ls),
        pd.Series(f1_ls),
        pd.Series(mcc_ls),
        pd.Series(fpr_ls),
        pd.Series(fnr_ls),
        pd.Series(fbeta_ls)], axis=1)

    metrics.columns = ['precision', 'recall', 'f1', 'mcc', 'fpr', 'fnr', 'fbeta']
    metrics.index = thresholds
    
    plt.rcParams["figure.facecolor"] = 'white'
    plt.rcParams["axes.facecolor"] = 'white'
    plt.rcParams["savefig.facecolor"] = 'white'
                
    fig, ax = plt.subplots(1, 1, figsize=fig_size, dpi=dpi)
    ax.plot(metrics['precision'], label='Precision')
    ax.plot(metrics['recall'], label='Recall')
    ax.plot(metrics['f1'], label='f1')
    ax.plot(metrics['mcc'], label='MCC')
    ax.plot(metrics['fbeta'], label='F-beta score')
    ax.plot(metrics['fpr'], label='False Positive Rate (FPR)', linestyle='dotted')
    ax.plot(metrics['fnr'], label='False Negative Rate (FNR)', linestyle='dotted')
    
    
    # Draw a threshold line
    disc_threshold = round(metrics[argmax].idxmax(), 3)
    ax.axvline(x=metrics[argmax].idxmax(), color='black', linestyle='dashed', label="$t_r$="+str(disc_threshold))

    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter('{x:.1f}')
    
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter('{x:.1f}')

    ax.xaxis.set_minor_locator(MultipleLocator(0.05))    
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))    

    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4, color='black') 
    
    plt.grid(True)
    
    plt.xlabel('Probability Threshold', fontsize=18)
    plt.ylabel('Scores', fontsize=18)
    plt.title(title, fontsize=18)
    leg = ax.legend(loc='best', frameon=True, framealpha=0.7)
    leg_frame = leg.get_frame()
    leg_frame.set_color('gold')
    plt.show()

    if (save_fig_path != None):
        path = pathlib.Path(save_fig_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, dpi=dpi)

    return fig, ax, disc_threshold


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

def plot_confusion_matrix(true, preds):

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(true, preds)).plot(cmap='Blues', colorbar=None, values_format='')
    for labels in disp.text_.ravel():
        labels.set_fontsize(16)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(true, preds, normalize='true')).plot(cmap='Blues', colorbar=None, values_format='.2f')
    for labels in disp.text_.ravel():
        labels.set_fontsize(16)
    plt.show()