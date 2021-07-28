  
# -*- coding: utf-8 -*-

"""Save Precision and Recall graphs for the best model."""

import matplotlib.pyplot as plt

def save_graphs(train_dir, val_dir, writer):
    #Plot P&R curve for training
    fig1 = plt.figure() 
    plt.plot(train_dir['recall'], train_dir['precision'], color='m')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Best Train Precision and Recall')
    plt.close('all')

    #Plot P&R curve for val
    fig2 = plt.figure() 
    plt.plot(val_dir['recall'], val_dir['precision'], color='c')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Best Val Precision and Recall')
    plt.close('all')

    writer.add_figure('Train Precision and Recall', fig1)
    writer.add_figure('Val Precision and Recall', fig2)