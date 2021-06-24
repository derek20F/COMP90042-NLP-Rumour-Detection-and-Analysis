# -*- coding: utf-8 -*-
"""
Created on Tue May 11 09:58:44 2021

This code is used to plot the training curve and the development curve.
"""

import matplotlib.pyplot as plt
import pandas as pd
#%%
train_loss = pd.read_csv("plot/train_loss.csv", header=None)
train_f1 = pd.read_csv("plot/train_f1.csv", header=None)

train_f1 = train_f1.values.tolist()[0]
train_loss = train_loss.values.tolist()[0]
f1 = []
loss = []
for element in train_f1:
    f1.append(float(element))
for element in train_loss:
    loss.append(float(element))
    
#%%
plt.rcParams["figure.dpi"]=600
x = range(0,len(train_f1))

plot1 = plt.plot(x, f1)
plot1 = plt.plot(x, loss)
plt.legend(('f1','loss'), loc='best')
plt.title("Training Curve")
plt.xlabel('Batch')
plt.ylabel('Percentage')
plt.show()

#%% smooth the curve
from scipy.ndimage.filters import gaussian_filter1d

f1_smoothed = gaussian_filter1d(f1, sigma=1)
loss_smoothed = gaussian_filter1d(loss, sigma=1)
plt.plot(x, f1_smoothed)
plt.plot(x,loss_smoothed)
plt.legend(('f1','loss'), loc='best')
plt.title("Training Curve")
plt.xlabel('Batch')
plt.ylabel('Percentage')
plt.show()
