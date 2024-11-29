##Importing Packages###
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import csv
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy.stats as st
import seaborn as sns
import scipy.spatial as sp
from scipy.stats.stats import pearsonr
from scipy.spatial import distance
from mpl_toolkits import mplot3d
import matplotlib as mpl
from scipy.spatial.distance import cosine
import collections, numpy
from collections import defaultdict
from scipy import signal
#from scipy.interpolate import spline
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
import time
import random
from scipy import signal
from matplotlib.collections import LineCollection
from scipy import spatial
import pickle
import math


import copy
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from scipy.ndimage import gaussian_filter1d

import matplotlib.patches as patches

import matplotlib as mpl
import scipy as sp

import scipy.ndimage.filters as sci

from scipy.ndimage import label
from itertools import groupby
#from sklearn.preprocessing import Imputer



import os
os.environ['R_HOME'] = '/usr/lib64/R'


###Folders
Data_folder_P='/Taskspace_abstraction/Data/' ## if working in P
base_dropbox='C:/Users/moham/team_mouse Dropbox/Mohamady El-Gaby/'

#base_dropbox='D:/team_mouse Dropbox/Mohamady El-Gaby/'

Data_folder_dropbox=base_dropbox+'/Taskspace_abstraction/Data/' ##if working in C
Behaviour_output_folder = '/Taskspace_abstraction/Results/Behaviour/'
Ephys_output_folder = '/Taskspace_abstraction/Results/Ephys/'
Ephys_output_folder_dropbox = base_dropbox+'/Taskspace_abstraction/Results/Ephys/'
Intermediate_object_folder_dropbox = Data_folder_dropbox+'/Intermediate_objects/'

Intermediate_object_folder = Data_folder_dropbox+'/Intermediate_objects/'

base_ceph='Z:/mohamady_el-gaby/'
Data_folder_ceph='Z:/mohamady_el-gaby/Taskspace_abstraction_2/Data/'
Data_folder_ceph1='Z:/mohamady_el-gaby/Taskspace_abstraction/Data/'
Data_folder_ceph2='Z:/mohamady_el-gaby/Taskspace_abstraction_2/Data/'

Intermediate_object_folder_ceph = Data_folder_ceph1+'/Intermediate_objects/'

Data_folder=Intermediate_object_folder ###


Code_folder='/Taskspace_abstraction/Code/'

'Data is here: https://drive.google.com/drive/folders/1vJw8AVZmHQrUnvqkASUwAd4t549uKN6b '
#import rpy2
#import rpy2.robjects as robjects

#from rpy2.robjects.packages import importr
# import R's "base" package
#base = importr('base')

# import R's "utils" package
#utils = importr('utils')

# import rpy2's package module
#import rpy2.robjects.packages as rpackages

# import R's utility package
#utils = rpackages.importr('utils')

# select a mirror for R packages
#utils.chooseCRANmirror(ind=1) # select the first mirror in the list

#import rpy2.robjects.numpy2ri
#rpy2.robjects.numpy2ri.activate()

def pearson_nonan(x,y):
    xy=column_stack_clean(x,y)
    return(st.pearsonr(xy[:,0],xy[:,1]))

def test(x):
    y=x/2
    return(y)

def rec_dd():
    return defaultdict(rec_dd)

def crossings_epochs(x,y):
    len_all=[]
    for start, end in x:
        xx_start=np.where(y[:,0] > start)[0]
        xx_end=np.where((y[:,0] < end))[0]
        crossingx=np.intersect1d(xx_start,xx_end)
        lenxx=len(crossingx)
        len_all.append(lenxx)
    return(len_all)

def crossings_epochs_first(x,y):
    if len(y)>0:
        first_all=[]
        for start, end in x:
            xx_start=np.where(y[:,0] > start)[0]
            xx_end=np.where((y[:,0] < end))[0]
            crossingind=np.intersect1d(xx_start,xx_end)
            if len(crossingind)>0:
                crossingtime=y[crossingind[0],0]
            else:
                crossingtime=np.inf
            first_all.append(crossingtime)
    else:
        first_all=np.repeat(np.inf,len(x))
    return(first_all)

def crossings_epochs_last(x,y):
    if len(y)>0:
        last_all=[]
        for start, end in x:
            xx_start=np.where(y[:,0] > start)[0]
            xx_end=np.where((y[:,0] < end))[0]
            crossingind=np.intersect1d(xx_start,xx_end)
            if len(crossingind)>0:
                crossingtime=y[crossingind[-1],0]
            else:
                crossingtime=-np.inf
            last_all.append(crossingtime)
        #first_all=np.vstack(first_all)
    else:
        last_all=np.repeat(-np.inf,len(x))
    return(last_all)

def first_pulse(period,pulse):
    period[period==-0]=0
    pulse[pulse==-0]=0
    xxx=period[:,0]
    first_pulseX=np.zeros(len(xxx))
    for ii in range(len(xxx)):
        diffs=xxx[ii]-pulse[:,0]
        if len(np.where(diffs<0)[0])>0:
            first_pulseX[ii]=pulse[np.where(diffs<0)[0][0],0]
        else:
            first_pulseX[ii]=pulse[0,0]
        
    return(first_pulseX)

#def split_mode(xx,num_bins):
#    xxx=np.array_split(xx,num_bins)
#    return(np.asarray([st.mode(xxx[ii],nan_policy='omit')[0][0] for ii in range(len(xxx))]))

def split_mode(xx,num_bins):
    xxx=np.array_split(xx,num_bins)
    return(np.asarray([st.mode(xxx[ii],nan_policy='omit')[0]\
                       if np.isnan(np.nanmean(xxx[ii]))==False\
                       else st.mode(xxx[ii])[0]
                       for ii in range(len(xxx))]))

def last_val(x,y):
    val_last_all=[]
    for i in range(0,np.shape(x)[0]):
        xx=x[i]
        yy=y[i]

        xy=np.column_stack((xx,yy))
        vallast=[]
        for i,value in enumerate(xy):
            if (xy[i,1]==-np.inf or xy[i,1]==-np.inf) and (xy[i,0]==-np.inf or xy[i,0]==np.inf):
                vallast.append(np.nan)
            elif xy[i,0] > xy[i,1]:
                vallast.append(1)
            elif xy[i,0] < xy[i,1]:
                vallast.append(-1)
            

        val_last_all.append(vallast)
    return(val_last_all)

def first_val(x,y):
    val_last_all=[]
    for i in range(0,np.shape(x)[0]):
        xx=x[i]
        yy=y[i]

        xy=np.column_stack((xx,yy))
        vallast=[]
        for i,value in enumerate(xy):
            if (xy[i,1]==-np.inf or xy[i,1]==-np.inf) and (xy[i,0]==-np.inf or xy[i,0]==np.inf):
                vallast.append(np.nan)
            elif xy[i,0] > xy[i,1]:
                vallast.append(-1)
            elif xy[i,0] < xy[i,1]:
                vallast.append(1)
            

        val_last_all.append(vallast)
    return(val_last_all)

#Extracting led displays for sessions A and B
def extract_AB(recording_days,file):
    sesAB_dic={}
    for x in recording_days: 
        mousex = re.search('(\w+)-', x)
        mouse = mousex.group(1)
        dayx = re.search('-(\w+)', x)
        day = dayx.group(1)
        with open(file, 'r') as f:
            desen = [i.split(" ") for i in f.read().split()] 
        ses8 = desen[17]
        ses9 = desen[19]
        #exec(str(mouse)+'_'+str(day)+'_ses8_desen = ses8')
        #exec(str(mouse)+'_'+str(day)+'_ses9_desen = ses9')
        seq2=(str(mouse),"_",str(day))
        desen_name=(ses8,ses9)
        sesAB_dic[''.join (seq2)] = desen_name
    return(sesAB_dic)

def normalizeV(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return (v)
    else:
        return (v / norm)

def traj_angle(x,y):
    diff=x[0]-y[0]

    newy=np.add(y,diff)

    xx=np.zeros(len(x))
    for ii in range(len(x)):
        if ii ==0:
            xx[ii]=np.nan
        else:
            AA=np.subtract(normalizeV(x[ii]),normalizeV(x[0]))
            BB=np.subtract(normalizeV(y[ii]),normalizeV(y[0]))

            xx[ii]=angle_between(AA,BB)

    return(xx)

#def angle_to_distance(xx):
#    return(np.asarray([1-math.cos(math.radians(xx[ii])) for ii in range(len(xx))]))

def zscore_manual(xxx,means,stds):
    if len(means)>1:
        return([(xxx[ii]-means[ii])/stds[ii] for ii in range(len(means))])
    else:
        return([(xxx-means)/stds])

def orthogonal_vector(k):
    x = np.random.randn(len(popnvec))
    x -= x.dot(k) * k / np.linalg.norm(k)**2
    return(x)

###Detecting assemblies (returns assembly weight vectors)
def assembly_detect(Z):
    from sklearn.decomposition import PCA
    from sklearn.decomposition import FastICA
    n_components=np.shape(Z)[0]

    ##PCA
    transformer_PCA = PCA(n_components=n_components)
    assembliesX=(transformer_PCA.fit_transform(Z.T)) 
    eigenvalues=transformer_PCA.explained_variance_


    ##Identifying significant assembly number by setting threshold (Marčenko–Pastur)
    q=np.shape(Z)[1]/np.shape(Z)[0]
    lambdamax=(1+np.sqrt(1/q))**2

    sign_assemblies=assembliesX[np.where(eigenvalues>lambdamax)[0]].T
    sign_eigens=eigenvalues[np.where(eigenvalues>lambdamax)[0]].T

    nassemblies=np.sum(eigenvalues>lambdamax)

    ###ICA
    if nassemblies>0:
        transformer_ICA = FastICA(n_components=nassemblies)
        transformer_ICA.fit_transform(Z.T)
        assemblies=transformer_ICA.components_
    else:
        assemblies=np.nan
            
    return(assemblies)

#Tracking assembly strengths (without smoothing)
def track_assembly(weights,Zb):
    P=np.outer(weights,weights)
    np.fill_diagonal(P,0) ## note this changes P 'in place' - sets main diagonal to zero
    Proj=np.inner(P,Zb)
    Rb=np.inner(Zb,Proj)
    return(Rb)

##convert nested dict into array
def dict_to_array(d):
    dictlist=[]
    for key, value in d.items():
        dictlist.append(value)
    return(np.asarray(dictlist))

def flatten(d):    
    res = []  # Result list
    for key, val in d.items():
        res.extend(dict_to_array(val))
    return (np.asarray(res))

def whl_strength(whl, strength):
    xxx=np.repeat(strength,500)
    strengthbinned=binned_array(xxx,512)
    if abs(len(whl)-len(strengthbinned)) < 3:
        whl_strength_1=np.column_stack(((whl//1)[:len(strengthbinned)],strengthbinned))
        return(whl_strength_1)
    else:
        raise LookupError('whl and strength files dont match')

def whl_strengthX(whl, strength,factor):
    xxx=np.repeat(strength,factor)
    strengthbinned=binned_array(xxx,512)
    if abs(len(whl)-len(strengthbinned)) < 3:
        whl_strength_1=np.column_stack(((whl//1)[:len(strengthbinned)],strengthbinned))
        return(whl_strength_1)
    else:
        raise LookupError('whl and strength files dont match')
        
def whl_strengthX2(whlx,strength,factor):
    xxx=np.repeat(strength,factor)
    strengthbinned=binned_array(xxx,512)
    if (len(whlx)-len(strengthbinned)<0) and (abs(len(whlx)-len(strengthbinned)) < (factor//512)+1):
        whl_strength_1=np.column_stack(((whlx//1),strengthbinned[:len(whlx)]))
        return(whl_strength_1)
    elif (len(whlx)-len(strengthbinned)>0) and (abs(len(whlx)-len(strengthbinned)) < (factor//512)+1):
        whl_strength_1=np.column_stack(((whlx//1)[:len(strengthbinned)],strengthbinned[:len(whlx)]))
        return(whl_strength_1)
    #else:
    #    raise LookupError('whl and strength files dont match')


def whlstrength_clean(whl_strength_1):
    whl_strength = whl_strength_1[np.logical_and(whl_strength_1[:,0] > 0, whl_strength_1[:,1] > 0)] ##removing -1s
    return(whl_strength)

def whlstrength_dic_clean(whl_strength_1):
    whl_clean_dic=rec_dd()
    for assembly, whl in whl_strength_1.items():
        whl_clean_dic[assembly]=whl[np.logical_and(whl[:,0] > 0, whl[:,1] > 0)]
        ##removing -1s
        
    return(whl_clean_dic)

def runs_test(d, v, alpha = 0.05):
    ##see: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35d.htm
    ##and https://medium.com/@sddkal/runs-test-implementation-in-python-6236b0a2b433
    
    # Get positive and negative values
    mask = d > v
    # get runs mask
    p = mask == True
    n = mask == False
    xor = np.logical_xor(p[:-1], p[1:]) 
    # A run can be identified by positive 
    # to negative (or vice versa) changes
    num_runs = sum(xor) + 1 # Get number of runs (because number of transiions is one less than number of runs)

    n_p = sum(p) # Number of positives
    n_n = sum(n)
    # Temporary intermediate values
    tmp = 2 * n_p * n_n 
    tmps = n_p + n_n
    # Expected value
    r_hat = np.float64(tmp) / tmps + 1
    # Variance
    s_r_squared = (tmp*(tmp - tmps)) / (tmps*tmps*(tmps-1))
    # Standard deviation
    s_r =  np.sqrt(s_r_squared)
    # Test score
    z = (num_runs - r_hat) / s_r ##actual num runs minus expected divided by standard deviation

    # Get normal table 
    z_alpha = st.norm.ppf(1-alpha)
    # Check hypothesis
    return (z, z_alpha)



###defining overlap function
##REMEMBER TO PUT BIG PULSES FIRST!!
def overlap(a,b):
    overlap1=[]
    for start, end in a:
        for ent, exi in b:
            interval = []
            if ent > start and exi < end:
                interval= (ent, exi)
            if ent > start and ent < end and exi > end:
                interval= (ent, end)
            if ent < start and exi > start and exi < end:
                interval= (start, exi)
            if interval:     
                overlap1.append(interval)
    overlap = np.asarray(overlap1)
    return(overlap)

#overlap divided into subintervals
def overlap2(a,b):
    overlap1=[]
    for start, end in a:
        overlap2=[]
        for ent, exi in b:
            interval = []
            if ent > start and exi < end:
                interval= (ent, exi)
            if ent > start and ent < end and exi > end:
                interval= (ent, end)
            if ent < start and exi > start and exi < end:
                interval= (start, exi)
            if interval:     
                overlap2.append(interval)
        if len(overlap2)>0:
            overlap1.append(overlap2)
    overlap = np.asarray(overlap1)
    return(overlap)

def overlap3(a,b):
    overlap1=[]
    for start, end in a:
        overlap2=[]
        for ent, exi in b:
            interval = []
            if ent > start and exi < end:
                interval= (ent, exi)
            if ent > start and ent < end and exi > end:
                interval= (ent, end)
            if ent < start and exi > start and exi < end:
                interval= (start, exi)
            if interval:     
                overlap2.append(interval)
        #if len(overlap2)>0:
        overlap1.append(overlap2)
    overlap = np.asarray(overlap1)
    return(overlap)


##making interval file for periods outside interval file ax
def outside_intervals(ax):
    x=ax[:,1]
    xx=np.insert(x,0,0)
    y=ax[:,0]
    yy=np.append(y,999999999999)
    a=np.column_stack((xx,yy))
    return(a)


###defining non-overlap function
##REMEMBER TO PUT BIG PULSES FIRST!!
def non_overlap(ax,b):
    ##making interval file for periods outside interval file ax
    x=ax[:,1]
    xx=np.insert(x,0,0)
    y=ax[:,0]
    yy=np.append(y,np.inf)
    a=np.column_stack((xx,yy))
    
    overlap1=[]
    for start, end in a:
        for ent, exi in b:
            interval = []
            if ent > start and exi < end:
                interval= (ent, exi)
            if ent > start and ent < end and exi > end:
                interval= (ent, end)
            if ent < start and exi > start and exi < end:
                interval= (start, exi)
            if interval:     
                overlap1.append(interval)
    overlap = np.asarray(overlap1)
    return(overlap)

def rank_repeat(a):
    arr=np.zeros(len(a))
    for n in np.unique(a):
        count=0
        for ii in range(len(a)):
            if a[ii]==n:
                arr[ii]=count
                count+=1

    arr=arr.astype(int)
    return(arr)


#nonoverlap divided into subintervals
def non_overlap2(ax,b):
    ##making interval file for periods outside interval file ax
    x=ax[:,1]
    xx=np.insert(x,0,0)
    y=ax[:,0]
    yy=np.append(y,np.inf)
    a=np.column_stack((xx,yy))
    
    overlap1=[]
    for start, end in a:
        overlap2=[]
        for ent, exi in b:
            interval = []
            if ent > start and exi < end:
                interval= (ent, exi)
            if ent > start and ent < end and exi > end:
                interval= (ent, end)
            if ent < start and exi > start and exi < end:
                interval= (start, exi)
            if interval:     
                overlap2.append(interval)
        if len(overlap2)>0:
            overlap1.append(overlap2)
    overlap = np.asarray(overlap1)
    return(overlap)

def indicesX(z,thr):
    indicesALL=[]
    for j in range(0, len(z)):
        zz=z[j]
        indices=[]
        for i in range(0,len(zz)):
            zzz=zz[i]
            diffx=(zzz[1]-zzz[0])+1
            indx=np.linspace(zzz[0],zzz[1],diffx)
            indices.append(indx)
        
        indices_crossing=np.unique(np.concatenate(indices))
        if len(indices_crossing)>thr:
            indicesALL.append(indices_crossing)
    indicesALL=np.asarray(indicesALL)
    return(indicesALL)

def indicesXX(z,thr):
    indicesALL=[]
    for j in range(0, len(z)):
        zz=z[j]
        indices=[]
        for i in range(0,len(zz)):
            zzz=zz[i]
            diffx=(zzz[1]-zzz[0])+1
            indx=np.linspace(zzz[0],zzz[1],diffx)
            indices.append(indx)
        
        if len(indices)>0:
            indices_crossing=np.unique(np.concatenate(indices))
            if len(indices_crossing)>thr:
                indicesALL.append(indices_crossing)
        else:
            indicesALL.append(np.array([]))
    indicesALL=np.asarray(indicesALL)
    return(indicesALL)

def indicesX2(z,thr):
    indicesx=[]
    for xx in range(0,len(z)):
        diffx=(z[xx,1]-z[xx,0]+1)
        if diffx>thr:
            indicesx.append(np.linspace\
                            (z[xx,0],z[xx,1],diffx))
    indices=np.asarray(indicesx)
    return(indices)





#two sets of indices, finds which of first set overlaps with the second
def overlap_indices(indices,xxx):
    indices_new=[]
    for ii in np.arange(len(indices)):
        indicesX=indices[ii]
        indicesY=np.intersect1d(indicesX,xxx)
        if len(indicesY)>0:
            indices_new.append(indicesY)
    return(indices_new)

#two sets of indices, finds which of first set doesnt overlap with the second
def non_overlap_indices(indices,xxx):
    indices_new=[]
    for ii in np.arange(len(indices)):
        indicesX=indices[ii]
        indicesY=np.setxor1d(indicesX,xxx)
        indicesXY=np.intersect1d(indicesX,indicesY)
        indices_new.append(indicesXY)
    return(indices_new)


def mean_indices(x,indices):
    x_indices=np.zeros((len(indices)))
    for ii in range(0,len(indices)):
        indy=indices[ii].astype(int)
        x_indices[ii]=np.nanmean(x[indy,1])
    return(x_indices)

def mean_indices2(x,indices):
    x_indices=np.zeros((len(indices)))
    for ii in range(0,len(indices)):
        indy=indices[ii].astype(int)
        x_indices[ii]=np.nanmean(x[indy])
    return(x_indices)

def value_indices(x,indices):
    x_indices=[]
    for ii in range(0,len(indices)):
        indy=indices[ii].astype(int)
        x_indices.append(x[indy,1])
    return(x_indices)

def value_indices2(x,indices):
    x_indices=[]
    for ii in range(0,len(indices)):
        indy=indices[ii].astype(int)
        x_indices.append(x[indy])
    return(x_indices)

def days_to_mice(dicX,mice,mean=True):
    mice_scores=[]
    for mouse in mice:
        mouse_scores=[]
        for index,item in dicX.items(): 
            if mouse in index:
                mouse_scores.append(item)
        mice_scores.append(mouse_scores)
    mice_means=np.asarray([np.nanmean(np.concatenate(xx)) for xx in mice_scores])
    if mean==True:
        return(mice_means)
    else:
        return(mice)
    
def days_to_miceX(dicX,mice,mean=True):
    mice_scores=[]
    for mouse in mice:
        mouse_scores=[]
        for index,item in dicX.items(): 
            if mouse in index:
                mouse_scores.append(item)
        if len(mouse_scores)>0:
            mice_scores.append(mouse_scores)
    mice_means=np.asarray(mean_complex2(mice_scores))
    if mean==True:
        return(mice_means)
    else:
        return(mice_scores)
    
def days_to_miceX2(dicX,mice,mean=True):
    mice_scores=[]
    for mouse in mice:
        mouse_scores=[]
        for index,item in dicX.items(): 
            if mouse in index:
                mouse_scores.append(item)
        if len(mouse_scores)>0:
            mice_scores.append(mouse_scores)
    mice_means=mean_complex2(np.asarray([mean_complex2(concatenate_complex2(xx)) for xx in mice_scores]))
    if mean==True:
        return(mice_means)
    else:
        return(mice_scores)
    
def divide_sim(Sim,thr):
    discrim_all=[]
    nondiscrim_all=[]
    for i in Sim:
        #print(i)
        discrim=len(np.where(i<thr)[0])
        nondiscrim=len(np.where(i>=thr)[0])
        nondiscrim_all.append(nondiscrim)
        discrim_all.append(discrim)

    asstype_all=np.column_stack((nondiscrim_all,discrim_all))
    return(asstype_all)

def assembly_sim_divide(sim,asstype):
    assembly_sim=[]
    for i in range(0,len(sim)):
        simx=sim[i]
        if asstype == 'discrim':
            simxx=simx[np.where(simx<0.55)[0]]
        elif asstype == 'nondiscrim':
            simxx=simx[np.where(simx>0.55)[0]]
        assembly_sim.append(simxx)
    return(assembly_sim)

def div_discrim(x_,thr):
    x_nondiscrim=[]
    x_discrim=[]
    for j in np.where((x_[:,-1]) < thr)[0]:
        x_discrim.append(x_[j,:])
    
    return(x_discrim)

def div_nondiscrim(x_,thr):
    x_nondiscrim=[]
    x_discrim=[]
    for i in np.where((x_[:,-1]) > thr)[0]:
        x_nondiscrim.append(x_[i,:])

    return(x_nondiscrim)

def mean_listoflists(x):
    yy=[]
    for i in range(0,len(x)):
        xx=x[i]
        shape=np.shape(np.shape(np.array((xx))))[0]
        cc=np.isnan(xx)
        if shape == 0 and cc == np.bool_(True):
            y = np.nan
        else:
            y = np.nanmean(np.asarray(xx))
        yy.append(y)
    return(yy)

def mean_listofarrays(x):
    yyy=[]
    for i in range(0,len(x)):
        xx=x[i]
        for j in range(0,len(xx)):
            yy=xx[j]
            y=np.asarray(np.nanmean(yy))
            
            if y:
                yyy.append(y)
    output=np.concatenate(np.vstack(yyy))
    return(output)

def concatenate_complex_nan(x):
    x_all=[]
    for i in range(0,len(x)):
        xx=np.concatenate(x[i])
        x_all.append(xx)
    xx=np.concatenate(x_all)

    return(xx)

def concatenate_complex(x):
    x_all=[]
    for i in range(0,len(x)):
        xx=np.concatenate(x[i])
        x_all.append(xx)
    xx=np.concatenate(x_all)

    xx = xx[~np.isnan(xx)]
    return(xx)

def concatenate_complex2(xx):

    ALL_elements=[]
    for ii in np.arange(len(xx)):
        xxii=xx[ii]
        for jj in np.arange(len(xxii)):
            xxiijj=xxii[jj]
            ALL_elements.append(np.asarray(xxiijj))
            
    return(np.asarray(ALL_elements))

def rank_arraybyarray(a,b):
    index = np.argsort(b[:, 0])[np.argsort(np.argsort(a[:, 0]))]
    bx = b[index, ...]
    return(bx)

def mean_complex(x):
    x_all=[]
    for i in range(0,len(x)):
        for j in range(0,len(x[i])):
            x_mean=np.nanmean(x[i][j])
            x_all.append(x_mean)
    return(x_all)

def remove_nan(x):
    x=x[~np.isnan(x)]
    return(x)

def remove_nanX(x):
    xx=[]
    for ii in x:
        if len(np.shape(ii)) != 0:
            xx.append(ii)
        else:
            if np.isnan(ii) == False:
                xx.append(ii)
    xx=np.asarray(xx)
    return(xx)

def discrim_nondiscrim_divide(kk,sim,thr,option):
    discrim_indices=[]
    nondiscrim_indices=[]
    for indexA, iii in enumerate(sim):
        if iii>=thr:
            nondiscrim_indices.append(indexA)
        if iii<thr:
            discrim_indices.append(indexA)

    discrim=[]
    nondiscrim=[]
    for j in range(0, np.shape(kk)[0]):
        if j in discrim_indices:
            discrim.append(kk[j])
        elif j in nondiscrim_indices:
            nondiscrim.append(kk[j])
            
    if option=="discrim":
        return(discrim)
    elif option=="nondiscrim":
        return(nondiscrim)

    
def emergence_point(y,thr):
    N=len(y)
    x=np.linspace(1,N,N)
    dip_point=x[np.where(y<thr)[0]]
    
    emergence_pointx=x[np.where(y>thr)[0]]
    
    if len(emergence_pointx)>0 and len(dip_point)>0:
        stable_emergence_pointx=emergence_pointx[np.where(emergence_pointx>dip_point[-1])[0]]
        if len(stable_emergence_pointx)>0:
            emergence_point=stable_emergence_pointx[0]
        else:
            emergence_point=np.nan
    elif len(emergence_pointx)>0 and len(dip_point)==0:
        emergence_point=emergence_pointx[0]
    else:
        emergence_point=np.nan
    return(emergence_point) 

def pred_pos(whl,cluwhl_ALL):

    mapbinned_all=[]
    mapbinned_flat_all=[]
    for iii in range(0,len(cluwhl_ALL)):
        cluwhl=cluwhl_ALL[iii]
        if len(whl) == len(cluwhl)+1:
            whl_ratex=np.column_stack((whl[:-1], cluwhl))
        else:
            whl_ratex=np.column_stack((whl, cluwhl))
        whl_rate=whl_ratex[np.where((whl_ratex[:,0]>0) & (whl_ratex[:,1]>0) & (whl_ratex[:,2]!=np.nan))[0]]
        mapx=MAP(whl_rate)
        mapbinned=bin_MAP(mapx,11,'values')
        mapbinned_flat=np.concatenate(mapbinned)
        mapbinned_all.append(mapbinned)
        mapbinned_flat_all.append(mapbinned_flat)

        mapbinned_x=bin_MAP(mapx,11,'x_edge')
        #mapbinned_xflat=np.concatenate(mapbinned_x)

        mapbinned_y=bin_MAP(mapx,11,'y_edge')
        #mapbinned_yflat=np.concatenate(mapbinned_y)



    mapbinned_all=np.asarray(mapbinned_all)
    mapbinned_flat_all=np.asarray(mapbinned_flat_all)

    interval_x=np.mean(np.diff(mapbinned_x))
    interval_y=np.mean(np.diff(mapbinned_y))

    x_coords=mapbinned_x[:-1]
    y_coords=mapbinned_y[:-1]
    xx=np.repeat(x_coords,10)
    yy=np.tile(y_coords,10)
    xy_coords=np.column_stack((xx,yy))


    pred_pos_all=[]
    for time in range(0,np.shape(whl)[0]):
        currentFRV=cluwhl_ALL[:,time]    

        mapbinned_all_z=st.zscore(mapbinned_all,axis=0)

        FRV_corr=np.dot(currentFRV,mapbinned_flat_all)


        if False in np.isnan(FRV_corr):
            pred_pos=np.nanargmax(FRV_corr)
        else:
            pred_pos=np.nan

        pred_pos_all.append(pred_pos)
    pred_pos_coord=xy_coords[pred_pos_all]

    return(pred_pos_coord)

def pred_posX(whl,cluwhl_ALL,bins):
    
    binsqrt=int(np.sqrt(bins))
    
    mapbinned_all=[]
    mapbinned_flat_all=[]
    for iii in range(0,len(cluwhl_ALL)):
        cluwhl=cluwhl_ALL[iii]
        if len(whl) == len(cluwhl)+1:
            whl_ratex=np.column_stack((whl[:-1], cluwhl))
        else:
            whl_ratex=np.column_stack((whl, cluwhl))
        whl_rate=whl_ratex[np.where((whl_ratex[:,0]>0) & (whl_ratex[:,1]>0) & (whl_ratex[:,2]!=np.nan))[0]]
        mapx=MAP(whl_rate)
        mapbinned=bin_MAP(mapx,binsqrt,'values')
        mapbinned_flat=np.concatenate(mapbinned)
        mapbinned_all.append(mapbinned)
        mapbinned_flat_all.append(mapbinned_flat)

        mapbinned_x=bin_MAP(mapx,binsqrt,'x_edge')
        #mapbinned_xflat=np.concatenate(mapbinned_x)

        mapbinned_y=bin_MAP(mapx,binsqrt,'y_edge')
        #mapbinned_yflat=np.concatenate(mapbinned_y)



    mapbinned_all=np.asarray(mapbinned_all)
    mapbinned_flat_all=np.asarray(mapbinned_flat_all)

    interval_x=np.mean(np.diff(mapbinned_x))
    interval_y=np.mean(np.diff(mapbinned_y))

    x_coords=mapbinned_x[:-1]
    y_coords=mapbinned_y[:-1]
    xx=np.repeat(x_coords,binsqrt-1)
    yy=np.tile(y_coords,binsqrt-1)
    xy_coords=np.column_stack((xx,yy))


    pred_pos_all=[]
    for time in range(0,np.shape(whl)[0]):
        currentFRV=cluwhl_ALL[:,time]    

        mapbinned_all_z=st.zscore(mapbinned_all,axis=0)

        FRV_corr=np.dot(currentFRV,mapbinned_flat_all)


        if False in np.isnan(FRV_corr):
            pred_pos=np.nanargmax(FRV_corr)
        else:
            pred_pos=np.nan

        pred_pos_all.append(pred_pos)
    pred_pos_coord=xy_coords[pred_pos_all]

    return(pred_pos_coord)

def pos_error(whl,pred_pos_coord):
    whl_pred_pos_coord=np.column_stack((whl,pred_pos_coord))
    pred_actual_diff=np.sqrt(np.square(whl_pred_pos_coord[:,0]-whl_pred_pos_coord[:,2])+np.square\
                             (whl_pred_pos_coord[:,1]-whl_pred_pos_coord[:,3]))
    pred_actual_diff[whl_pred_pos_coord[:,0]<0]=np.nan
    return(pred_actual_diff)

def mean_pos_error(whl,pred_pos_coord):
    whl_pred_pos_coord=np.column_stack((whl,pred_pos_coord))
    pred_actual_diff=np.sqrt(np.square(whl_pred_pos_coord[:,0]-whl_pred_pos_coord[:,2])+np.square\
                             (whl_pred_pos_coord[:,1]-whl_pred_pos_coord[:,3]))
    pred_actual_diff[whl_pred_pos_coord[:,0]<0]=np.nan

    mean_diff=np.nanmean(pred_actual_diff)
    return(mean_diff)

def whl_interval(whl,interval):
    interval=interval/512
    whlinterval_all=[]
    for ii in range(0,len(interval)):
        whlinterval=np.linspace(interval[ii][0],interval[ii][1],interval[ii][1]/512)
        whlinterval_all.append(whlinterval)
    whlinterval_all=np.concatenate(whlinterval_all).astype(int)
    
    if len(whl) in whlinterval_all:
        whlinterval_all=np.delete(whlinterval_all,-1)
    whl_interval=whl[whlinterval_all]
    return(whl_interval)

def max_dist(whl):
    whlyy=whl[np.where(whl[:,0]>0)]

    lenx=np.max(whlyy[:,0])-np.min(whlyy[:,0])
    leny=np.max(whlyy[:,1])-np.min(whlyy[:,1])

    lenmax=np.sqrt(np.square(lenx)+np.square(leny))
    return(lenmax)

def distTOcrossing(dist,thr,factor1,factor2):
    
    indx=((np.where(dist<thr)[0]*factor1)//factor2).astype(int)
    if len(indx)>0:
        diffs=np.hstack((1000,np.diff(indx)))
        indxdiffs1=np.column_stack((indx,diffs))
        indxdiffs2=np.column_stack((indx[:-1],diffs[1:]))
        entry=indx[np.where(indxdiffs1[:,1]>10)[0]]
        exitx=indx[np.where(indxdiffs2[:,1]>10)[0]]
        exit=np.hstack((exitx,indxdiffs1[-1,0]))
        crossings=np.column_stack((entry,exit))
    else:
        crossings=[]
    return(crossings)


def shuffle_whl(whl,pred_pos_coord,number):
    import random
    pos_error_all=[]
    for i in range(0,number):
        pred_pos_coord_shuffled=pred_pos_coord.copy()
        random.shuffle(pred_pos_coord_shuffled)
        pos_error=mean_pos_error(whl,pred_pos_coord_shuffled)
        pos_error_all.append(pos_error)
    return(pos_error_all)

    
def mean_intervals(x,indx):
    means=np.zeros((len(indx)))
    for ii in range (0, len(indx)):
        z=indx[ii]
        ind=z.astype(int)
        #ind=ind[np.where(ind<len(x))[0]]
        mean=np.nanmean(x[ind])
        means[ii]=mean
    #means=np.asarray(means)
    return(means)

def sum_range(start,end):
    sumx=0
    for x in range(start,end):
        sumx= sumx +x
    return(sumx)

def binned_array(arr, n):
    end =  n * int(len(arr)/n)
    return (np.mean(arr[:end].reshape(-1, n), 1))

def binned_arrayX(arr, n,statistic='mean'):        
    m=int(len(arr.T)/n)
    if statistic=='mean':
        return(np.mean(arr.reshape(len(arr), n, m), 2))
    elif statistic=='sum':
        return(np.mean(arr.reshape(len(arr), n, m), 2))

def kth_diag_indices(a, k):
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return (rows[-k:], cols[:k])
    elif k > 0:
        return (rows[:-k], cols[k:])
    else:
        return (rows, cols)

def cross_corr_fast(x,y):
    data_length = len(x)

    b = x
    a = np.zeros(data_length * 2)

    a[data_length//2:data_length//2+data_length] = y # This works for data_length being even

    # Do an array flipped convolution, which is a correlation.
    c = signal.fftconvolve(b, a[::-1], mode='valid')
    return(c/100)

def crosscorrelate(t, u, bins):
    """
    Arguments:
        t (array): first array of "points" to correlate. The array needs
            to be monothonically increasing.
        u (array): second array of "points" to correlate. The array needs
            to be monothonically increasing.
        bins (array): bin edges for lags where correlation is computed.
        normalize (bool): if True, normalize the correlation function
            as typically done in FCS using :func:`pnormalize`. If False,
            return the unnormalized correlation function.
    Returns:
        Array containing the correlation of `t` and `u`.
        The size is `len(bins) - 1`.
    See also:
    """
    nbins = len(bins) - 1

    # Array of counts (histogram)
    counts = np.zeros(nbins, dtype=np.int64)

    # For each bins, imin is the index of first `u` >= of each left bin edge
    imin = np.zeros(nbins, dtype=np.int64)
    # For each bins, imax is the index of first `u` >= of each right bin edge
    imax = np.zeros(nbins, dtype=np.int64)

    # For each ti, perform binning of (u - ti) and accumulate counts in Y
    for ti in t:
        for k, (tau_min, tau_max) in enumerate(zip(bins[:-1], bins[1:])):
            #print ('\nbin %d' % k)

            if k == 0:
                j = imin[k]
                # We start by finding the index of the first `u` element
                # which is >= of the first bin edge `tau_min`
                while j < len(u):
                    if u[j] - ti >= tau_min:
                        break
                    j += 1

            imin[k] = j
            if imax[k] > j:
                j = imax[k]
            while j < len(u):
                if u[j] - ti >= tau_max:
                    break
                j += 1
            imax[k] = j
            # Now j is the index of the first `u` element >= of
            # the next bin left edge
        counts += imax - imin
    G = counts / np.diff(bins)
    return (G)

def Plot_timecourseX(yyy,end,bins,color):
    yy_all=[]
    for ii in np.arange(len(yyy)):
        yy=yyy[ii]
        group=end//bins
        yy_binned=yy[:end].reshape(-1,group).mean(axis=-1)
        yy_all.append(yy_binned)

    yerr=st.sem(yy_all,axis=0,nan_policy='omit')
    y=np.nanmean(yy_all,axis=0)
    x=np.linspace(1,len(y),len(y))

    plt.errorbar(x,y,yerr=yerr, color=color, marker='o')

def spike_diffs(ass1,ass2):
    ALL_diffs=[]
    for xx in np.arange(len(ass1)):
        xxx=ass1[xx]
        for yy in np.arange(len(ass2)):
            yyy=ass2[yy]
            diffs=np.zeros(len(xxx))
            for ii in np.arange(len(xxx)):
                diff=abs(yyy-xxx[ii])
                nearest_point=np.where(diff==np.min(diff))[0]
                if len(nearest_point)==1:
                    diffs[ii]=(yyy[nearest_point]-xxx[ii])
            diffs=diffs[abs(diffs)<20000*(50/1000)]
            ALL_diffs.append(diffs)
    return(ALL_diffs)

def linear_model(xxx,yyy,nameX,nameY):
    ###GENERALIZE
    allvaluesX=np.vstack((xxx,yyy))
    allvalues=(np.concatenate(allvaluesX)).astype(float)
    noDV2=np.shape(allvaluesX)[1]
    DV1=np.hstack((np.repeat(nameX,noDV2*len(yyy)), np.repeat(nameY,noDV2*len(xxx))))
    DV2=(np.repeat(np.arange(noDV2),len(allvaluesX))).astype(str)
    
    list_of_tuplesX = np.column_stack((Asstype,percentile, allvalues))
    list_of_tuples = np.column_stack((Asstype,percentile, allvalues))#[np.isnan(allvalues)==False]

    df=pd.DataFrame(data=list_of_tuples, columns=['DV1','DV2','value']) 
    df['value']=df['value'].astype('float') ##key step to make value a numerical variable!!
    
    model = ols('value ~ C(DV1)*C(DV2)', df).fit()
    
    return(model,df)

def ANOVA_R(A1,A2,B1,B2):
    ##Generalize this!!
    robjects.globalenv['A1']=A1 
    robjects.globalenv['A2']=A2 
    robjects.globalenv['B1']=B1 
    robjects.globalenv['B2']=B2
    
    #returns 'Intercept','Factor1(non repeated measure e.g. detection context)',
    ##'Factor2(repeated measure e.g. tracking context)','Factor1:Factor2 interaction'
    
    robjects.r('''
        Factor1 <- c('A','A','B','B')
        Factor2 <- c('Same','Other','Same','Other')
        idata <- data.frame(Factor1, Factor2)
        Bind <- cbind(A1,A2,B1,B2)
        model <- lm(Bind~1)
        library(car)
        analysis <- Anova(model, idata=idata, idesign=~Factor1 + Factor2 + Factor1*Factor2, type="III")
        anova_sum = summary(analysis)
        #anova_sum
        ''')

    
    return(np.concatenate(np.vstack((robjects.globalenv['anova_sum'][3][20],\
                                    robjects.globalenv['anova_sum'][3][21],\
                                    robjects.globalenv['anova_sum'][3][22],\
                                    robjects.globalenv['anova_sum'][3][23]))),\
           ('Intercept','non-repeated-measure','repeated-measure','interaction'))

def TukeyHSD(Data,alpha):
    Datax=[]
    for ii in range(np.shape(Data)[1]):
        xxx=Data[:,ii]
        yyy=np.repeat(ii,len(xxx))
        Datax.append(np.column_stack((xxx,yyy)))
    Data_reorg=np.vstack(Datax)
    mc = MultiComparison(Data_reorg[:,0], Data_reorg[:,1])
    mc_results = mc.tukeyhsd(alpha=alpha)
    return(print(mc_results))

def rearrange_for_ANOVA(xxx):
    x=np.repeat(np.arange(len(xxx)),np.shape(xxx)[1])
    y=(np.tile(np.arange(np.shape(xxx)[1]),len(xxx))).astype(int)
    z=np.concatenate(xxx)
    
    return(np.column_stack(((x).astype(int),(y).astype(int),z)))

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
from scipy import stats

def rearrange_for_ANOVAX(dicX):
    all_xxx=[]
    for label,array in dicX.items():
        xxx=array
        indx_all=[]
        for ii in np.arange(len(xxx)):
            indx=np.repeat(ii,len(xxx[ii]))
            indx_all.append(indx)

        all_xxx.extend(np.column_stack((np.repeat(label,len(np.concatenate(xxx))),\
                                        np.concatenate(indx_all),np.concatenate(xxx))))

    return(np.asarray(all_xxx))

def timestamp_to_binary(timestampsx,lenses):
    yyy=np.zeros((lenses))
    timestamps=(timestampsx).astype(int)
    np.put(yyy,timestamps,1)
    return(yyy)

def fill_nans(xxx,length):
    return(np.vstack((xxx,[np.repeat(np.nan,len(xxx.T)) for ii in range(length)])))
def fill_nansX(xxx,length):
    return(np.vstack([np.hstack((xxx[ii],np.repeat(np.nan,length-len(xxx[ii]))))[:length] if len(xxx[ii])<length\
                      else xxx[ii][:length] for ii in range(len(xxx)) ]))

def angle_to_state(angle):
    if angle>0 and angle<90:
        state='A'
    if angle>90 and angle<180:
        state='B'
    if angle>180 and angle<270:
        state='C'
    if angle>270 and angle<360:
        state='D'
    return(state)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return (vector / np.linalg.norm(vector))

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return (np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def matrix_triangle(a,direction='upper',return_indices=False):
    if direction=='upper':
        indices=np.triu_indices(len(a), k = 1)
    if direction=='lower':
        indices=np.tril_indices(len(a), k = -1)
    triangle=a[indices]
    if return_indices==True:
        return(triangle,indices)
    else:
        return(triangle)

######################################
########################################
########PLOTTING#######################
######################################
######################################



###function to plot scatter plots (e.g. comparing assembly strength at correct vs incorrect dispensers)
def plot_scatter(x,y,name='none'):
    plt.plot(x, y, 'o')
    z= [-10000, 0, 10000]
    plt.plot(z,z,'k--')

    xy=np.hstack((x,y))

    xmin=min(xy)-np.mean(xy)*0.1
    xmax=max(xy)+np.mean(xy)*0.1
    ymin=min(xy)-np.mean(xy)*0.1
    ymax=max(xy)+np.mean(xy)*0.1
    


    #plt.xlim(-0.2,0.2)
    #plt.ylim(-0.2,0.2)

    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    
    plt.gca().set_aspect('equal', adjustable='box')
    
    if name != 'none':
        plt.savefig(name)
    plt.show()

###function to plot scatter plots (e.g. comparing assembly strength at correct vs incorrect dispensers)
def noplot_scatter(x,y, color):
    plt.plot(x, y, 'o', color=color, alpha=0.7,markersize=7)
    z= [-10000, 0, 10000]
    plt.plot(z,z,'k--')

    xy=np.hstack((x,y))
    
    global xmin
    global xmax
    global ymin
    global ymax
    
    xmin=min(xy)-np.mean(xy)*0.1
    xmax=max(xy)+np.mean(xy)*0.1    
    ymin=min(xy)-np.mean(xy)*0.1
    ymax=max(xy)+np.mean(xy)*0.1

    


    #plt.xlim(-0.2,0.2)
    #plt.ylim(-0.2,0.2)

    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.gca().set_aspect('equal', adjustable='box')
    
def rand_jitter(arr):
    stdev = .05*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def rand_jitterX(arr, X):
    stdev = X*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def bar_plot(x,y,name,pairing,ymin,ymax):
    plt.figure(figsize=(3,3))
    
    ##bars
    x_mean=np.nanmean(x)
    y_mean=np.nanmean(y)
    x_sem=st.sem(x, nan_policy='omit')
    y_sem=st.sem(y, nan_policy='omit')
    xxx=[0.35,0.65]
    data= [x, y]    
    means= [x_mean, y_mean]
    error= [x_sem, y_sem]  
    xlocations = np.array(range(len(data)))
    width=0.2
    plt.bar(xxx, means, width, yerr=error, alpha=1, color=['black','red','blue'],\
           error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2), align='center')
    #plt.margins(0.05)
    plt.ylim(ymin-(0.05*(ymax-ymin)),ymax+(0.05*(ymax-ymin)))
    plt.xlim(0,1)
    
    ##points and lines
    xx=np.column_stack((x,np.repeat(0.35,len(x))))
    yy=np.column_stack((y,np.repeat(0.65,len(y))))
    xy=np.vstack((xx,yy))
    jittered=rand_jitter(xy[:,1])
    
    if pairing == 'paired': 
        x1=np.split(jittered,2)[0]
        x2=np.split(jittered,2)[1]
        xxxx=np.column_stack((x1,x2))
        yyy=np.column_stack((x,y))
        for i in range(0,len(yy)):
            yyyy=yyy[i]
            plt.plot(xxxx[i],yyyy, color='gray')
    plt.plot(jittered,xy[:,0],'o',markersize=7,color='white',markeredgecolor='black')

    if name != 'none':
        plt.savefig(name)

    plt.show()

###N.B. this needs rand_jitter!!




##Specific functions

def bar_noplotX(y,name,ymin,ymax,points,pairing,jitt):
    leny=len(y)
    plt.figure(figsize=(leny*(3/2),leny*(3/2)))
    
    if ymin =='auto':
        ymin=np.min(np.concatenate(y))
    if ymax =='auto':
        ymax=np.max(np.concatenate(y))
    
    ##bars
    y_mean=((np.zeros(len(y))))
    y_sem=((np.zeros(len(y))))
    for ii in range(0, len(y)):
        ymeanx=np.nanmean(y[ii])
        y_mean[ii]=ymeanx
        ysemx=st.sem(y[ii], nan_policy='omit')
        y_sem[ii]=ysemx
   
    
    xxx=np.linspace(0.15, 0.2+(0.2*(leny-1)), leny)

    xlocations = np.array(range(len(xxx)))
    width=0.2
    plt.bar(xxx, y_mean, width, yerr=y_sem, alpha=1, 
           error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2), align='center')
    
    if points != 'points':
        ymin=np.min(y_mean-y_sem) #-np.max(y_sem)
        ymax=np.max(y_mean+y_sem) #+np.max(y_sem)
    
    if ymin>0:
        ymin=0
    plt.ylim(ymin-(0.05*(ymax-ymin)),ymax+(0.05*(ymax-ymin)))
    plt.xlim(0,np.max(xxx)+0.15)
    
    

    ###points and lines
    if points == 'points':
        yyALL=[]
        for ii in range(0, len(y)):
            yy=np.column_stack((y[ii],np.repeat(xxx[ii],len(y[ii]))))
            yyALL.append(yy)

        xy=np.vstack((yyALL))
        jittered=rand_jitterX(xy[:,1],jitt)

        if pairing == 'paired':
            for ii in range(0, leny):
                x1=np.split(jittered,len(y))[ii]
                if ii == 0:
                    x1_all=x1
                else:
                    x1_all=np.column_stack((x1_all,x1))

            for jj in range(0,np.shape(y)[1]):
                yyyy=np.asarray(y)[:,jj]
                plt.plot(x1_all[jj],yyyy, color='gray')
        plt.plot(jittered,xy[:,0],'o',markersize=7,color='white',markeredgecolor='black')
    
    if name != 'none':
        plt.savefig(name)

def bar_plotX(y,name,ymin,ymax,points,pairing,jitt):
    leny=len(y)
    plt.figure(figsize=(leny*(3/2),6))
    
    if ymin =='auto':
        ymin=np.min(np.concatenate(y))
    if ymax =='auto':
        ymax=np.max(np.concatenate(y))
    
    ##bars
    y_mean=((np.zeros(len(y))))
    y_sem=((np.zeros(len(y))))
    for ii in range(0, len(y)):
        ymeanx=np.nanmean(y[ii])
        y_mean[ii]=ymeanx
        ysemx=st.sem(y[ii], nan_policy='omit')
        y_sem[ii]=ysemx
   
    
    xxx=np.linspace(0.15, 0.2+(0.2*(leny-1)), leny)

    xlocations = np.array(range(len(xxx)))
    width=0.2
    plt.bar(xxx, y_mean, width, yerr=y_sem, alpha=1, 
           error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2), align='center')
    
    if points != 'points' and ymin == 'auto':
        ymin=np.min(y_mean-y_sem) #-np.max(y_sem)
        ymax=np.max(y_mean+y_sem) #+np.max(y_sem)
    
    #if ymin>0:
    #    ymin=0
    plt.ylim(ymin-(0.05*(ymax-ymin)),ymax+(0.05*(ymax-ymin)))
    plt.xlim(0,np.max(xxx)+0.15)
    
    

    ###points and lines
    if points == 'points':
        yyALL=[]
        for ii in range(0, len(y)):
            yy=np.column_stack((y[ii],np.repeat(xxx[ii],len(y[ii]))))
            yyALL.append(yy)

        xy=np.vstack((yyALL))
        jittered=rand_jitterX(xy[:,1],jitt)

        if pairing == 'paired':
            for ii in range(0, leny):
                x1=np.split(jittered,len(y))[ii]
                if ii == 0:
                    x1_all=x1
                else:
                    x1_all=np.column_stack((x1_all,x1))

            for jj in range(0,np.shape(y)[1]):
                yyyy=np.asarray(y)[:,jj]
                plt.plot(x1_all[jj],yyyy, color='gray')
        plt.plot(jittered,xy[:,0],'o',markersize=7,color='white',markeredgecolor='black')
    
    if name != 'none':
        plt.savefig(name)
    
    #plt.show()
    
def cumulativeDist_plot(x,y,colorx,colory,name):
    fig, ax = plt.subplots(figsize=(3,3))
    hfont = {'fontname':'Arial'}
    x = x[x>-1E38]
    x = x[x<1E38]
    values, base = np.histogram(x, bins=40)
    cumulative = np.cumsum(values)
    plt.plot(base[:-1], cumulative, c=colorx)

    y = y[y>-1E38]
    y = y[y<1E38]
    values, base = np.histogram(y, bins=40)
    cumulative = np.cumsum(values)
    plt.plot(base[:-1], cumulative, c=colory)
    
    if name != 'none':
        plt.savefig(name)
    plt.show()
    
def cumulativeDist_plot_norm(x,y,colorx,colory,binsize,name,xmin,xmax):
    fig, ax = plt.subplots(figsize=(3,3))
    hfont = {'fontname':'Arial'}
    x = x[x>-1E38]
    x = x[x<1E38]
    
    y = y[y>-1E38]
    y = y[y<1E38]
    xy=np.hstack((x,y))
    bins=np.arange(np.min(xy)-binsize,np.max(xy)+binsize,binsize)
    
    values, base = np.histogram(x, bins=bins)
    cumulative = np.cumsum(values)/len(x)
    plt.plot(base[:-1], cumulative, c=colorx)

   
    
    #bins=np.arange(np.min(y)-binsize,np.max(y)+binsize,binsize)
    values, base = np.histogram(y, bins=bins)
    cumulative = np.cumsum(values)/len(y)
    plt.plot(base[:-1], cumulative, c=colory)
    
    plt.xlim(xmin,xmax)
    if name != 'none':
        plt.savefig(name)
    plt.show()
    
#def noplot_timecourseA(x,y,colorA,colorB):
#    y1=np.nanmean(x, axis=0)
#    y1err=st.sem(x, axis=0)
#    y2=np.nanmean(y, axis=0)
#    y2err=st.sem(y, axis=0)

#    end1=len(y1)
#    x1=np.linspace(0,1,end1)

#    end2=len(y2)
#    x2=np.linspace(0,1,end2)
#    #plt.figure(figsize=(15,5))
#    plt.errorbar(x1,y1, yerr=y1err, color=colorA, marker='o')
#    plt.errorbar(x2,y2, yerr=y2err, color=colorB, marker='o')


def plot_timecourseA(x,y,colorA,colorB):
    y1=np.nanmean(x, axis=0)
    y1err=st.sem(x, axis=0)
    y2=np.nanmean(y, axis=0)
    y2err=st.sem(y, axis=0)



    end1=len(y1)
    x1=np.linspace(0,1,end1)

    end2=len(y2)
    x2=np.linspace(0,1,end2)
    plt.figure(figsize=(15,5))
    plt.errorbar(x1,y1, yerr=y1err, color=colorA, marker='o')
    plt.errorbar(x2,y2, yerr=y2err, color=colorB, marker='o')
    #plt.xlim(-0.02,1)
    #plt.ylim(-0.7,2.8)
    plt.show()

def plot_timecourseA_save(x,y,colorA,colorB,name):
    y1=np.nanmean(x, axis=0)
    y1err=st.sem(x, axis=0)
    y2=np.nanmean(y, axis=0)
    y2err=st.sem(y, axis=0)



    end1=len(y1)
    x1=np.linspace(0,1,end1)

    end2=len(y2)
    x2=np.linspace(0,1,end2)
    plt.figure(figsize=(15,5))
    plt.errorbar(x1,y1, yerr=y1err, color=colorA, marker='o')
    plt.errorbar(x2,y2, yerr=y2err, color=colorB, marker='o')
    #plt.xlim(-0.02,1)
    #plt.ylim(-0.7,2.8)
    if name != 'none':
        plt.savefig(name)
    plt.show()

def plot_timecourseA_binned(x,y,bins,colorA,colorB):
    
    y1=np.nanmean(x, axis=0)
    y1err=st.sem(x, axis=0)
    y2=np.nanmean(y, axis=0)
    y2err=st.sem(y, axis=0)
    ##Binned
    width1=len(y1)//bins
    y1bin=y1[:(y1.size // width1) * width1].reshape(-1, width1).mean(axis=1)
    width2=len(y2)//bins
    y2bin=y2[:(y2.size // width2) * width2].reshape(-1, width2).mean(axis=1)

    end1=len(y1bin)
    x1bin=np.linspace(0,1,end1)

    end2=len(y2bin)
    x2bin=np.linspace(0,1,end2)    
    
    
    plt.figure(figsize=(15,5))
    plt.errorbar(x1bin,y1bin, yerr=y1err, color=colorA, marker='o')
    plt.errorbar(x2bin,y2bin, yerr=y2err, color=colorB, marker='o')
    #plt.xlim(-0.02,1)
    #plt.ylim(-0.1,0.225)
    plt.show()
    
def plot_timecourseA_binned_save(x,y,bins,colorA,colorB,name):
    
    y1=np.nanmean(x, axis=0)
    y1err=st.sem(x, axis=0)
    y2=np.nanmean(y, axis=0)
    y2err=st.sem(y, axis=0)
    ##Binned
    width1=len(y1)//bins
    y1bin=y1[:(y1.size // width1) * width1].reshape(-1, width1).mean(axis=1)
    width2=len(y2)//bins
    y2bin=y2[:(y2.size // width2) * width2].reshape(-1, width2).mean(axis=1)

    end1=len(y1bin)
    x1bin=np.linspace(0,1,end1)

    end2=len(y2bin)
    x2bin=np.linspace(0,1,end2)
    
    plt.figure(figsize=(15,5))
    plt.errorbar(x1bin,y1bin, yerr=y1err, color=colorA, marker='o')
    plt.errorbar(x2bin,y2bin, yerr=y2err, color=colorB, marker='o')
    #plt.xlim(-0.02,1)
    #plt.ylim(-0.1,0.225)
    
    if name != 'none':
        plt.savefig(name)
    plt.show()

def noplot_timecourseBx(x,y,color):
    ymean=mean_complex2(y)
    yerr=[st.sem(i,nan_policy='omit') for i in y]
    plt.errorbar(x,y=ymean, color=color, marker='o')
    plt.fill_between(x, ymean-yerr, ymean+yerr, color=color,alpha=0.5)

def binning(xx,width):
    x=np.nanmean(xx,axis=0)
    xbin=x[:(x.size // width) * width].reshape(-1, width).mean(axis=1)
    return(xbin)

def binning2(xx,bins):
    x=np.nanmean(xx,axis=0)
    width=len(y1)//bins
    xbin=x[:(x.size // width) * width].reshape(-1, width).mean(axis=1)
    return(xbin)

def concatenate_complex(x):
    x_all=[]
    for i in range(0,len(x)):
        xx=np.concatenate(x[i])
        x_all.append(xx)
    xx=np.concatenate(x_all)

    xx = xx[~np.isnan(xx)]
    return(xx)

def mean_complex(x):
    x_all=[]
    for i in range(0,len(x)):
        xx=np.nanmean(x[i],axis=1)
        x_all.append(xx)
    xx=np.concatenate(x_all)

    xx = xx[~np.isnan(xx)]
    return(xx)

def mean_complex2(x):
    means=[]
    for i in x:
        meanx=np.nanmean(i)
        means.append(meanx)
    means=np.asarray(means)
    return(means)

def mean_complex3(x):
    means=[]
    for i in x:
        if np.isnan(np.mean(i))==False:
            meanx=np.nanmean(i,axis=0)
            means.append(meanx)
        else:
            means.append((np.repeat(np.nan,len(i.T))))

    means=np.asarray(means)
    return(means)

def median_complex2(x):
    medians=[]
    for i in x:
        medianx=np.nanmedian(i)
        medians.append(medianx)
    medians=np.asarray(medians)
    return(medians)

def mean_complex_nan(x):
    x_all=[]
    for i in range(0,len(x)):
        xx=np.nanmean(x[i],axis=1)
        x_all.append(xx)
    xx=np.concatenate(x_all)

    return(xx)

def flip_diff(xxx):
    yyy=np.zeros((len(xxx),10))
    for ii in np.arange(len(xxx)):
        yyy[ii]=np.hstack((xxx[ii][-1],np.diff(np.flipud(xxx[ii]))))
    return(yyy)

def equal_timestamp(xxx):
    xxx_diff=(xxx[:,1]-xxx[:,0])
    xxx_new=np.delete(xxx,np.where(xxx_diff<(np.max(xxx_diff)-10))[0],axis=0)
    
    return(xxx_new)

def rearrange_matrix(x,indices):
    xx=x[indices]
    xxx=xx[:,indices]
    return(xxx)

def remove_outlier_array(xxx,nstd):
    xxx[xxx==np.inf]=np.nan
    xxx[xxx==-np.inf]=np.nan
    yyy=np.zeros((np.shape(xxx)[1],np.shape(xxx)[0]))
    for ii in np.arange(np.shape(xxx)[1]):
        x=xxx[:,ii]
        thr=np.nanmean(x)+nstd*np.nanstd(x)
        x[abs(x)>thr]=np.nan
        yyy[ii]=x
    yyy=yyy.T
    
    return(yyy)

def remove_outlier_arrayX(xxx,nstd):
    xxx[xxx==np.inf]=np.nan
    xxx[xxx==-np.inf]=np.nan
    thr=np.nanmean(xxx)+nstd*np.nanstd(xxx)
    xxx[abs(xxx)>thr]=np.nan
    return(xxx)

####Correlations using spike timing tiling coefficient (see Cutts and Eglen 2014)
def STTC(A,B,END,dt):
    A=np.asarray(A)
    B=np.asarray(B)
    
    Adiff=np.diff(A)
    Bdiff=np.diff(B)

    AAdiff=np.column_stack((A[:-1],A[1:],Adiff))
    BBdiff=np.column_stack((B[:-1],B[1:],Bdiff))

    TAx=(len(np.where(AAdiff[:,2]>dt)[0])+1)*dt*2
    if A[0] < dt:
        TAx=TAx-(dt-A[0])
    if END<(A[-1]+dt):
        TAx=TAx-((AAdiff[:,:2][-1,-1]+dt)-END)

    TBx=(len(np.where(BBdiff[:,2]>dt)[0])+1)*dt*2
    if B[0] < dt:
        TBx=TBx-(dt-B[0])
    if END<(B[-1]+dt):
        TBx=TBx-((BBdiff[:,:2][-1,-1]+dt)-END)


    TA=TAx/END
    TB=TBx/END

    lenA=len(A)
    lenB=len(B)
    
    numcorrA=np.zeros((lenA))
    for ii in range(0,lenA):
        ABdiff=B-A[ii]
        numcorr=len(np.where(np.logical_and(-dt<ABdiff,ABdiff<dt))[0])
        numcorrA[ii]=numcorr
    
    numcorrB=np.zeros((lenB))
    for ii in range(0,lenB):
        BAdiff=A-B[ii]
        numcorr=len(np.where(np.logical_and(-dt<BAdiff,BAdiff<dt))[0])
        numcorrB[ii]=numcorr


    PA=len(np.where(numcorrA>0)[0])/len(A)
    PB=len(np.where(numcorrB>0)[0])/len(B)

    STTCx=0.5*(((PA-TB)/(1-(PA*TB)))+((PB-TA)/(1-(PB*TA))))
    return(STTCx)

def assembly_spiketimes(spiketimes,strength,ntsd):
    thr=np.mean(strength)+(nstd*np.std(strength))
    indicesy=np.concatenate(np.asarray([np.where(strength>thr)[0]])*500)
    indicesx=indicesy-500
    indicesxy=np.column_stack((indicesx,indicesy))
    indices=np.concatenate(indicesX2(indicesxy,0))

    return(np.intersect1d(spiketimes,indices))

def assembly_spiketimes_members(spiketimesX,strength,nstd):
    thr=np.mean(strength)+(nstd*np.std(strength))
    indicesy=np.concatenate(np.asarray([np.where(strength>thr)[0]])*500)
    indicesx=indicesy-500
    indicesxy=np.column_stack((indicesx,indicesy))
    indices=np.concatenate(indicesX2(indicesxy,0))
    
    spike_activations=[]
    for ii in spiketimesX:
        spike_activations.append(np.intersect1d(ii,indices))
    return(np.asarray(spike_activations))



##allows stacking two columns and removes rows containing nans/infs/-infs
def column_stack_clean(x,y):
    xy=np.column_stack((x,y))
    xy=xy[~np.isnan(xy).any(axis=1)]
    xy=xy[~np.isinf(xy).any(axis=1)]
    x=xy[:,0]
    y=xy[:,1]
    xy_new=np.column_stack((x,y))
    return(xy_new)

def reorder_vector(xx,indices):
    no_neur=len(xx)
    ind_all=(np.linspace(0,no_neur-1,no_neur)).astype(int)
    ind_remain=np.setdiff1d(ind_all,indices)
    xx_reordered=np.concatenate((xx[indices],xx[ind_remain]))
    return(xx_reordered)

def reorder_matrix(xx,indices):
    no_neur=len(xx)
    ind_all=(np.linspace(0,no_neur-1,no_neur)).astype(int)
    ind_remain=np.setdiff1d(ind_all,indices)
    xx_reordered=np.vstack((xx[indices,:],xx[ind_remain,:]))
    return(xx_reordered)

def remove_excess(x,num):
    x_pruned=[]
    for i in x:
        xx=np.asarray(i)
        if len(xx) == (num-1):
            xx=np.append(xx,np.nan)
        if len(xx) == (num-2):
            xx=np.append(xx,(np.nan,np.nan))
        #xx=np.concatenate(xx)
        x_pruned.append(xx[:num])
    x_pruned=np.array(x_pruned)
    return(x_pruned)

def remove_excess2(x,num):
    x_pruned=[]
    for i in x:
        xx=np.asarray(i)
        if len(xx) == (num-1):
            xx=np.append(xx,np.nan)
        if len(xx) == (num-2):
            xx=np.append(xx,(np.nan,np.nan))
        #xx=np.concatenate(xx)
        x_pruned.append(xx[:num])
    #x_pruned=np.array(x_pruned)
    return(x_pruned)

def plot_corr_matrix(x,name,length,width):
    corrx=np.corrcoef(x)

    np.fill_diagonal(corrx,0)

    fig, ax = plt.subplots(figsize=(length,width))
    cax = ax.matshow(corrx, cmap='bwr', vmin=-0.2, vmax=0.2)

    cbar = fig.colorbar(cax, ticks=[-0.2, 0, 0.2], orientation='vertical')
    cbar.ax.set_xticklabels(['-0.2', '0', '0.2'])  # horizontal colorbar
    
    if name != 'none':
        plt.savefig(name)
    #plt.show()
    
def PSTH(resi,pulses,bin_width,pre,post,figname):
    x=np.linspace(1,bin_width,bin_width)

    start = pulses[:,0].astype(int)
    end = pulses[:,1].astype(int)
    counts = [0] * bin_width

    for i in range(start.size - 1):
        tmpCounts, tmpEdges = np.histogram(resi, bins = bin_width, range = (start[i]-pre, end[i]+post))
        counts = counts + tmpCounts

    fig,ax = plt.subplots(1)

    fig=plt.bar(x,counts,color='black',width=1)

    length=np.mean(end-start)+pre+post

    plotstart=(pre*bin_width)//length
    plotend=((length-post)*bin_width)//length

    plt.axvline(x=plotstart, color='g', linestyle='solid', linewidth=1)
    plt.axvline(x=plotend, color='g', linestyle='dashed', linewidth=1)    

    xy=(plotstart,0)
    width=plotend-plotstart
    height=np.max(counts)+10
    rect=patches.Rectangle(xy, width, height, color='lightgreen', alpha=0.3)

    ax.add_patch(rect)
    
    if figname != 'none':
        plt.savefig(figname)
    plt.show()
    
def select_members(x,thrx,direction):
    mean=np.nanmean(x)
    thr=thrx*np.std(x)
    if direction == 'high':
        member_indx=np.where(x > mean+thr)[0]
    elif direction == 'low':
        member_indx=np.where(x < mean+thr)[0]
    return(member_indx)

def reorder_xy(x,y):
    xy=np.column_stack((x,y[x]))
    xy_sortedx=xy[(-xy[:,1]).argsort()]
    xy_sorted=xy_sortedx[:,0].astype(int)
    return(xy_sorted)

def makethetaindices(indx):
    ind=[]
    for i, val in enumerate(indx):
        z=np.linspace(val-3,val+3,7)
        ind.extend(z)
    ind=np.asarray(ind)
    ind=ind.astype(int)
    return(ind)

def makethetaindices2(startx,endx):
    indx=np.column_stack((startx,endx))
    ind=[]
    for start, end in indx:
        diff=end-start
        z=np.linspace(start,end,diff)
        ind.extend(z)
    ind=np.asarray(ind)
    ind=ind.astype(int)
    return(ind)

def binning_individual(y1,bins):
    width1=len(y1)//bins
    y1bin=y1[:(y1.size // width1) * width1].reshape(-1, width1).mean(axis=1)
    return(y1bin)

def stack_min(xxx):
    minimum=min(map(len, xxx))
    z=[]
    for i,array in enumerate(xxx):
        zz=array[:minimum]
        z.append(zz)
    return(z)

###Specific functions
def normalizeV(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return (v)
    else:
        return (v / norm)

def traj_angle(x,y):
    diff=x[0]-y[0]

    newy=np.add(y,diff)

    xx=np.zeros(len(x))
    for ii in range(len(x)):
        if ii ==0:
            xx[ii]=np.nan
        else:
            AA=np.subtract(normalizeV(x[ii]),normalizeV(x[0]))
            BB=np.subtract(normalizeV(y[ii]),normalizeV(y[0]))

            xx[ii]=angle_between(AA,BB)

    return(xx)

def zscore_manual(xxx,means,stds):
    return([(xxx[ii]-means[ii])/stds[ii] for ii in range(len(means))])

def orthogonal_vector(k):
    x = np.random.randn(len(popnvec))
    x -= x.dot(k) * k / np.linalg.norm(k)**2
    return(x)

def whl_strength(whl, strength):
    xxx=np.repeat(strength,500)
    strengthbinned=binned_array(xxx,512)
    if abs(len(whl)-len(strengthbinned)) < 3:
        whl_strength_1=np.column_stack(((whl//1)[:len(strengthbinned)],strengthbinned))
        return(whl_strength_1)
    else:
        raise LookupError('whl and strength files dont match')
        
def lightONvsOFF(dicX):
    BB_last_ON=concatenate_complex2([np.hstack(dict_to_array(dicX['B']['B_last'][rec_day])) \
                                        for rec_day in Behaviour_lightON_days])

    AA_last_ON=concatenate_complex2([np.hstack(dict_to_array(dicX['A']['A_last'][rec_day])) \
                                        for rec_day in Behaviour_lightON_days])

    BB_last_OFF=concatenate_complex2([np.hstack(dict_to_array(dicX['B']['B_last'][rec_day])) \
                                        for rec_day in Behaviour_lightOFF_days])

    AA_last_OFF=concatenate_complex2([np.hstack(dict_to_array(dicX['A']['A_last'][rec_day])) \
                                        for rec_day in Behaviour_lightOFF_days])

    congruence_ON=np.hstack((AA_last_ON,BB_last_ON))
    congruence_OFF=np.hstack((AA_last_OFF,BB_last_OFF))
    
    return(congruence_OFF,congruence_ON)

def lightONvsOFF_perassembly(dicX):
    BB_last_ON=concatenate_complex2([mean_complex2(dict_to_array(dicX['B']['B_last'][rec_day]))\
                                        for rec_day in Behaviour_lightON_days])

    AA_last_ON=concatenate_complex2([mean_complex2(dict_to_array(dicX['A']['A_last'][rec_day])) \
                                        for rec_day in Behaviour_lightON_days])

    BB_last_OFF=concatenate_complex2([mean_complex2(dict_to_array(dicX['B']['B_last'][rec_day])) \
                                        for rec_day in Behaviour_lightOFF_days])

    AA_last_OFF=concatenate_complex2([mean_complex2(dict_to_array(dicX['A']['A_last'][rec_day])) \
                                        for rec_day in Behaviour_lightOFF_days])

    congruence_ON=np.hstack((AA_last_ON,BB_last_ON))
    congruence_OFF=np.hstack((AA_last_OFF,BB_last_OFF))
    
    return(congruence_OFF,congruence_ON)

def lightONvsOFF_neuron(dicX):
    BB_last_ON=concatenate_complex2([np.hstack(dict_to_array(dicX['B_last'][rec_day])) \
                                        for rec_day in Behaviour_lightON_days])

    AA_last_ON=concatenate_complex2([np.hstack(dict_to_array(dicX['A_last'][rec_day])) \
                                        for rec_day in Behaviour_lightON_days])

    BB_last_OFF=concatenate_complex2([np.hstack(dict_to_array(dicX['B_last'][rec_day])) \
                                        for rec_day in Behaviour_lightOFF_days])

    AA_last_OFF=concatenate_complex2([np.hstack(dict_to_array(dicX['A_last'][rec_day])) \
                                        for rec_day in Behaviour_lightOFF_days])

    congruence_ON=np.hstack((AA_last_ON,BB_last_ON))
    congruence_OFF=np.hstack((AA_last_OFF,BB_last_OFF))
    
    return(congruence_OFF,congruence_ON)

def cumulativeDist_plot(x,y,colorx,colory,name):
    fig, ax = plt.subplots(figsize=(3,3))
    hfont = {'fontname':'Arial'}
    x = x[x>-1E38]
    x = x[x<1E38]
    values, base = np.histogram(x, bins=40)
    cumulative = np.cumsum(values)
    plt.plot(base[:-1], cumulative, c=colorx)

    y = y[y>-1E38]
    y = y[y<1E38]
    values, base = np.histogram(y, bins=40)
    cumulative = np.cumsum(values)
    plt.plot(base[:-1], cumulative, c=colory)
    
    if name != 'none':
        plt.savefig(name)
    plt.show()

    
def bar_plot(x,y,name,pairing,ymin,ymax):
    plt.figure(figsize=(3,3))
    
    ##bars
    x_mean=np.nanmean(x)
    y_mean=np.nanmean(y)
    x_sem=st.sem(x, nan_policy='omit')
    y_sem=st.sem(y, nan_policy='omit')
    xxx=[0.35,0.65]
    data= [x, y]    
    means= [x_mean, y_mean]
    error= [x_sem, y_sem]  
    xlocations = np.array(range(len(data)))
    width=0.2
    plt.bar(xxx, means, width, yerr=error, alpha=1, color=['black','red','blue'],\
           error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2), align='center')
    #plt.margins(0.05)
    plt.ylim(ymin-(0.05*(ymax-ymin)),ymax+(0.05*(ymax-ymin)))
    plt.xlim(0,1)
    
    ##points and lines
    xx=np.column_stack((x,np.repeat(0.35,len(x))))
    yy=np.column_stack((y,np.repeat(0.65,len(y))))
    xy=np.vstack((xx,yy))
    jittered=rand_jitter(xy[:,1])
    
    if pairing == 'paired': 
        x1=np.split(jittered,2)[0]
        x2=np.split(jittered,2)[1]
        xxxx=np.column_stack((x1,x2))
        yyy=np.column_stack((x,y))
        for i in range(0,len(yy)):
            yyyy=yyy[i]
            plt.plot(xxxx[i],yyyy, color='gray')
    plt.plot(jittered,xy[:,0],'o',markersize=7,color='white',markeredgecolor='black')

    if name != 'none':
        plt.savefig(name)

    plt.show()
    
def pearson_nonan(x,y):
    xy=column_stack_clean(x,y)
    return(st.pearsonr(xy[:,0],xy[:,1]))

def extract_subset_dic(dicX, subset):
    return(np.asarray([dict_to_array(dicX[rec_day]) for rec_day in subset]))

def cumulativeDist_plot_norm(A,B,colorx,colory,binsize,name,xmin,xmax):
    fig, ax = plt.subplots(figsize=(3,3))
    hfont = {'fontname':'Arial'}
    A = A[A>-1E38]
    A = A[A<1E38]
    
    B = B[B>-1E38]
    B = B[B<1E38]
    bins=np.arange(xmin-binsize,xmax+binsize,binsize)
    
    values, base = np.histogram(A, bins=bins)
    cumulative = np.cumsum(values)/len(A)
    plt.plot(base[:-1], cumulative, c=colorx)

   
    
    #bins=np.arange(np.min(y)-binsize,np.max(y)+binsize,binsize)
    values, base = np.histogram(B, bins=bins)
    cumulative = np.cumsum(values)/len(B)
    plt.plot(base[:-1], cumulative, c=colory)
    
    plt.xlim(xmin,xmax)
    if name != 'none':
        plt.savefig(name)
    plt.show()
    
    
def smooth_MAP_gaussian(ALL_map,sigma,occupancy):

    V=ALL_map.copy()
    V[np.isnan(ALL_map)]=0
    VV=sp.ndimage.gaussian_filter(V,sigma=sigma)

    W=0*ALL_map.copy()+1
    W[np.isnan(ALL_map)]=0
    WW=sp.ndimage.gaussian_filter(W,sigma=sigma)

    Z=VV/WW
    
    Z[occupancy==0]=np.nan
    
    return(Z)

def smooth_MAPX(image_array,smf):
    new_image = image_array.copy()
    n = 0
    average_sum = 0
    for i in range(0, len(image_array)):
        for j in range(0, len(image_array[i])):
            for k in range(-smf, smf):
                for l in range(-smf, smf):
                    if (len(image_array) > (i + k) >= 0) and (len(image_array[i]) > (j + l) >= 0) and\
                    np.isnan(image_array[i+k][j+l]) != True:
                        average_sum += image_array[i+k][j+l]
                        n += 1
            if n > 0:
                new_image[i][j] = average_sum/n#(int(round(average_sum/n)))
            average_sum = 0
            n = 0
    return(new_image)  

def Place_map_minX(H,cmap,figname,interpolation):
    plt.imshow(H, cmap=cmap, interpolation=interpolation)
    plt.colorbar()
    if figname!='none':
        plt.savefig(figname)
    plt.show()

def PSTH(resi,pulses,bin_width,pre,post,figname):
    x=np.linspace(1,bin_width,bin_width)

    start = pulses[:,0].astype(int)
    end = pulses[:,1].astype(int)
    counts = [0] * bin_width

    for i in range(start.size - 1):
        tmpCounts, tmpEdges = np.histogram(resi, bins = bin_width, range = (start[i]-pre, end[i]+post))
        counts = counts + tmpCounts

    fig,ax = plt.subplots(1)

    fig=plt.bar(x,counts,color='black',width=1)

    length=np.mean(end-start)+pre+post

    plotstart=(pre*bin_width)//length
    plotend=((length-post)*bin_width)//length

    plt.axvline(x=plotstart, color='g', linestyle='solid', linewidth=1)
    plt.axvline(x=plotend, color='g', linestyle='dashed', linewidth=1)    

    xy=(plotstart,0)
    width=plotend-plotstart
    height=np.max(counts)+10
    rect=patches.Rectangle(xy, width, height, color='lightgreen', alpha=0.3)

    ax.add_patch(rect)
    
    if figname != 'none':
        plt.savefig(figname)
    plt.show()

def non_unique(a):
    s = np.sort(a, axis=None)
    xx = s[:-1][s[1:] == s[:-1]]
    return(xx)

def cosine_sim_complex(x,y):
    cosdistance_all=[]
    for i in range(0,len(x)):
        for j in range(0,len (x[i])):
            xx=x[i][j]
            yy=y[i][j]
            #print(x[i][j])
            cosdistance=1-cosine(xx,yy)
            cosdistance_all.append(cosdistance)
    return(cosdistance_all)

def innerproduct_complex(x,y):
    innerproduct_all=[]
    for i in range(0,len(x)):
        for j in range(0,len (x[i])):
            xx=x[i][j]
            yy=y[i][j]
            #print(x[i][j])
            innerproduct=np.inner(xx,yy)
            innerproduct_all.append(innerproduct)
    return(innerproduct_all)

from scipy import stats, linalg


"""Taking X and Y two variables of interest and Z the matrix with all the variable minus {X, Y},
    the algorithm can be summarized as
    1) perform a normal linear least-squares regression with X as the target and Z as the predictor
    2) calculate the residuals in Step #1
    3) perform a normal linear least-squares regression with Y as the target and Z as the predictor
    4) calculate the residuals in Step #3
    5) calculate the correlation coefficient between the residuals from Steps #2 and #4; 
    The result is the partial correlation between X and Y while controlling for the effect of Z"""

def partial_corr(C,variable,corrtype):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    P_signif = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0] ##gradient(s) of line(s) fit to the compared variables
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot( beta_i) ##residuals (actual minus predicted values)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
            
            
            if corrtype == 'pearson':
                corr = stats.pearsonr(res_i, res_j)[0] 
                signif = stats.pearsonr(res_i, res_j)[1]
            elif corrtype == 'spearman':
                corr = stats.spearmanr(res_i, res_j)[0] 
                signif = stats.spearmanr(res_i, res_j)[1]
            
            P_corr[i, j] = corr
            P_corr[j, i] = corr
            
            P_signif[i, j] = signif
            P_signif[j, i] = signif
    if variable == 'correlation':
        return (P_corr)
    elif variable == 'significance':
        return (P_signif)
    
    
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_density import KDEMultivariate

#x_grid = np.linspace(-3, 1, 1000)

def zscore_nonan(a):
    az=(a- np.nanmean(a))/np.nanstd(a)
    return(az)

def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""

    kde = gaussian_kde(x, bw_method='scott', **kwargs) #'scott'
    return kde.evaluate(x_grid)

def kde_statsmodels_m(x, x_grid, bandwidth=0.2, **kwargs):
    """Multivariate Kernel Density Estimation with Statsmodels"""
    kde = KDEMultivariate(x, bw='cv_ml',
                          var_type='c', **kwargs)
    return kde.pdf(x_grid)


def plot_kde(y,color,x_grid,y2,color2,name,show):
    fig, ax = plt.subplots(1, sharey=True,
                          figsize=(7, 7))
    fig.subplots_adjust(wspace=0)

    #pdf = kde_statsmodels_m(y, x_grid)
    pdf = kde_scipy(y, x_grid)

    ax.fill_between(x_grid, pdf, color=color, alpha=0.3, lw=0.1)
    
    if y2 != 'none':
        #pdf2 = kde_statsmodels_m(y2, x_grid)
        pdf2 = kde_scipy(y2, x_grid)
        ax.fill_between(x_grid, pdf2, color=color2, alpha=0.3, lw=0.1)

    
    xmin=np.min(x_grid)
    xmax=np.max(x_grid)
    ax.set_xlim(xmin,xmax)
    
    if name != 'none':
        plt.savefig(name)
    if show != 'none':
        plt.show()
        
def rot_matrix(x):
    rotmatrix=np.vstack((np.asarray((math.cos(x),-math.sin(x))),np.asarray((math.sin(x),math.cos(x)))))
    return(rotmatrix)
        
def bootstrap(data, n=1000, func=np.mean):
    """
    Generate `n` bootstrap samples, evaluating `func`
    at each resampling. `bootstrap` returns a function,
    which can be called to obtain confidence intervals
    of interest.
    """
    simulations = list()
    sample_size = len(data)
    xbar_init = np.mean(data)
    for c in range(n):
        itersample = np.random.choice(data, size=sample_size, replace=True)
        simulations.append(func(itersample))
    simulations.sort()
    def ci(p):
        """
        Return 2-sided symmetric confidence interval specified
        by p.
        """
        u_pval = (1+p)/2.
        l_pval = (1-u_pval)
        l_indx = int(np.floor(n*l_pval))
        u_indx = int(np.floor(n*u_pval))
        return(simulations[l_indx],simulations[u_indx])
    return(ci,simulations)

def plot_DABEST(x,y,name,ymin,ymax,N=1000,axes_keep=True,pairing=False,Multiple=False,colors=[],res=0.01):
    ##needs: rand_jitter, bootstrap
    if Multiple == True:
        xcomplex=np.copy(x)
        ycomplex=np.copy(y)

        x=np.hstack(x)
        y=np.hstack(y)

    ##points and lines
    xx=np.column_stack((x,np.repeat(0.25,len(x))))
    yy=np.column_stack((y,np.repeat(0.75,len(y))))
    xy=np.vstack((xx,yy))
    jittered=rand_jitter(xy[:,1])

    fig, ax1 = plt.subplots(figsize=(6,6))
    ax1.axhline(np.nanmean(x),color='grey',ls='dashed',xmin=0.2)
    ax1.axhline(np.nanmean(y),color='black',xmin=0.2)

    if pairing == True: 
        x1=np.split(jittered,2)[0]
        x2=np.split(jittered,2)[1]
        xxxx=np.column_stack((x1,x2))
        yyy=np.column_stack((x,y))
        for i in range(0,len(yy)):
            yyyy=yyy[i]
            ax1.plot(xxxx[i],yyyy, color='gray')
    if Multiple == True:
        cumsum=np.cumsum(np.asarray([len(xcomplex[ii]) for ii in np.arange(len(xcomplex))]))
        starts=np.hstack((0,cumsum[:-1]))
        ends=cumsum
        for iii in np.arange(len(xcomplex)):       
            ax1.plot(jittered[starts[iii]:ends[iii]],xy[starts[iii]:ends[iii],0],'o',markersize=7,\
                     color=colors[iii],markeredgecolor='black')
            ax1.plot(jittered[starts[iii]+len(x):ends[iii]+len(x)],xy[starts[iii]+len(x):ends[iii]++len(x),0]\
                 ,'o',markersize=7,color=colors[iii],markeredgecolor='black')

    elif Multiple == False and len(colors)>1:
        for jjj in np.arange(len(x)):       
            ax1.plot(jittered[jjj],xy[jjj,0],'o',markersize=7,color=colors[jjj],markeredgecolor='black')
            ax1.plot(jittered[jjj+len(x)],xy[jjj,0],'o',markersize=7,color=colors[jjj],\
                     markeredgecolor='black')


    else:
        ax1.plot(jittered,xy[:,0],'o',markersize=7,color='white',markeredgecolor='black')

    if axes_keep==False:
        ax1.set_yticklabels([])

    if pairing == True: 
        diffX=x-y
    elif pairing==False:
        diffX=np.nanmean(x)-y

    diff=remove_nan(diffX)

    ##bootstrapping based CI

    boot=bootstrap(diff,n=N)
    CI=boot[0](0.95)
    yerr=(CI[1]-CI[0])/2

    yminX=ymin-(0.05*(ymax-ymin))
    ymaxX=ymax+(0.05*(ymax-ymin))

    plt.ylim(yminX,ymaxX)

    CI_pos=1
    ax2 = ax1.twinx()
    ax2.set_ylim(yminX-np.nanmean(x),ymaxX-np.nanmean(x))
    ax2.errorbar(x=CI_pos,y=(np.nanmean(y))-np.nanmean(x),color='black',fmt='o',yerr=yerr)

    #KDE
    density = st.gaussian_kde(boot[1])
    xs = np.arange(np.min(boot[1]),np.max(boot[1]),res)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    hh=np.column_stack((density(xs),xs))
    ax2.fill_between(density(xs)*0.06/np.max(hh[:,0])+CI_pos,\
    xs+(-hh[np.argmax(hh[:,0]),1]+(np.nanmean(y))-np.nanmean(x)),
                     color='lightgrey',alpha=0.5)

    if axes_keep==False:
        ax2.set_yticklabels([])

    plt.xlim(0,1.2)    
    if name != 'none':
        plt.savefig(name)

    #plt.show()
    
def plot_DABEST_median(x,y,name,ymin,ymax,N=1000,axes_keep=True,pairing=False):

    ##needs: rand_jitter, bootstrap

    ##points and lines
    xx=np.column_stack((x,np.repeat(0.25,len(x))))
    yy=np.column_stack((y,np.repeat(0.75,len(y))))
    xy=np.vstack((xx,yy))
    jittered=rand_jitter(xy[:,1])

    fig, ax1 = plt.subplots(figsize=(6,6))
    ax1.axhline(np.nanmedian(x),color='grey',ls='dashed',xmin=0.2)
    ax1.axhline(np.nanmedian(y),color='black',xmin=0.2)

    if pairing == True: 
        x1=np.split(jittered,2)[0]
        x2=np.split(jittered,2)[1]
        xxxx=np.column_stack((x1,x2))
        yyy=np.column_stack((x,y))
        for i in range(0,len(yy)):
            yyyy=yyy[i]
            ax1.plot(xxxx[i],yyyy, color='gray')

    ax1.plot(jittered,xy[:,0],'o',markersize=7,color='white',markeredgecolor='black')
    if axes_keep==False:
        ax1.set_yticklabels([])
    
    if pairing == True: 
        diffX=x-y
    elif pairing==False:
        diffX=np.nanmedian(x)-y
    
    diff=remove_nan(diffX)

    ##bootstrapping based CI

    boot=bootstrap(diff,n=N)
    CI=boot[0](0.95)
    yerr=(CI[1]-CI[0])/2

    yminX=ymin-(0.05*(ymax-ymin))
    ymaxX=ymax+(0.05*(ymax-ymin))

    plt.ylim(yminX,ymaxX)

    CI_pos=1
    ax2 = ax1.twinx()
    ax2.set_ylim(yminX-np.nanmedian(x),ymaxX-np.nanmedian(x))
    ax2.errorbar(x=CI_pos,y=(np.nanmedian(y))-np.nanmedian(x),color='black',fmt='o',yerr=yerr)

    #KDE
    density = st.gaussian_kde(boot[1])
    xs = np.arange(np.min(boot[1]),np.max(boot[1]),0.01)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    hh=np.column_stack((density(xs),xs))
    ax2.fill_between(density(xs)*0.06/np.max(hh[:,0])+CI_pos,\
    xs+(-hh[np.argmax(hh[:,0]),1]+(np.nanmedian(y))-np.nanmedian(x)),
                     color='lightgrey',alpha=0.5)
    
    if axes_keep==False:
        ax2.set_yticklabels([])

    plt.xlim(0,1.2)    
    if name != 'none':
        plt.savefig(name)
    
    #plt.show()

        
def tone_ind(tone):
    tone_indx=tone//512
    indicesall=[]
    for i in tone_indx:
        num=i[1]-i[0]
        indices= np.linspace(i[0],i[1],num+1)
        indices=indices.astype(int)
        indicesall.append(indices)
    return(indicesall)

def tone_ind_part(tone,start,end):
    tone_indx=tone/512
    indicesall=[]
    for i in tone_indx:
        startx=start*(20000/512)
        endx=end*(20000/512)
        indices= np.linspace(i[0]+startx,i[1]+endx,endx+1-startx)
        indices=indices.astype(int)
        indicesall.append(indices)
    return(indicesall)

def tone_end_ind(tone,end):
    tone_indx=tone/512
    endx=(end*20000)/512
    indicesall=tone_indx[:,1]+endx
    indicesall=np.floor(indicesall).astype(int)
    return(indicesall)

def distance(target,location):
    dist=np.sqrt(np.square(location[:,0]-target[0])+np.square(location[:,1]-target[1]))
    return(dist)

def plot_occ_path(whlx,indices,name):
    whl = whlx[np.logical_and(whlx[:,0] > 0, whlx[:,1] > 0)] ##removing -1s
    plt.figure(figsize=(6,6))
    plt.plot(whl[:,0],whl[:,1],color='gray',alpha=0.4, lw=2)
    for iii in indices:
        plt.plot(whlx[iii,0],whlx[iii,1]) ##,color='green'
    if name != 'none':
        plt.savefig(name)
    #plt.show()

    
def plot_occ_path2(whlx,indices_end,indices_all,name):
    whl = whlx[np.logical_and(whlx[:,0] > 0, whlx[:,1] > 0)] ##removing -1s ONLY FOR PLOTTING OCCUPANCY
    plt.figure(figsize=(6,6))
    plt.plot(whl[:,0],whl[:,1],color='gray',alpha=0.4, lw=2)
    for iii in indices_all:
        if -1 not in whlx[iii,0]:
            plt.plot(whlx[iii,0],whlx[iii,1]) ##,color='green'        
    indices_clean=np.where(whlx[indices_end,0]>0)[0]
    plt.scatter(whlx[indices_end,0][indices_clean],whlx[indices_end,1][indices_clean],color='black',s=100, marker='o')
    if name != 'none':
        plt.savefig(name)
        
def plot_occ_pathX(whlx,indices_end,indices_all,Corr_coord,Incorr_coord,thr,delay,name):
    whl = whlx[np.logical_and(whlx[:,0] > 0, whlx[:,1] > 0)] ##removing -1s ONLY FOR PLOTTING OCCUPANCY
    plt.figure(figsize=(6,6))
    plt.plot(whl[:,0],whl[:,1],color='gray',alpha=0.4, lw=2)


    for iii in indices_all:
        distance_corr=distance(Corr_coord,whlx[iii])
        distance_incorr=distance(Incorr_coord,whlx[iii])
        corr_inds=np.where(distance_corr<thr)[0]
        incorr_inds=np.where(distance_incorr<thr)[0]
        if (len(np.hstack((corr_inds,incorr_inds))) > 0):
            end_inds=np.min(np.hstack((corr_inds,incorr_inds)))

            if (-1 not in whlx[iii[:end_inds+delay]]):
                if len(iii)<(end_inds+delay):
                    delay=len(iii)-end_inds-1
                plt.plot(whlx[iii[:end_inds+delay],0],whlx[iii[:end_inds+delay],1]) ##,color='green'
                plt.scatter(whlx[iii[end_inds+delay],0],whlx[iii[end_inds+delay],1],\
                            color='black',s=100, marker='o')
    if name != 'none':
        plt.savefig(name)
    
def plot_path(indices,name):
    plt.figure(figsize=(6,6))
    for iii in indices:
        plt.plot(whlx[iii,0],whlx[iii,1]) ##,color='green'
    if name != 'none':
        plt.savefig(name)
    #plt.show()
    
def plot_occ(whlx,name):
    whl = whlx[np.logical_and(whlx[:,0] > 0, whlx[:,1] > 0)] ##removing -1s
    plt.figure(figsize=(6,6))
    plt.plot(whl[:,0],whl[:,1],color='gray',alpha=0.4, lw=2)
    if name != 'none':
        plt.savefig(name)
    #plt.show()
    
def plot_occ_pos(whlx,indices,name):
    whl = whlx[np.logical_and(whlx[:,0] > 0, whlx[:,1] > 0)] ##removing -1s
    plt.figure(figsize=(6,6))
    plt.plot(whl[:,0],whl[:,1],color='gray',alpha=0.4, lw=2)
    plt.scatter(whlx[indices,0],whlx[indices,1],color='green',s=200, alpha=0.5,marker='x')
    if name != 'none':
        plt.savefig(name)
    #plt.show()
    
def reorder_vector(xx,indices):
    no_neur=len(xx)
    ind_all=(np.linspace(0,no_neur-1,no_neur)).astype(int)
    ind_remain=np.setdiff1d(ind_all,indices)
    xx_reordered=np.concatenate((xx[indices],xx[ind_remain]))
    return(xx_reordered)

def reorder_matrix(xx,indices):
    no_neur=len(xx)
    ind_all=(np.linspace(0,no_neur-1,no_neur)).astype(int)
    ind_remain=np.setdiff1d(ind_all,indices)
    xx_reordered=np.vstack((xx[indices,:],xx[ind_remain,:]))
    return(xx_reordered)

def plot_corr_matrix(x,name,length,width):
    corrx=np.corrcoef(x)

    np.fill_diagonal(corrx,0)

    fig, ax = plt.subplots(figsize=(length,width))
    cax = ax.matshow(corrx, cmap='bwr', vmin=-0.2, vmax=0.2)

    cbar = fig.colorbar(cax, ticks=[-0.2, 0, 0.2], orientation='vertical')
    cbar.ax.set_xticklabels(['-0.2', '0', '0.2'])  # horizontal colorbar
    
    if name != 'none':
        plt.savefig(name)
    #plt.show()

def reorder_xy(x,y):
    xy=np.column_stack((x,y[x]))
    xy_sortedx=xy[(-xy[:,1]).argsort()]
    xy_sorted=xy_sortedx[:,0].astype(int)
    return(xy_sorted)

def non_unique(a):
    s = np.sort(a, axis=None)
    xx = s[:-1][s[1:] == s[:-1]]
    return(xx)

def cosine_sim_complex(x,y):
    cosdistance_all=[]
    for i in range(0,len(x)):
        for j in range(0,len (x[i])):
            xx=x[i][j]
            yy=y[i][j]
            #print(x[i][j])
            cosdistance=1-cosine(xx,yy)
            cosdistance_all.append(cosdistance)
    return(cosdistance_all)

def innerproduct_complex(x,y):
    innerproduct_all=[]
    for i in range(0,len(x)):
        for j in range(0,len (x[i])):
            xx=x[i][j]
            yy=y[i][j]
            #print(x[i][j])
            innerproduct=np.inner(xx,yy)
            innerproduct_all.append(innerproduct)
    return(innerproduct_all)

def strength_quartiles(strength):
    strengthsxx4=np.array_split(strength,4)
    strengthxxmeans=np.zeros(4)
    for i in range(0,4):
        mean=np.nanmean(strengthsxx4[i])
        strengthxxmeans[i]=mean
    return(strengthxxmeans)

def strength_div(strength,division):
    strengthsxxdiv=np.array_split(strength,division)
    strengthxxmeans=np.zeros(division)
    for i in range(0,division):
        mean=np.nanmean(strengthsxxdiv[i])
        strengthxxmeans[i]=mean
    return(strengthxxmeans)

def strength_divX(strength,division):
    strengthsxxdiv=np.array_split(strength,division)
    strengthxxmeans=np.zeros((division,np.shape(strength)[1]))
    for i in range(0,division):
        mean=np.nanmean(strengthsxxdiv[i],axis=0)
        strengthxxmeans[i]=mean
    return(strengthxxmeans)

#Tracking assembly strengths (without smoothing)
def track_assembly(weights,Zb):
    P=np.outer(weights,weights)
    np.fill_diagonal(P,0) ## note this changes P 'in place' - sets main diagonal to zero
    Proj=np.inner(P,Zb)
    Rb=np.inner(Zb,Proj)
    return(Rb)

def extract_members(assemblyweights,thr,cellList):
    indx=select_members(assemblyweights,thr,'high')
    clus=cellList[indx]
    return(clus)

def slicing_arrays10(Z):
    ###generalize this!!
    Z_split=np.array_split(Z,10,axis=1)

    Z1=np.hstack((Z_split[0],Z_split[2],Z_split[4],Z_split[6],Z_split[8]))
    Z2=np.hstack((Z_split[1],Z_split[3],Z_split[5],Z_split[7],Z_split[9]))
  
    return(Z1,Z2)

def slicing_arrays10X(Z,axis):
    ###generalize this!!
    Z_split=np.array_split(Z,10,axis=axis)
    
    if axis== 0:
        Z1=np.vstack((Z_split[0],Z_split[2],Z_split[4],Z_split[6],Z_split[8]))
        Z2=np.vstack((Z_split[1],Z_split[3],Z_split[5],Z_split[7],Z_split[9]))
    if axis == 1:
        Z1=np.hstack((Z_split[0],Z_split[2],Z_split[4],Z_split[6],Z_split[8]))
        Z2=np.hstack((Z_split[1],Z_split[3],Z_split[5],Z_split[7],Z_split[9]))
        
    return(Z1,Z2)

def bin_array(array,binningarray,bins):
    interval=np.mean(np.diff(bins))
    z_all=[]
    for xx in bins[:-1]:
        zz=array[np.where((binningarray>xx) & (binningarray<(xx+interval)))[0]]
        z_all.append(zz)
    return(z_all)

def bin_arrayX(array,factor):
    bins=np.arange(len(array.T)//factor+1)*factor
    array_binned=np.vstack(([st.binned_statistic(np.arange(len(array.T)),array[ii],\
                                bins=bins,statistic=np.nanmean)[0] for ii in np.arange(len(array))]))
    return(array_binned)

def MAP(A):
    x=A[:,0].astype(int)
    y=A[:,1].astype(int)
    z=A[:,2]

    xy=np.column_stack((x,y))

    sortidx = np.lexsort(xy.T)
    sorted_coo =  xy[sortidx]

    unqID_mask = np.append(True,np.any(np.diff(sorted_coo,axis=0),axis=1)) # Get mask of start of each unique XY

    ##appending the first true as first value is unique by default


    ID = unqID_mask.cumsum()-1 ## uses cumulative sum function to rank unique elements hence giving them id


    unq_coo = sorted_coo[unqID_mask]# Get unique XY's


    average_values = np.bincount(ID,z[sortidx])/np.bincount(ID) ##N.B np.bincount(ID,z[sortidx]) is equivalent to 
    ###np.bincount(ID,weights=z[sortidx])


    xymeans=np.column_stack((unq_coo,average_values))

    return(xymeans)

def bin_MAP(mapx,bnf,option):

    from scipy.stats import binned_statistic_2d

    x=mapx[:,0]
    y=mapx[:,1]
    z=mapx[:,2]

    binx = np.linspace(np.min(x),np.max(x),bnf)
    biny = np.linspace(np.min(y),np.max(y),bnf)

    ret = binned_statistic_2d(x, y, z, 'mean', bins=[binx,biny], \
        expand_binnumbers=True)
    
    if option == 'values':
        return(ret.statistic)
    elif option == 'binnumber':
        return(ret.binnumber)
    elif option == 'x_edge':
        return(ret.x_edge)
    elif option == 'y_edge':
        return(ret.y_edge)

def bin_MAPX(mapx,bnf,option):

    from scipy.stats import binned_statistic_2d

    x=mapx[:,0]
    y=mapx[:,1]
    z=mapx[:,2]

    binx = np.linspace(np.min(x),np.max(x),bnf)
    biny = np.linspace(np.min(y),np.max(y),bnf)

    statistic,x,y,binnumber = binned_statistic_2d(x, y, z, option, bins=[binx,biny], \
        expand_binnumbers=True)
    
    return(statistic)    

def smooth_MAP(image_array,smf):
    new_image = image_array.copy()
    n = 0
    average_sum = 0
    for i in range(0, len(image_array)):
        for j in range(0, len(image_array[i])):
            for k in range(-smf, smf):
                for l in range(-smf, smf):
                    if (len(image_array) > (i + k) >= 0) and (len(image_array[i]) > (j + l) >= 0) and\
                    np.isnan(image_array[i+k][j+l]) != True:
                        average_sum += image_array[i+k][j+l]
                        n += 1
            if n > 0:
                new_image[i][j] = (int(round(average_sum/n)))
            average_sum = 0
            n = 0
    return(new_image)  

def smooth_MAPX(image_array,smf):
    new_image = image_array.copy()
    n = 0
    average_sum = 0
    for i in range(0, len(image_array)):
        for j in range(0, len(image_array[i])):
            for k in range(-smf, smf):
                for l in range(-smf, smf):
                    if (len(image_array) > (i + k) >= 0) and (len(image_array[i]) > (j + l) >= 0) and\
                    np.isnan(image_array[i+k][j+l]) != True:
                        average_sum += image_array[i+k][j+l]
                        n += 1
            if n > 0:
                new_image[i][j] = average_sum/n#(int(round(average_sum/n)))
            average_sum = 0
            n = 0
    return(new_image)  


def smooth(x,window_len,window):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return (x)


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return (y[int(window_len/2-1):-int(window_len/2)])
    
def Place_map(whl,cluwhl,bins,smf,cmap,figname):
    whl_ratex=np.column_stack((whl[:-1], cluwhl*(512)))
    whl_rate=whl_ratex[np.where((whl_ratex[:,0]>0) & (whl_ratex[:,1]>0) & (whl_ratex[:,2]!=np.nan))[0]]
    mapx=MAP(whl_rate)
    mapbinned=bin_MAP(mapx,bins,'values')
    mapsmoothed=smooth_MAP(mapbinned,smf)
    H=mapsmoothed
    Hnonan=H[~np.isnan(H)]
    plt.imshow(H, cmap=cmap, vmax=np.max(Hnonan),interpolation='gaussian')
    plt.colorbar()
    if figname!='none':
        plt.savefig(figname)
    plt.show()
    
def Place_map_min(H,cmap,figname):
    Hnonan=H[~np.isnan(H)]
    plt.imshow(H, cmap=cmap, vmax=np.max(Hnonan),interpolation='gaussian')
    plt.colorbar()
    if figname!='none':
        plt.savefig(figname)
    plt.show()

def Place_map_min_thr(H,cmap,thr,figname):
    Hnonan=H[~np.isnan(H)]
    zincrements=np.insert(st.zscore(np.diff(Hnonan[np.argsort(Hnonan)])),0,0)
    vmaxx=np.nanmax(Hnonan[np.argsort(Hnonan)[np.where(zincrements<thr)[0]]])
    plt.imshow(H, cmap=cmap, vmax=vmaxx,interpolation='gaussian')
    plt.colorbar()
    if figname!='none':
        plt.savefig(figname)
    plt.show()
    
def plot_MAP(H, figname,cmap,zero,levelno):
    Hnonan=H[~np.isnan(H)]
    #if zero == 'zero':
    #    plt.imshow(H, cmap=cmap, vmin=-np.max(Hnonan), vmax=np.max(Hnonan))
    #elif zero == 'nozero':
    #    plt.imshow(H, cmap=cmap, vmin=np.min(Hnonan), vmax=np.max(Hnonan))    
    
    if figname != 'none':
        plt.savefig(figname)
    
    fig, ax = plt.subplots()
    maxval=np.max(Hnonan)
    minval=np.min(Hnonan)
    interval= (maxval-minval)/levelno
    levels=np.arange(minval,maxval,interval)
    
    ax.contourf(H, levels=levels, cmap=cmap, vmin=minval, vmax=-minval)
    
    plt.show()


def plot_assemblyMAP_old(whl_strengthx, thr, minval, maxval, levelno, figname):
    whl_strength_nonan=extract_pandastrengths(whl_strengthx,thr)


    zline = whl_strength_nonan[:,2]# - np.nanmean(whl_strength_nonan[:,2])
    xline = whl_strength_nonan[:,0]
    yline = whl_strength_nonan[:,1]

    ##Gridding data

    numcols, numrows = 30, 30
    xi = np.linspace(np.min(xline), np.max(xline), numcols)
    yi = np.linspace(np.min(yline), np.max(yline), numrows)
    xi, yi = np.meshgrid(xi, yi)

    ## Interpolate at the points in xi, yi

    x, y, z = xline, yline, zline
    zi = griddata(x, y, z, xi, yi, interp='linear')

    ## Display the results
    fig, ax = plt.subplots()
    
    interval= (maxval-minval)/levelno
    levels=np.arange(minval,maxval,interval)
    im = ax.contourf(xi, yi, zi, levels=levels, cmap='coolwarm', vmin=minval, vmax=-minval)

    fig.colorbar(im)

    xB_mid=((mme12_170407_x_pumpB_max-mme12_170407_x_pumpB_min)/2)+mme12_170407_x_pumpB_min
    yB_mid=((mme12_170407_y_pumpB_max-mme12_170407_y_pumpB_min)/2)+mme12_170407_y_pumpB_min

    xA_mid=((mme12_170407_x_pumpA_max-mme12_170407_x_pumpA_min)/2)+mme12_170407_x_pumpA_min
    yA_mid=((mme12_170407_y_pumpA_max-mme12_170407_y_pumpA_min)/2)+mme12_170407_y_pumpA_min

    plt.plot(xA_mid, yA_mid, 'o', color='green')
    plt.plot(xB_mid, yB_mid, 'o', color='red')
    
    #circle1=plt.Circle((xA_mid, yA_mid), 30, color='g', alpha=0.2, fill=False)
    #circle2=plt.Circle((xB_mid, yB_mid), 30, color='r', alpha=0.2, fill=False)
    
    #ax.add_artist(circle1)
    #ax.add_artist(circle2)
    
    if figname != 'none':
        plt.savefig(figname)

    plt.show()
    
def plot_assemblyMAP(MAP, minval, maxval, levelno, figname):


    zline = MAP[:,2]# - np.nanmean(whl_strength_nonan[:,2])
    xline = MAP[:,0]
    yline = MAP[:,1]

    ##Gridding data

    numcols, numrows = 30, 30
    xi = np.linspace(np.min(xline), np.max(xline), numcols)
    yi = np.linspace(np.min(yline), np.max(yline), numrows)
    xi, yi = np.meshgrid(xi, yi)

    ## Interpolate at the points in xi, yi

    x, y, z = xline, yline, zline
    zi = griddata(x, y, z, xi, yi, interp='linear')

    ## Display the results
    fig, ax = plt.subplots()
    
    interval= (maxval-minval)/levelno
    levels=np.arange(minval,maxval,interval)
    im = ax.contourf(xi, yi, zi, levels=levels, cmap='coolwarm', vmin=minval, vmax=-minval)

    fig.colorbar(im)

    
    if figname != 'none':
        plt.savefig(figname)

    plt.show()
    
def plot_assemblyMAP2(H, figname,cmap,zero):
    Hnonan=H[~np.isnan(H)]
    
    fig, ax = plt.subplots()
    if zero == 'zero':
        im = ax.imshow(H, cmap=cmap, vmin=-np.max(Hnonan), vmax=np.max(Hnonan),interpolation='nearest')
    elif zero == 'nozero':
        im = ax.imshow(H, cmap=cmap, vmin=np.min(Hnonan), vmax=np.max(Hnonan),interpolation='nearest')    
    
    fig.colorbar(im)
    if figname != 'none':
        plt.savefig(figname)
        
    plt.show()
    
    
def plot_corr_matrix2(x,name,length,width,maxx,minn):
    corrx=x

    np.fill_diagonal(corrx,0)

    fig, ax = plt.subplots(figsize=(length,width))
    cax = ax.matshow(corrx, cmap='bwr', vmin=minn, vmax=maxx)

    cbar = fig.colorbar(cax, ticks=[minn, 0, maxx], orientation='vertical')
    cbar.ax.set_xticklabels([str(minn), '0', str(maxx)])  # horizontal colorbar
    
    if name != 'none':
        plt.savefig(name)
        
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return (plt.cm.get_cmap(name, n))
        
def STTC(A,B,END,dt):
    A=np.asarray(A)
    B=np.asarray(B)

    Adiff=np.diff(A)
    Bdiff=np.diff(B)

    AAdiff=np.column_stack((A[:-1],A[1:],Adiff))
    BBdiff=np.column_stack((B[:-1],B[1:],Bdiff))


    ## Calculating TA
    TAxxx=np.sum(Adiff[np.where(Adiff<dt)[0]])+(len(np.where(Adiff>=dt*2)[0]))*dt*2+\
    np.sum(Adiff[np.where(np.logical_and((Adiff>dt),(Adiff<dt*2)))[0]])+dt


    if END<(A[-1]+dt):
        TAxx=TAxxx-((A[-1]+dt)-END)
    else:
        TAxx=TAxxx

    if A[0]<26:
        TAx=TAxx+A[0]
    else:
        TAx=TAxx+25

    TA=TAx/END

    ##Calculating TB
    TBxxx=np.sum(Bdiff[np.where(Bdiff<dt)[0]])+(len(np.where(Bdiff>=dt*2)[0]))*dt*2+\
    np.sum(Bdiff[np.where(np.logical_and((Bdiff>dt),(Bdiff<dt*2)))[0]])+dt

    if END<(B[-1]+dt):
        TBxx=TBxxx-((B[-1]+dt)-END)
    else:
        TBxx=TBxxx


    if B[0]<26:
        TBx=TBxx+B[0]
    else:
        TBx=TBxx+25

    TB=TBx/END


    ##PA and PB
    lenA=len(A)
    lenB=len(B)

    numcorrA=np.zeros((lenA))
    for ii in range(0,lenA):
        ABdiff=B-A[ii]
        numcorr=len(np.where(np.logical_and(-dt<ABdiff,ABdiff<dt))[0])
        numcorrA[ii]=numcorr

    numcorrB=np.zeros((lenB))
    for ii in range(0,lenB):
        BAdiff=A-B[ii]
        numcorr=len(np.where(np.logical_and(-dt<BAdiff,BAdiff<dt))[0])
        numcorrB[ii]=numcorr


    PA=len(np.where(numcorrA>0)[0])/len(A)
    PB=len(np.where(numcorrB>0)[0])/len(B)

    STTCx=0.5*(((PA-TB)/(1-(PA*TB)))+((PB-TA)/(1-(PB*TA))))
    return(STTCx)

def remove_empty(xx):

    yy= [x for x in xx if x != []]
    return(yy)

def remove_empty(xx):

    yy= [x for x in xx if len(x) > 0]
    return(yy)

def extract_assemblyWeights_last(recording_days):
    assembly_weight_dic=rec_dd()
    for rec_day in recording_days:
        source1 = '/mnfs/ca3d2/data/melgaby_analysis/assembly/'+str(rec_day)+'/'
        for root, dirs, filenames in os.walk(source1):
            for file in filenames:
                if any( c in file for c in ('2_25_p1.APweight', '2_25.APweight')) and 'f1' not in file: ##removed '2_25_' and 30/1/19
                    fullpath = os.path.join(source1, file)
                    log = open(fullpath, 'r')
                    mousex = re.search('assembly/(\w+)', source1)
                    dayx = re.search('-(\w+)/', source1)
                    assembly_typex = re.search('(\w+).APweight', fullpath)
                    assembly_nox = re.search('APweight.(\w+)', fullpath)
                    ###
                    mouse = mousex.group(1)
                    day = dayx.group(1)
                    assembly_type = assembly_typex.group(1)
                    assembly_no = assembly_nox.group(1)

                    with open(fullpath, 'r') as f: 
                        assembly_weights = [i.split(" ") for i in f.read().split()]
                        seq=(assembly_type,'_weights_assembly', assembly_no)
                        globals()[''.join (seq)]  = np.array(assembly_weights).astype(np.float)[1:]
                        seq2=(str(assembly_type),\
                              "_weights_assembly",str(assembly_no))
                        assembly_weight_name=globals()[''.join(seq2)]
                        
                        seq3=str(mouse)+'_'+str(day)
                        assembly_weight_dic[seq3][''.join (seq)] = np.array(assembly_weight_name)\
                        .astype(np.float)
    return(assembly_weight_dic)

def extract_assemblyWeights_baseline(recording_days):
    assembly_weight_dic=rec_dd()
    for rec_day in recording_days:
        source1 = '/mnfs/ca3d2/data/melgaby_analysis/assembly/'+str(rec_day)+'/'
        for root, dirs, filenames in os.walk(source1):
            for file in filenames:
                if any( c in file for c in ('0_25_p1.APweight', '0_25.APweight')) and 'f1' not in file: ##removed '2_25_' and 30/1/19
                    fullpath = os.path.join(source1, file)
                    log = open(fullpath, 'r')
                    mousex = re.search('assembly/(\w+)', source1)
                    dayx = re.search('-(\w+)/', source1)
                    assembly_typex = re.search('(\w+).APweight', fullpath)
                    assembly_nox = re.search('APweight.(\w+)', fullpath)
                    ###
                    mouse = mousex.group(1)
                    day = dayx.group(1)
                    assembly_type = assembly_typex.group(1)
                    assembly_no = assembly_nox.group(1)

                    with open(fullpath, 'r') as f: 
                        assembly_weights = [i.split(" ") for i in f.read().split()]
                        seq=(assembly_type,'_weights_assembly', assembly_no)
                        globals()[''.join (seq)]  = np.array(assembly_weights).astype(np.float)[1:]
                        seq2=(str(assembly_type),\
                              "_weights_assembly",str(assembly_no))
                        assembly_weight_name=globals()[''.join(seq2)]
                        
                        seq3=str(mouse)+'_'+str(day)
                        assembly_weight_dic[seq3][''.join (seq)] = np.array(assembly_weight_name)\
                        .astype(np.float)
    return(assembly_weight_dic)

def extract_assemblyWeights_f1(recording_days):
    assembly_weight_dic=rec_dd()
    for rec_day in recording_days:
        source1 = '/mnfs/ca3d2/data/melgaby_analysis/assembly/'+str(rec_day)+'/'
        for root, dirs, filenames in os.walk(source1):
            for file in filenames:
                if any( c in file for c in ('f1_1_25_p1.APweight', 'f1_1_25.APweight')):
                    fullpath = os.path.join(source1, file)
                    log = open(fullpath, 'r')
                    mousex = re.search('assembly/(\w+)', source1)
                    dayx = re.search('-(\w+)/', source1)
                    assembly_typex = re.search('(\w+).APweight', fullpath)
                    assembly_nox = re.search('APweight.(\w+)', fullpath)
                    ###
                    mouse = mousex.group(1)
                    day = dayx.group(1)
                    assembly_type = assembly_typex.group(1)
                    assembly_no = assembly_nox.group(1)

                    with open(fullpath, 'r') as f: 
                        assembly_weights = [i.split(" ") for i in f.read().split()]
                        seq=(assembly_type,'_weights_assembly', assembly_no)
                        globals()[''.join (seq)]  = np.array(assembly_weights).astype(np.float)[1:]
                        seq2=(str(assembly_type),\
                              "_weights_assembly",str(assembly_no))
                        assembly_weight_name=globals()[''.join(seq2)]
                        
                        seq3=str(mouse)+'_'+str(day)
                        assembly_weight_dic[seq3][''.join (seq)] = np.array(assembly_weight_name)\
                        .astype(np.float)
    return(assembly_weight_dic)

def strength_quartiles(strength):
    strengthsxx4=np.array_split(strength,4)
    strengthxxmeans=np.zeros(4)
    for i in range(0,4):
        mean=np.nanmean(strengthsxx4[i])
        strengthxxmeans[i]=mean
    return(strengthxxmeans)

def x_interval(x,interval,sampratio):
    interval=interval//sampratio
    xinterval_all=[]
    for ii in range(0,len(interval)):
        xinterval=np.linspace(interval[ii][0],interval[ii][1],(interval[ii][1]-interval[ii][0]))
        xinterval_all.append(xinterval)
    xinterval_all=np.concatenate(xinterval_all).astype(int)
    
    if len(x) in xinterval_all:
        xinterval_all=np.delete(xinterval_all,-1)
    x_interval=x[xinterval_all]
    return(x_interval)

def outside_intervals(ax):
    x=ax[:,1]
    xx=np.insert(x,0,0)
    y=ax[:,0]
    yy=np.append(y,999999999999)
    a=np.column_stack((xx,yy))
    return(a)

def emergence_point(y,thr):
    dip_point=x[np.where(y<thr)[0]]
    
    emergence_pointx=x[np.where(y>thr)[0]]
    
    if len(emergence_pointx)>0 and len(dip_point)>0:
        stable_emergence_pointx=emergence_pointx[np.where(emergence_pointx>dip_point[-1])[0]]
        if len(stable_emergence_pointx)>0:
            emergence_point=stable_emergence_pointx[0]
        else:
            emergence_point=np.nan
    elif len(emergence_pointx)>0 and len(dip_point)==0:
        emergence_point=emergence_pointx[0]
    else:
        emergence_point=np.nan
    return(emergence_point)   


def whl_speed_strength(whl,strength,speed):
    iis=[]
    for i in range(0, len(whl)):
        ii = (i*512)//500
        if ii < len(strength):
            iis.append(ii)
        else:
            iis.append(ii-1)
    
    speed10=np.repeat(speed,10)
    
    whl_strength_1=np.column_stack((whl//1,strength[iis]))
    whl_strength_2=np.column_stack((whl_strength_1[:len(speed10)], speed10))
    whl_strength = whl_strength_2[np.logical_and(whl_strength_2[:,0] > 0, whl_strength_2[:,1] > 0)] ##removing -1s
    return(whl_strength)


def noplot_timecourseA(x,y,color):
    ymean=np.nanmean(y, axis=0)
    yerr=st.sem(y, axis=0, nan_policy='omit')
    plt.errorbar(x,ymean, yerr=yerr, color=color, marker='o')
    
def noplot_timecourseAx(x,y,color):
    ymean=mean_complex2(y)
    yerr=[st.sem(i,nan_policy='omit') for i in y]
    plt.errorbar(x,ymean, yerr=yerr, color=color, marker='o')
    
def noplot_timecourseB(x,y,color):
    ymean=np.nanmean(y, axis=0)
    yerr=st.sem(y, axis=0, nan_policy='omit')
    plt.errorbar(x,y=ymean, color=color, marker='o')
    plt.fill_between(x, ymean-yerr, ymean+yerr, color=color,alpha=0.5)
    
def noplot_timecourseBx(x,y,color):
    ymean=mean_complex2(y)
    yerr=[st.sem(i,nan_policy='omit') for i in y]
    plt.errorbar(x,y=ymean, color=color, marker='o')
    plt.fill_between(x, ymean-yerr, ymean+yerr, color=color,alpha=0.5)

    
def plot_corr_matrix(x,name,length,width):
    corrx=np.corrcoef(x)

    np.fill_diagonal(corrx,0)

    fig, ax = plt.subplots(figsize=(length,width))
    cax = ax.matshow(corrx, cmap='bwr', vmin=-0.2, vmax=0.2)

    cbar = fig.colorbar(cax, ticks=[-0.2, 0, 0.2], orientation='vertical')
    cbar.ax.set_xticklabels(['-0.2', '0', '0.2'])  # horizontal colorbar
    
    if name != 'none':
        plt.savefig(name)
    #plt.show()
    
def plot_corr_matrix2(x,name,length,width,maxx,minn):
    corrx=x

    np.fill_diagonal(corrx,0)

    fig, ax = plt.subplots(figsize=(length,width))
    cax = ax.matshow(corrx, cmap='bwr', vmin=minn, vmax=maxx)

    cbar = fig.colorbar(cax, ticks=[minn, 0, maxx], orientation='vertical')
    cbar.ax.set_xticklabels([str(minn), '0', str(maxx)])  # horizontal colorbar
    
    if name != 'none':
        plt.savefig(name)
         
def maximize_diagonal(matrix):
    matrixT=matrix.T
    indx=[]
    for ii in np.arange(len(matrixT)):
        indx_sorted=np.argsort(matrixT[ii])
        indx_sorted_nooverlap=[i for i in indx_sorted if i not in indx]
        if len(indx)>0 and len(indx_sorted_nooverlap)>0:
            maxx=indx_sorted_nooverlap[-1]
        elif len(indx)==0 and len(indx_sorted_nooverlap)>0:
            maxx=indx_sorted[-1]
        indx.append(maxx)

    ALL_indices=np.arange(np.shape(matrix)[0])
    remaining_indices=np.setdiff1d(ALL_indices,indx)
    indxX=np.hstack((indx,remaining_indices))

    xx = np.unique(indxX, return_index=True)[1]
    indices=np.asarray([indxX[index] for index in sorted(xx)])
    return(matrix[indices])


def corr_matrix_nan(xx):
    corrxy_all=np.zeros((len(xx),len(xx)))
    for ii in np.arange(len(xx)):
        for jj in np.arange(len(xx)):
            xy=column_stack_clean(xx[ii],xx[jj])
            corrxy=st.pearsonr(st.zscore(xy[:,0]),st.zscore(xy[:,1]))[0]
            corrxy_all[ii][jj]=corrxy

    return(corrxy_all)

def flatcorr_matrix_nan(xx):
    iijj_=[]
    corrxy_all=[]
    for ii in np.arange(len(xx)):
        for jj in np.arange(len(xx)):
            iijj=ii,jj
            if ii !=jj and iijj not in iijj_:
                xy=column_stack_clean(xx[ii],xx[jj])
                corrxy=st.pearsonr(st.zscore(xy[:,0]),st.zscore(xy[:,1]))[0]
                corrxy_all.append(corrxy)
                iijj_.append((jj,ii))
    return(corrxy_all)

def bin_strengths(strength,binsize,nan_policy):
    if nan_policy=='omit':
        binsize = max(1, binsize)
        xxx=(strength[i:i+binsize] for i in range(0, len(strength), binsize))
        binned_strength=mean_complex2(np.asarray(list(xxx)))
    else:
        x=np.arange(len(strength))
        end=len(strength)//binsize
        bins=(np.hstack((0,np.linspace(binsize,end*binsize,end)))).astype(int)
        binned_strength,edges,binnumber=st.binned_statistic(x,strength,bins=bins)

    return(binned_strength)

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def bin_strengths_nans(l, n):
    n = max(1, n)
    xxx=(l[i:i+n] for i in range(0, len(l), n))
    return(mean_complex2(np.asarray(list(xxx))))

def prune_array(xx,lenx):
    xx_new=np.zeros((len(xx),lenx))
    for ii in range(len(xx)):
        xx_new[ii]=xx[ii][:lenx]
    return(xx_new)

def skaggs_information(whl,cluwhl_ALL,bins):
    #I(R∣X)≈∑ip(x⃗ i)f(x⃗ i)log2(f(x⃗ i)F)
    #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2680063/
    #But see: https://www.biorxiv.org/content/biorxiv/early/2017/09/15/189084.full.pdf
    
    #whl,cluwhl_ALL,bins=whl_rrem,whlstrength_rremX,36

    binsqrt=int(np.sqrt(bins))

    skaggs_all=np.zeros((len(cluwhl_ALL),(binsqrt-1)**2))
    #skaggssum_all=np.zeros((len(cluwhl_ALL)))
    for iii in range(0,len(cluwhl_ALL)):
        cluwhl=cluwhl_ALL[iii]

        thr=2*np.std(cluwhl)
        Activations=np.zeros(len(cluwhl))
        Activations[np.where(cluwhl>thr)[0]]=1

        mean_activations=(len(np.where(cluwhl>thr))/len(cluwhl))*(20000/512)
        if len(whl) == len(cluwhl)+1:
            whl_ratex=np.column_stack((whl[:-1], Activations))
        else:
            whl_ratex=np.column_stack((whl, Activations))
        whl_rate=whl_ratex[np.where((whl_ratex[:,0]>0) & (whl_ratex[:,1]>0) & (whl_ratex[:,2]!=np.nan))[0]]
        mapx=MAP(whl_rate)
        mapbinned=bin_MAP(mapx,binsqrt,'values')*(20000/512)
        mapbinned_flat=np.concatenate(mapbinned)

        occupancy_map=np.asarray(bin_MAPX(mapx,binsqrt,'count'))
        probability_map=np.concatenate(occupancy_map/np.sum(occupancy_map))

        for position in range(len(mapbinned_flat)):
            skaggs=probability_map[position]*(mapbinned_flat[position])*\
            np.log2((mapbinned_flat[position])/mean_activations)
            skaggs_all[iii][position]=skaggs

    skaggs_sum=np.nansum(skaggs_all,axis=1)
    return(skaggs_sum/mean_activations)    


##decoding functions - scikit
def reshape_crossings(corrx,incorrx):
    corrxx=np.asarray(corrx).T
    incorrxx=np.asarray(incorrx).T

    x=len(corrx[0])
    xx=np.linspace(1,1,x)
    y=len(incorrx[0])
    yy=np.linspace(0,0,y)
    corr=np.column_stack((corrxx,xx))
    incorr=np.column_stack((incorrxx,yy))
    total=np.vstack((corr,incorr))
    total=total[~np.isnan(total).any(axis=1)]
    return(total)

def reshape_crossings2(corrx,incorrx,corry,incorry):
    corrxx=np.asarray(corrx).T
    incorrxx=np.asarray(incorrx).T
    corryy=np.asarray(corry).T
    incorryy=np.asarray(incorry).T

    x=len(corrx[0])
    xx=np.linspace(1,1,x)
    y=len(incorrx[0])
    yy=np.linspace(0,0,y)
    corr=np.column_stack((corrxx,xx))
    incorr=np.column_stack((incorrxx,yy))
    
    x1=len(corry[0])
    xx1=np.linspace(3,3,x1)
    y1=len(incorry[0])
    yy1=np.linspace(2,2,y1)
    corr1=np.column_stack((corryy,xx1))
    incorr1=np.column_stack((incorryy,yy1))
    
    total=np.vstack((corr,incorr,corr1,incorr1))
    total=total[~np.isnan(total).any(axis=1)]
    return(total)

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


def decode_sciKit(data,priors):
    gnb = GaussianNB(priors=priors)
    X=data[:,:-1]
    X=X[:,~np.any(np.isnan(X),axis=0)]
    Y=data[:,-1]
    y_pred_all=[]
    if len(np.where(Y==0)[0])>1 and len(np.where(Y==1)[0])>1:
        for i in range(0, len(X)):
            Xtrain=np.delete(X,(i),axis=0)
            Ytrain=np.delete(Y,(i),axis=0)
            y_pred = gnb.fit(Xtrain,Ytrain).predict([X[i]])
            y_pred_all.extend(y_pred)

        return(getAccuracy(data,y_pred_all))
    else:
        return(np.nan)
 
    
def decode_Logit(data):
    logit = LogisticRegression()
    X=data[:,:-1]
    X=X[:,~np.any(np.isnan(X),axis=0)]
    Y=data[:,-1]
    y_pred_all=[]
    if len(np.where(Y==0)[0])>1 and len(np.where(Y==1)[0])>1:
        for i in range(0, len(X)):
            Xtrain=np.delete(X,(i),axis=0)
            Ytrain=np.delete(Y,(i),axis=0)
            y_pred = logit.fit(Xtrain,Ytrain).predict([X[i]])
            y_pred_all.extend(y_pred)

        return(getAccuracy(data,y_pred_all))
    else:
        return(np.nan)
    
def decode_GNB_Xsession(data_training,data_test,priors):
    gnb = GaussianNB(priors=priors)
    X_training=data_training[:,:-1]
    X_training=X_training[~np.all(np.isnan(X_training),axis=1)] ##removing rows (trials) with no valid entries
    X_training=X_training[:,~np.any(np.isnan(X_training),axis=0)]
    Y_training=data_training[:,-1]
    
    X_test=data_test[:,:-1]
    X_test=X_test[~np.all(np.isnan(X_test),axis=1)] ##removing rows (trials) with no valid entrie
    X_test=X_test[:,~np.any(np.isnan(X_test),axis=0)]
    Y_test=data_test[:,-1]
    y_pred_all=[]
    model=gnb.fit(X_training,Y_training)
    for i in range(0, len(X_test)):
        y_pred = model.predict([X_test[i]])
        y_pred_all.extend(y_pred)
    y_prob=np.asarray(model.predict_proba(X_test))
    
    return(getAccuracy(data_test[~np.all(np.isnan(data_test[:,:-1]),axis=1)],y_pred_all),np.asarray(y_pred_all),y_prob)

def decode_Logits_Xsession(data_training,data_test):
    logit = LogisticRegression()
    X_training=data_training[:,:-1]
    X_training=X_training[~np.all(np.isnan(X_training),axis=1)] ##removing rows (trials) with no valid entries
    X_training=X_training[:,~np.any(np.isnan(X_training),axis=0)]
    Y_training=data_training[:,-1]
    
    X_test=data_test[:,:-1]
    X_test=X_test[~np.all(np.isnan(X_test),axis=1)] ##removing rows (trials) with no valid entries
    X_test=X_test[:,~np.any(np.isnan(X_test),axis=0)]
    Y_test=data_test[:,-1]
    y_pred_all=[]
    model=logit.fit(X_training,Y_training)
    for i in range(0, len(X_test)):
        y_pred = model.predict([X_test[i]])
        y_pred_all.extend(y_pred)
    y_prob=np.asarray(model.predict_proba(X_test))
    
    return(getAccuracy(data_test[~np.all(np.isnan(data_test[:,:-1]),axis=1)],y_pred_all),np.asarray(y_pred_all),y_prob)
    
def decode_sciKit_IMP(data,priors):
    gnb = GaussianNB(priors=priors)
    X=data[:,:-1]
    imp = Imputer(missing_values=np.nan, strategy='mean')
    imp = imp.fit(X)
    X_imp = imp.transform(X)
    Y=data[:,-1]
    y_pred_all=[]
    if len(np.where(Y==0)[0])>1 and len(np.where(Y==1)[0])>1:
        for i in range(0, len(X)):
            Xtrain=np.delete(X_imp,(i),axis=0)
            Ytrain=np.delete(Y,(i),axis=0)
            y_pred = gnb.fit(Xtrain,Ytrain).predict([X_imp[i]])
            y_pred_all.extend(y_pred)

        return(getAccuracy(data,y_pred_all))
    else:
        return(np.nan)


def decode_Logit_IMP(data):
    logit = LogisticRegression()
    X=data[:,:-1]
    imp = Imputer(missing_values=np.nan, strategy='mean')
    imp = imp.fit(X)
    X_imp = imp.transform(X)
    Y=data[:,-1]
    y_pred_all=[]
    if len(np.where(Y==0)[0])>1 and len(np.where(Y==1)[0])>1:
        for i in range(0, len(X)):
            Xtrain=np.delete(X_imp,(i),axis=0)
            Ytrain=np.delete(Y,(i),axis=0)
            y_pred = logit.fit(Xtrain,Ytrain).predict([X_imp[i]])
            y_pred_all.extend(y_pred)

        return(getAccuracy(data,y_pred_all))
    else:
        return(np.nan)

def decode_Logit_coefs(data):
    logit = LogisticRegression()
    X=data[:,:-1]
    X=X[:,~np.any(np.isnan(X),axis=0)]
    Y=data[:,-1]
    y_coef_all=[]
    if len(np.where(Y==0)[0])>1 and len(np.where(Y==1)[0])>1:
        for i in range(0, len(X)):
            Xtrain=np.delete(X,(i),axis=0)
            Ytrain=np.delete(Y,(i),axis=0)
            y_pred = logit.fit(Xtrain,Ytrain).predict([X[i]])
            coef=logit.coef_
            y_coef_all.append(coef)
        y_coef_mean=np.concatenate(np.mean(y_coef_all, axis=0))
        return(y_coef_mean)
    else:
        return(np.nan)
    
def extract_subset_dic(dicX, subset):
    return(np.asarray([dict_to_array(dicX[rec_day]) for rec_day in subset]))

def resampled_decoding(data,method,priors):
    resampled_distrib=[]
    for i in range(0,1000):
        samplex=data[:,-1]
        #allsample=xxx
        sample=np.random.permutation(samplex)
        resampled=np.column_stack((data[:,:-1],sample))
        if method == 'scikitGNB':
            x=decode_sciKit(resampled,priors)
        elif method == 'scikitLogit':
            x=decode_Logit(resampled)
        xx=np.nanmean(x)
        resampled_distrib.append(xx)
    return(resampled_distrib)

def resampled_decodingX(data,method,priors,iterations):
    resampled_distrib=[]
    for i in range(0,iterations):
        samplex=data[:,-1]
        #allsample=xxx
        sample=np.random.permutation(samplex)
        resampled=np.column_stack((data[:,:-1],sample))
        if method == 'scikitGNB':
            x=decode_sciKit(resampled,priors)
        elif method == 'scikitLogit':
            x=decode_Logit(resampled)
        xx=np.nanmean(x)
        resampled_distrib.append(xx)
    return(resampled_distrib)

def decode_sciKit2(data,priors):
    gnb = GaussianNB(priors=priors)
    X=data[:,:-1]
    X=X[:,~np.any(np.isnan(X),axis=0)]
    Y=data[:,-1]
    y_pred_all=[]
    for i in range(0, len(X)):
        Xtrain=np.delete(X,(i),axis=0)
        Ytrain=np.delete(Y,(i),axis=0)
        y_pred = gnb.fit(Xtrain,Ytrain).predict([X[i]])
        y_pred_all.extend(y_pred)

    return(getAccuracy(data,y_pred_all))

def decode_Logit2(data):
    logit = LogisticRegression()
    X=data[:,:-1]
    X=X[:,~np.any(np.isnan(X),axis=0)]
    Y=data[:,-1]
    y_pred_all=[]
    for i in range(0, len(X)):
        Xtrain=np.delete(X,(i),axis=0)
        Ytrain=np.delete(Y,(i),axis=0)
        y_pred = logit.fit(Xtrain,Ytrain).predict([X[i]])
        y_pred_all.extend(y_pred)

    return(getAccuracy(data,y_pred_all))
def resampled_decodingX2(data,method,priors,iterations):
    resampled_distrib=[]
    for i in range(0,iterations):
        samplex=data[:,-1]
        #allsample=xxx
        sample=np.random.permutation(samplex)
        resampled=np.column_stack((data[:,:-1],sample))
        if method == 'scikitGNB':
            x=decode_sciKit2(resampled,priors)
        elif method == 'scikitLogit':
            x=decode_Logit2(resampled)
        xx=np.nanmean(x)
        resampled_distrib.append(xx)
    return(resampled_distrib)


def resampled_decoding_IMP(data,method,priors,iterations):
    resampled_distrib=[]
    for i in range(0,iterations):
        samplex=data[:,-1]
        #allsample=xxx
        sample=np.random.permutation(samplex)
        resampled=np.column_stack((data[:,:-1],sample))
        if method == 'scikitGNB':
            x=decode_sciKit_IMP(resampled,priors)
        elif method == 'scikitLogit':
            x=decode_Logit_IMP(resampled)
        xx=np.nanmean(x)
        resampled_distrib.append(xx)
    return(resampled_distrib)

def feature_contributions(data,priors):
    #priors=np.array([0.5,0.5])
    gnb = GaussianNB(priors=priors)
    #data=reshape_crossings(corrx,incorrx)
    X=data[:,:-1]
    Y=data[:,-1]

    y_pred_all=[]
    for i in range(0, len(X)):
        Xtrain=np.delete(X,(i),axis=0)
        Ytrain=np.delete(Y,(i),axis=0)
        y_pred = gnb.fit(Xtrain,Ytrain).predict([X[i]])
        y_pred_all.extend(y_pred)
    total_accuracy=getAccuracy(data,y_pred_all)
    
    delta_accuracy_all=[]     
    for j in range(0,np.shape(X)[1]):
        X_=np.delete(X,(j),axis=1)
        Y_=Y
        data_=np.delete(data,(j),axis=1)
        y_pred_all=[]
        for i in range(0, len(X)):
            Xtrain=np.delete(X_,(i),axis=0)
            Ytrain=np.delete(Y_,(i),axis=0)

            y_pred = gnb.fit(Xtrain,Ytrain).predict([X_[i]])
            y_pred_all.extend(y_pred)
        
        accuracy=getAccuracy(data_,y_pred_all)
        delta_accuracy=total_accuracy-accuracy
        delta_accuracy_all.append(delta_accuracy)
    return(delta_accuracy_all)
            
            
            
###########################
####EXTRACTING STUFF#######
###########################


def extract_pulse(rec_days):
    #Extracting pulse, whl and rrem files
    ##'mme03-160724' 

    pulse_dic={}
    for i in rec_days:
        source2 = '/mnfs/ca3d2/data/melgaby_merged/'+str(i)+'/'
        for root, dirs, filenames in os.walk(source2):
            for file in filenames:
                filebsnm=str(i)+'_'
                if str(filebsnm) in file and 'pulse' in file and not 'tracking' in file and not '~' in file:
                    fullpath = os.path.join(source2, file)
                    log = open(fullpath, 'r')
                    mousex = re.search('merged/(\w+)', source2)
                    dayx = re.search('-(\w+)/', source2)
                    ttlx = re.search('.(\w+)_pulse', fullpath)
                    sessionx = re.search('_(\w+)\.', fullpath)
                    ###
                    mouse = mousex.group(1)
                    day = dayx.group(1)
                    ttl = ttlx.group(1)
                    session = sessionx.group(1)
                    #print(session)
                    if os.path.getsize(fullpath) > 0:
                        #print(fullpath)
                        with open(fullpath, 'r') as f:
                            ttl_start1=[]
                            for start in f:
                                ttl_start1.append(start.split()[0])
                            seq=(mouse, '_', day, '_', ttl, '_start_ses', session)
                            #pulse_dic[''.join (seq)] = np.array(ttl_start1).astype(np.float)
                            globals()[''.join (seq)]  = np.array(ttl_start1).astype(np.float)
                        with open(fullpath, 'r') as f:
                            ttl_end1=[]
                            for end in f:
                                ttl_end1.append(end.split()[1])
                            seq=(mouse, '_', day, '_', ttl, '_end_ses', session)
                            #pulse_dic[''.join (seq)] = np.array(ttl_start1).astype(np.float)
                            globals()[''.join (seq)]  = np.array(ttl_end1).astype(np.float)
                        exec(str(mouse)+'_'+str(day)+'_'+str(ttl)+'_ses'+str(session)+' = np.column_stack(( '\
                             +str(mouse)+'_'+str(day)+'_'+str(ttl)+'_start_ses'+str(session)+', '+str(mouse)+'_'+\
                             str(day)+'_'+str(ttl)+'_end_ses'+str(session)+'))')
                        #creating dictionary with all pulse times for each pulse file
                        seq=(mouse, '_', day, '_', ttl, '_ses', session)
                        exec("pulse_dic[''.join (seq)] = np.array("+str(mouse)+"_"+str(day)+"_"+str(ttl)+"_ses"+\
                             str(session)+").astype(np.float)")
    return(pulse_dic)

def extract_pulse2(rec_days, source):
    #Extracting pulse, whl and rrem files
    ##'mme03-160724' 

    pulse_dic={}
    for i in rec_days:
        source2 = source+str(i)+'/'
        for root, dirs, filenames in os.walk(source2):
            for file in filenames:
                filebsnm=str(i)+'_'
                if str(filebsnm) in file and 'pulse' in file and not 'tracking' in file and not '~' in file:
                    fullpath = os.path.join(source2, file)
                    if os.path.exists(fullpath):
                        log = open(fullpath, 'r')
                        mousex=re.search('(\w+)-', i)
                        dayx=re.search('-(\w+)', i)
                        ttlx = re.search('.(\w+)_pulse', fullpath)
                        sessionx = re.search('_(\w+)\.', fullpath)
                        ###
                        mouse = mousex.group(1)
                        day = dayx.group(1)
                        ttl = ttlx.group(1)
                        session = sessionx.group(1)
                        #print(session)
                        if os.path.getsize(fullpath) > 0:
                            #print(fullpath)
                            with open(fullpath, 'r') as f:
                                ttl_start1=[]
                                for start in f:
                                    ttl_start1.append(start.split()[0])
                                seq=(mouse, '_', day, '_', ttl, '_start_ses', session)
                                #pulse_dic[''.join (seq)] = np.array(ttl_start1).astype(np.float)
                                globals()[''.join (seq)]  = np.array(ttl_start1).astype(np.float)
                            with open(fullpath, 'r') as f:
                                ttl_end1=[]
                                for end in f:
                                    ttl_end1.append(end.split()[1])
                                seq=(mouse, '_', day, '_', ttl, '_end_ses', session)
                                #pulse_dic[''.join (seq)] = np.array(ttl_start1).astype(np.float)
                                globals()[''.join (seq)]  = np.array(ttl_end1).astype(np.float)
                            exec(str(mouse)+'_'+str(day)+'_'+str(ttl)+'_ses'+str(session)+' = np.column_stack(( '\
                                 +str(mouse)+'_'+str(day)+'_'+str(ttl)+'_start_ses'+str(session)+', '+str(mouse)+'_'+\
                                 str(day)+'_'+str(ttl)+'_end_ses'+str(session)+'))')
                            #creating dictionary with all pulse times for each pulse file
                            seq=(mouse, '_', day, '_', ttl, '_ses', session)
                            exec("pulse_dic[''.join (seq)] = np.array("+str(mouse)+"_"+str(day)+"_"+str(ttl)+"_ses"+\
                                 str(session)+").astype(np.float)")
    return(pulse_dic)


def extract_whl(rec_days):
    whl_dic={}
    for i in rec_days:
        source2 = '/mnfs/ca3d2/data/melgaby_merged/'+str(i)+'/'
        for root, dirs, filenames in os.walk(source2):
            for file in filenames:
                filebsnm=str(i)+'_'
                if str(filebsnm) in file and 'whl' in file and not 'binned' in file and not '~' in file:
                    fullpath = os.path.join(source2, file)
                    if os.path.exists(fullpath):
                        log = open(fullpath, 'r')
                        mousex = re.search('merged/(\w+)', source2)
                        dayx = re.search('-(\w+)/', source2)
                        sessionx = re.search('_(\w+)\.', fullpath)
                        ###
                        mouse = mousex.group(1)
                        day = dayx.group(1)
                        session = sessionx.group(1)
                        #print(session)
                        with open(fullpath, 'r') as f:
                            whl_x=[]
                            for x in f:
                                whl_x.append(x.split()[0])
                            seq=(mouse, '_', day, '_whl_x_ses', session)
                            globals()[''.join (seq)]  = np.array(whl_x).astype(np.float)
                        with open(fullpath, 'r') as f:
                            whl_y=[]
                            for y in f:
                                whl_y.append(y.split()[1])
                            seq=(mouse, '_', day, '_whl_y_ses', session)
                            globals()[''.join (seq)]  = np.array(whl_y).astype(np.float)
                        exec(str(mouse)+'_'+str(day)+'_whl_ses'+str(session)+' = np.column_stack(( '+str(mouse)+'_'+\
                             str(day)+'_whl_x_ses'+str(session)+', '+str(mouse)+'_'+str(day)+'_whl_y_ses'+str(session)+'))')
                        #creating dictionary with all pulse times for each pulse file
                        seq=(mouse, '_', day, '_whl_ses', session)
                        exec("whl_dic[''.join (seq)] = np.array("+str(mouse)+"_"+str(day)+"_whl_ses"+str(session)+\
                             ").astype(np.float)")

    return(whl_dic)

def extract_whl2(rec_days, source):
    whl_dic={}
    for i in rec_days:
        source2 = source+str(i)+'/'
        for root, dirs, filenames in os.walk(source2):
            for file in filenames:
                filebsnm=str(i)+'_'
                if str(filebsnm) in file and 'whl' in file and not 'binned' in file and not '~' in file:
                    fullpath = os.path.join(source2, file)
                    if os.path.exists(fullpath):
                        log = open(fullpath, 'r')
                        mousex=re.search('(\w+)-', i)
                        dayx=re.search('-(\w+)', i)
                        sessionx = re.search('_(\w+)\.', fullpath)
                        ###
                        mouse = mousex.group(1)
                        day = dayx.group(1)
                        session = sessionx.group(1)
                        #print(session)
                        with open(fullpath, 'r') as f:
                            whl_x=[]
                            for x in f:
                                whl_x.append(x.split()[0])
                            seq=(mouse, '_', day, '_whl_x_ses', session)
                            globals()[''.join (seq)]  = np.array(whl_x).astype(np.float)
                        with open(fullpath, 'r') as f:
                            whl_y=[]
                            for y in f:
                                whl_y.append(y.split()[1])
                            seq=(mouse, '_', day, '_whl_y_ses', session)
                            globals()[''.join (seq)]  = np.array(whl_y).astype(np.float)
                        exec(str(mouse)+'_'+str(day)+'_whl_ses'+str(session)+' = np.column_stack(( '+str(mouse)+'_'+\
                             str(day)+'_whl_x_ses'+str(session)+', '+str(mouse)+'_'+str(day)+'_whl_y_ses'+str(session)+'))')
                        #creating dictionary with all pulse times for each pulse file
                        seq=(mouse, '_', day, '_whl_ses', session)
                        exec("whl_dic[''.join (seq)] = np.array("+str(mouse)+"_"+str(day)+"_whl_ses"+str(session)+\
                             ").astype(np.float)")

    return(whl_dic)

#Extracting binned assembly strengths
def extract_binned_assemblyStrength(recording_days):
    assembly_strength_dic={}
    #Extracting binned assembly strengths
    for i in recording_days:
        source1 = '/mnfs/ca3d2/data/melgaby_analysis/assembly/'+str(i)+'/'
        for root, dirs, filenames in os.walk(source1):
            for file in filenames:
                if '.APbinnedStrength_' in file and 'all.' in file:
                    fullpath = os.path.join(source1, file)
                    log = open(fullpath, 'r')
                    mousex = re.search('assembly/(\w+)', source1)
                    dayx = re.search('-(\w+)/', source1)
                    assembly_typex = re.search('(\w+).APbinnedStrength', fullpath)
                    sessionx = re.search('.APbinnedStrength_(\w+)all', fullpath)
                    assembly_nox = re.search('all.1.(\w+)', fullpath)
                    ###
                    mouse = mousex.group(1)
                    day = dayx.group(1)
                    assembly_type = assembly_typex.group(1)
                    session = sessionx.group(1)
                    assembly_no = assembly_nox.group(1)
                    #print (mouse)


                    with open(fullpath, 'r') as f: 
                            assembly_strength = [i.split(" ") for i in f.read().split()]
                            seq=(mouse, '_', day, '_', assembly_type,'_ses',session, '_binnedStrength_assembly', assembly_no)
                            globals()[''.join (seq)]  = np.array(assembly_strength).astype(np.float)
                            seq2=(str(mouse),"_",str(day),"_",str(assembly_type),"_ses",str(session),\
                                  "_binnedStrength_assembly",str(assembly_no))
                            assembly_strength_name=globals()[''.join(seq2)]
                            assembly_strength_dic[''.join (seq)] = np.array(assembly_strength_name).astype(np.float)
    return(assembly_strength_dic)
                    
#Extracting assembly weights
def extract_assemblyWeights(recording_days):
    assembly_weight_dic={}
    for i in recording_days:
        source1 = '/mnfs/ca3d2/data/melgaby_analysis/assembly/'+str(i)+'/'
        for root, dirs, filenames in os.walk(source1):
            for file in filenames:
                if '2_25.APweight' in file and 'f1' not in file:
                    fullpath = os.path.join(source1, file)
                    log = open(fullpath, 'r')
                    mousex = re.search('assembly/(\w+)', source1)
                    dayx = re.search('-(\w+)/', source1)
                    assembly_typex = re.search('(\w+).APweight', fullpath)
                    assembly_nox = re.search('APweight.(\w+)', fullpath)
                    ###
                    mouse = mousex.group(1)
                    day = dayx.group(1)
                    assembly_type = assembly_typex.group(1)
                    assembly_no = assembly_nox.group(1)
                    #print (mouse)


                    with open(fullpath, 'r') as f: 
                            assembly_weights = [i.split(" ") for i in f.read().split()]
                            seq=(mouse, '_', day, '_', assembly_type,'_weights_assembly', assembly_no)
                            globals()[''.join (seq)]  = np.array(assembly_weights).astype(np.float)[1:]
                            seq2=(str(mouse),"_",str(day),"_",str(assembly_type),\
                                  "_weights_assembly",str(assembly_no))
                            assembly_weight_name=globals()[''.join(seq2)]
                            assembly_weight_dic[''.join (seq)] = np.array(assembly_weight_name).astype(np.float)

                if '0_25.APweight' in file and 'f1' not in file:
                    fullpath = os.path.join(source1, file)
                    log = open(fullpath, 'r')
                    mousex = re.search('assembly/(\w+)', source1)
                    dayx = re.search('-(\w+)/', source1)
                    assembly_typex = re.search('(\w+).APweight', fullpath)
                    assembly_nox = re.search('APweight.(\w+)', fullpath)
                    ###
                    mouse = mousex.group(1)
                    day = dayx.group(1)
                    assembly_type = assembly_typex.group(1)
                    assembly_no = assembly_nox.group(1)
                    #print (mouse)


                    with open(fullpath, 'r') as f: 
                            assembly_weights = [i.split(" ") for i in f.read().split()]
                            seq=(mouse, '_', day, '_', assembly_type,'_weights_assembly', assembly_no)
                            globals()[''.join (seq)]  = np.array(assembly_weights).astype(np.float)[1:]
                            seq2=(str(mouse),"_",str(day),"_",str(assembly_type),\
                                  "_weights_assembly",str(assembly_no))
                            assembly_weight_name=globals()[''.join(seq2)]
                            assembly_weight_dic[''.join (seq)] = np.array(assembly_weight_name).astype(np.float)

    return(assembly_weight_dic)

#Extracting lists of P1 neurons
def extract_P1neuron_list(recording_days):
    cellList_dic={}
    for i in recording_days:
        source1 = '/mnfs/ca3d2/data/melgaby_analysis/assembly/'+str(i)+'/'
        for root, dirs, filenames in os.walk(source1):
            for file in filenames:
                if '2_25_p1.cellList' in file and 'Circle' in file and '~' not in file:
                    fullpath = os.path.join(source1, file)
                    log = open(fullpath, 'r')
                    mousex = re.search('assembly/(\w+)', source1)
                    dayx = re.search('-(\w+)/', source1)

                    ###
                    mouse = mousex.group(1)
                    day = dayx.group(1)

                    #print (mouse)


                    with open(fullpath, 'r') as f: 
                            cellList = [i.split(" ") for i in f.read().split()]
                            seq=(mouse, '_', day, '_cellList')
                            globals()[''.join (seq)]  = np.concatenate(np.array(cellList).astype(np.int))
                            seq2=(str(mouse),"_",str(day),"_cellList")
                            cellList_name=globals()[''.join(seq2)]
                            cellList_dic[''.join (seq)] = np.array(cellList_name).astype(np.int)
    return(cellList_dic)




###Extracting assembly similarity scores
def extract_similarity_scores(recording_days):
    assembly_sim_dic={}
    for i in recording_days:
        source1 = '/mnfs/ca3d2/data/melgaby_analysis/assembly/results/'+str(i)+'/'
        for root, dirs, filenames in os.walk(source1):
            for file in filenames:
                if 'maxsimilarity' in file and 'Circle_2_25' in file and 'Strip_2_25' in file:
                    #print("yup")
                    fullpath = os.path.join(source1, file)
                    log = open(fullpath, 'r')
                    mousex = re.search('results/(\w+)', source1)
                    dayx = re.search('-(\w+)/', source1)
                    comparison_typex = re.search('maxsimilarity_(\w+).txt', fullpath)
                    ###
                    mouse = mousex.group(1)
                    day = dayx.group(1)
                    comparison_type = comparison_typex.group(1)

                    #print (mouse)


                    with open(fullpath, 'r') as f: 
                            assembly_sim = [i.split(" ") for i in f.read().split()]
                            seq=(mouse, '_', day, '_', comparison_type)
                            #globals()[''.join (seq)]  = np.array(assembly_sim).astype(np.float)
                            #seq2=(str(mouse),"_",str(day),"_",str(comparison_type))
                            #assembly_sim_name=globals()[''.join(seq)]
                            assembly_sim_dic[''.join (seq)] = np.array(assembly_sim).astype(np.float)
    return(assembly_sim_dic)                   


#Extracting clu files
def extract_clu(recording_days):
    clu_dic={}
    #clu files
    for i in recording_days:
        source1 = '/mnfs/ca3d2/data/melgaby_merged/'+str(i)+'/'
        for root, dirs, filenames in os.walk(source1):
            for file in filenames:
                if '.clu' in file and '_' in file and 'jj' not in file and 'clu.' not in file:
                    fullpath = os.path.join(source1, file)
                    log = open(fullpath, 'r')
                    mousex=re.search('(\w+)-', i)
                    dayx = re.search('-(\w+)/', source1)
                    sessionx = re.search('_(\w+).clu', fullpath)
                    ###
                    mouse = mousex.group(1)
                    day = dayx.group(1)
                    session = sessionx.group(1)
                    #print (mouse)


                    with open(fullpath, 'r') as f: 
                            clu = [i.split(" ") for i in f.read().split()]
                            seq=(mouse, '_', day,'_ses',session, '_clu')
                            globals()[''.join (seq)]  = np.array(clu).astype(np.int)
                            seq2=(str(mouse),"_",str(day),"_ses",str(session),\
                                  "_clu")
                            clu_name=globals()[''.join(seq2)]
                            clu_dic[''.join (seq)] = np.array(clu_name).astype(np.int)
    return(clu_dic)


#Extracting res files
def extract_res(recording_days):
    res_dic={}
    #res files
    for i in recording_days:
        source1 = '/mnfs/ca3d2/data/melgaby_merged/'+str(i)+'/'
        for root, dirs, filenames in os.walk(source1):
            for file in filenames:
                if '.res' in file and '_' in file and 'jj' not in file and 'res.' not in file and 'resofs' not in file:
                    fullpath = os.path.join(source1, file)
                    log = open(fullpath, 'r')
                    mousex=re.search('(\w+)-', i)
                    dayx = re.search('-(\w+)/', source1)
                    sessionx = re.search('_(\w+).res', fullpath)
                    ###
                    mouse = mousex.group(1)
                    day = dayx.group(1)
                    session = sessionx.group(1)
                    #print (mouse)


                    with open(fullpath, 'r') as f: 
                            res = [i.split(" ") for i in f.read().split()]
                            seq=(mouse, '_', day,'_ses',session, '_res')
                            globals()[''.join (seq)]  = np.array(res).astype(np.int)
                            seq2=(str(mouse),"_",str(day),"_ses",str(session),\
                                  "_res")
                            res_name=globals()[''.join(seq2)]
                            res_dic[''.join (seq)] = np.array(res_name).astype(np.int)
    return(res_dic)


#Extracting des files
def extract_des(recording_days):
    des_dic={}
    #des files
    for i in recording_days:
        source1 = '/mnfs/ca3d2/data/melgaby_merged/'+str(i)+'/'
        for root, dirs, filenames in os.walk(source1):
            for file in filenames:
                if '.des' in file and '_' not in file and 'jj' not in file and 'desen' not in file and 'desel' not in file\
                and 'des.' not in file and 'jnk' not in file and '~' not in file:
                    fullpath = os.path.join(source1, file)
                    log = open(fullpath, 'r')
                    mousex=re.search('(\w+)-', i)
                    dayx = re.search('-(\w+)/', source1)
                    ###
                    mouse = mousex.group(1)
                    day = dayx.group(1)
                    #print (file)


                    with open(fullpath, 'r') as f: 
                            des = [i.split(" ") for i in f.read().split()]
                            seq=(mouse, '_', day, '_des')
                            globals()[''.join (seq)]  = np.concatenate(np.array(des).astype(np.str))
                            seq2=(str(mouse),"_",str(day),"_des")
                            des_name=globals()[''.join(seq2)]
                            des_dic[''.join (seq)] = np.array(des_name).astype(np.str)
    return(des_dic)



#Extracting clu files
def extract_clu2(recording_days,source):
    clu_dic={}
    #clu files
    for i in recording_days:
        source1 = source+str(i)+'/'
        for root, dirs, filenames in os.walk(source1):
            for file in filenames:
                if '.clu' in file and '_' in file and 'jj' not in file and 'clu.' not in file:
                    fullpath = os.path.join(source1, file)
                    log = open(fullpath, 'r')
                    mousex=re.search('(\w+)-', i)
                    dayx = re.search('-(\w+)/', source1)
                    sessionx = re.search('_(\w+).clu', fullpath)
                    ###
                    mouse = mousex.group(1)
                    day = dayx.group(1)
                    session = sessionx.group(1)
                    #print (mouse)


                    with open(fullpath, 'r') as f: 
                            clu = [i.split(" ") for i in f.read().split()]
                            seq=(mouse, '_', day,'_ses',session, '_clu')
                            globals()[''.join (seq)]  = np.array(clu).astype(np.int)
                            seq2=(str(mouse),"_",str(day),"_ses",str(session),\
                                  "_clu")
                            clu_name=globals()[''.join(seq2)]
                            clu_dic[''.join (seq)] = np.array(clu_name).astype(np.int)
    return(clu_dic)

#Extracting
def extract_clu3(recording_days,source):
    clu_dic={}
    #clu files
    for i in recording_days:
        source1 = source+str(i)+'/'
        for root, dirs, filenames in os.walk(source1):
            for file in filenames:
                if '.clu.' in file and '_' in file and 'jj' not in file:
                    fullpath = os.path.join(source1, file)
                    log = open(fullpath, 'r')
                    mousex=re.search('(\w+)-', i)
                    dayx = re.search('-(\w+)/', source1)
                    sessionx = re.search('_(\w+).clu', fullpath)
                    ###
                    mouse = mousex.group(1)
                    day = dayx.group(1)
                    session = sessionx.group(1)
                    #print (mouse)


                    with open(fullpath, 'r') as f: 
                            clu = [i.split(" ") for i in f.read().split()]
                            seq=(mouse, '_', day,'_ses',session, '_clu')
                            globals()[''.join (seq)]  = np.array(clu).astype(np.int)
                            seq2=(str(mouse),"_",str(day),"_ses",str(session),\
                                  "_clu")
                            clu_name=globals()[''.join(seq2)]
                            clu_dic[''.join (seq)] = np.array(clu_name).astype(np.int)
    return(clu_dic)

#Extracting res files
def extract_res2(recording_days,source):
    res_dic={}
    #res files
    for i in recording_days:
        source1 = source+str(i)+'/'
        for root, dirs, filenames in os.walk(source1):
            for file in filenames:
                if '.res' in file and '_' in file and 'jj' not in file and 'res.' not in file and 'resofs' not in file:
                    fullpath = os.path.join(source1, file)
                    log = open(fullpath, 'r')
                    mousex=re.search('(\w+)-', i)
                    dayx = re.search('-(\w+)/', source1)
                    sessionx = re.search('_(\w+).res', fullpath)
                    ###
                    mouse = mousex.group(1)
                    day = dayx.group(1)
                    session = sessionx.group(1)
                    #print (mouse)


                    with open(fullpath, 'r') as f: 
                            res = [i.split(" ") for i in f.read().split()]
                            seq=(mouse, '_', day,'_ses',session, '_res')
                            globals()[''.join (seq)]  = np.array(res).astype(np.int)
                            seq2=(str(mouse),"_",str(day),"_ses",str(session),\
                                  "_res")
                            res_name=globals()[''.join(seq2)]
                            res_dic[''.join (seq)] = np.array(res_name).astype(np.int)
    return(res_dic)


#Extracting des files
def extract_des2(recording_days,source):
    des_dic={}
    #des files
    for i in recording_days:
        source1 = source+str(i)+'/'
        for root, dirs, filenames in os.walk(source1):
            for file in filenames:
                if '.des' in file and '_' not in file and 'jj' not in file and 'desen' not in file and 'desel' not in file\
                and 'des.' not in file and 'jnk' not in file and '~' not in file:
                    fullpath = os.path.join(source1, file)
                    log = open(fullpath, 'r')
                    mousex=re.search('(\w+)-', i)
                    dayx = re.search('-(\w+)/', source1)
                    ###
                    mouse = mousex.group(1)
                    day = dayx.group(1)
                    #print (file)


                    with open(fullpath, 'r') as f: 
                            des = [i.split(" ") for i in f.read().split()]
                            seq=(mouse, '_', day, '_des')
                            globals()[''.join (seq)]  = np.concatenate(np.array(des).astype(np.str))
                            seq2=(str(mouse),"_",str(day),"_des")
                            des_name=globals()[''.join(seq2)]
                            des_dic[''.join (seq)] = np.array(des_name).astype(np.str)
    return(des_dic)

#Extracting des files
def extract_des3(recording_days,source):
    des_dic={}
    #des files
    for i in recording_days:
        source1 = source+str(i)+'/'
        for root, dirs, filenames in os.walk(source1):
            for file in filenames:
                if '.des.' in file and '_' not in file and 'jj' not in file and 'desen' not in file and 'desel' not in file\
                and 'jnk' not in file and '~' not in file:
                    fullpath = os.path.join(source1, file)
                    log = open(fullpath, 'r')
                    mousex=re.search('(\w+)-', i)
                    dayx = re.search('-(\w+)/', source1)
                    ###
                    mouse = mousex.group(1)
                    day = dayx.group(1)
                    #print (file)


                    with open(fullpath, 'r') as f: 
                            des = [i.split(" ") for i in f.read().split()]
                            seq=(mouse, '_', day, '_des')
                            if len(des)>0:
                                globals()[''.join (seq)]  = np.concatenate(np.array(des).astype(np.str))
                            else:
                                globals()[''.join (seq)]  = []
                            seq2=(str(mouse),"_",str(day),"_des")
                            des_name=globals()[''.join(seq2)]
                            des_dic[''.join (seq)] = np.array(des_name).astype(np.str)
    return(des_dic)


#Extracting led displays for sessions A and B
def extract_AB(recording_days,file):
    sesAB_dic={}
    for x in recording_days: 
        mousex = re.search('(\w+)-', x)
        mouse = mousex.group(1)
        dayx = re.search('-(\w+)', x)
        day = dayx.group(1)
        with open(file, 'r') as f:
            desen = [i.split(" ") for i in f.read().split()] 
        ses8 = desen[17]
        ses9 = desen[19]
        #exec(str(mouse)+'_'+str(day)+'_ses8_desen = ses8')
        #exec(str(mouse)+'_'+str(day)+'_ses9_desen = ses9')
        seq2=(str(mouse),"_",str(day))
        desen_name=(ses8,ses9)
        sesAB_dic[''.join (seq2)] = desen_name
    return(sesAB_dic)


#Extracting cell List files (cell lists used to calculate assemblies) - this should be identical to P1 except if using p3 in same
#assembly or when there is an error

def extract_assemblyCellList(recording_days):
    CellList_dic={}
    #res files
    for i in recording_days:
        source1 = '/mnfs/ca3d2/data/melgaby_analysis/'+str(i)+'/'
        for root, dirs, filenames in os.walk(source1):
            for file in filenames:
                if '25.cellList' in file and 'Circle' and '~' not in file:
                    fullpath = os.path.join(source1, file)
                    log = open(fullpath, 'r')
                    mousex = re.search('merged/(\w+)', source1)
                    dayx = re.search('-(\w+)/', source1)
                    ###
                    mouse = mousex.group(1)
                    day = dayx.group(1)
                    #print (file)


                    with open(fullpath, 'r') as f: 
                            des = [i.split(" ") for i in f.read().split()]
                            seq=(mouse, '_', day, '_des')
                            globals()[''.join (seq)]  = np.concatenate(np.array(des).astype(np.str))
                            seq2=(str(mouse),"_",str(day),"_des")
                            des_name=globals()[''.join(seq2)]
                            des_dic[''.join (seq)] = np.array(des_name).astype(np.str)
    return(CellList_dic)

def extract_rrem(rec_days):
    #Extracting rrem files

    rrem_dic={}

    for i in rec_days:
        source2 = '/mnfs/ca3d2/data/melgaby_merged/'+str(i)+'/'
        for root, dirs, filenames in os.walk(source2):
            for file in filenames:
                filebsnm=str(i)+'_'
                if str(filebsnm) in file and '.rrem' in file and not '~' in file:
                    fullpath = os.path.join(source2, file)
                    log = open(fullpath, 'r')
                    mousex = re.search('merged/(\w+)', source2)
                    dayx = re.search('-(\w+)/', source2)
                    sessionx = re.search('_(\w+)\.', fullpath)
                    ###
                    mouse = mousex.group(1)
                    day = dayx.group(1)
                    session = sessionx.group(1)
                    #print(session)
                    with open(fullpath, 'r') as f:
                        rrem_x=[]
                        for x in f:
                            rrem_x.append(x.split()[0])
                        seq=(mouse, '_', day, '_rrem_start_ses', session)
                        globals()[''.join (seq)]  = np.array(rrem_x).astype(np.float)
                    with open(fullpath, 'r') as f:
                        rrem_y=[]
                        for y in f:
                            rrem_y.append(y.split()[1])
                        seq=(mouse, '_', day, '_rrem_end_ses', session)
                        globals()[''.join (seq)]  = np.array(rrem_y).astype(np.float)
                    exec(str(mouse)+'_'+str(day)+'_rrem_ses'+str(session)+' = np.column_stack(( '+str(mouse)+'_'+\
                         str(day)+'_rrem_start_ses'+str(session)+', '+str(mouse)+'_'+str(day)+'_rrem_end_ses'+\
                         str(session)+'))')
                    #creating dictionary with all pulse times for each pulse file
                    seq=(mouse, '_', day, '_rrem_ses', session)
                    exec("rrem_dic[''.join (seq)] = np.array("+str(mouse)+"_"+str(day)+"_rrem_ses"+str(session)+\
                         ").astype(np.float)")
    return(rrem_dic)




def extract_layout(rec_days):
    
    layout_dic={}
    
    for i in rec_days:
        source3 = '/mnfs/ca3d2/data/melgaby_analysis/Behaviour/'+str(i)+'/'
        #print(source3)
        for root, dirs, filenames in os.walk(source3):
            for file in filenames:
                if str(i) in file and 'layout' in file and not '~' in file:
                        fullpath = os.path.join(source3, file)
                        #print(fullpath)
                        mousex = re.search('Behaviour/(\w+)', source3)
                        dayx = re.search('-(\w+)/', source3)
                        ###
                        mouse = mousex.group(1)
                        day = dayx.group(1)

                        with open(fullpath,"r") as input:
                            with open("jjlayout.txt","wb") as output: 
                                for line in input:
                                    #string = "led"
                                    if "led" not in line:
                                        line = line.encode() ##converting file into binary
                                        output.write(line)

                        with open("jjlayout.txt", 'r') as f:
                            x_coordinates1=[]
                            for co in f:
                                x_coordinates1.append(co.split()[1])
                            seq=(mouse, '_', day, '_x_coordinates')
                            globals()[''.join (seq)]  = np.array(x_coordinates1).astype(np.float)
                        with open("jjlayout.txt", 'r') as f:
                            y_coordinates1=[]
                            for co in f:
                                y_coordinates1.append(co.split()[2])
                            seq=(mouse, '_', day, '_y_coordinates')
                            globals()[''.join (seq)]  = np.array(y_coordinates1).astype(np.float)


                        with open(fullpath, 'r') as f:
                            ledCircle_pump=[]
                            for co in f:
                                if "ledCircle" in co:
                                    ledCircle_pump.append(co.split()[1])
                            seq=(mouse, '_', day, '_ledCircle_pump')
                            globals()[''.join (seq)]  = np.array(ledCircle_pump)[0]
                        with open(fullpath, 'r') as f:
                            ledStrip_pump=[]
                            for co in f:
                                if "ledStrip" in co:
                                    ledStrip_pump.append(co.split()[1])
                            seq=(mouse, '_', day, '_ledStrip_pump')
                            globals()[''.join (seq)]  = np.array(ledStrip_pump)[0]

                        #df = pd.read


                        exec(str(mouse)+'_'+str(day)+'_layout'+' = np.column_stack(( '+str(mouse)+'_'+\
                             str(day)+'_x_coordinates'+', '+str(mouse)+'_'+str(day)+'_y_coordinates'+'))')
                        
                        seq=(mouse, '_', day, '_layout')
                        exec("layout_dic[''.join (seq)] = np.array("+str(mouse)+"_"+str(day)+"_layout).astype(np.float)")
    return(layout_dic)

def extract_layout2(rec_days,source):
    
    layout_dic={}
    
    for i in rec_days:
        source3 = source+str(i)+'/'
        #print(source3)
        for root, dirs, filenames in os.walk(source3):
            for file in filenames:
                if str(i) in file and 'layout' in file and not '~' in file:
                        fullpath = os.path.join(source3, file)
                        #print(fullpath)
                        #mousex = re.search('Behaviour/(\w+)', source3)
                        #dayx = re.search('-(\w+)/', source3)
                        mousex=re.search('(\w+)-', i)
                        dayx=re.search('-(\w+)', i)
                        ###
                        mouse = mousex.group(1)
                        day = dayx.group(1)

                        with open(fullpath,"r") as input:
                            with open("jjlayout.txt","wb") as output: 
                                for line in input:
                                    #string = "led"
                                    if "led" not in line:
                                        line = line.encode() ##converting file into binary
                                        output.write(line)

                        with open("jjlayout.txt", 'r') as f:
                            x_coordinates1=[]
                            for co in f:
                                x_coordinates1.append(co.split()[1])
                            seq=(mouse, '_', day, '_x_coordinates')
                            globals()[''.join (seq)]  = np.array(x_coordinates1).astype(np.float)
                        with open("jjlayout.txt", 'r') as f:
                            y_coordinates1=[]
                            for co in f:
                                y_coordinates1.append(co.split()[2])
                            seq=(mouse, '_', day, '_y_coordinates')
                            globals()[''.join (seq)]  = np.array(y_coordinates1).astype(np.float)


                        with open(fullpath, 'r') as f:
                            ledCircle_pump=[]
                            for co in f:
                                if "ledCircle" in co:
                                    ledCircle_pump.append(co.split()[1])
                            seq=(mouse, '_', day, '_ledCircle_pump')
                            globals()[''.join (seq)]  = np.array(ledCircle_pump)[0]
                        with open(fullpath, 'r') as f:
                            ledStrip_pump=[]
                            for co in f:
                                if "ledStrip" in co:
                                    ledStrip_pump.append(co.split()[1])
                            seq=(mouse, '_', day, '_ledStrip_pump')
                            globals()[''.join (seq)]  = np.array(ledStrip_pump)[0]

                        #df = pd.read


                        exec(str(mouse)+'_'+str(day)+'_layout'+' = np.column_stack(( '+str(mouse)+'_'+\
                             str(day)+'_x_coordinates'+', '+str(mouse)+'_'+str(day)+'_y_coordinates'+'))')
                        
                        seq=(mouse, '_', day, '_layout')
                        exec("layout_dic[''.join (seq)] = np.array("+str(mouse)+"_"+str(day)+"_layout).astype(np.float)")
    return(layout_dic)
        

#extracting Tuning measures files

def extract_Tuning_measures(all_recording_days):
    for i in all_recording_days:
        source3 = '/mnfs/ca3d2/data/melgaby_analysis/PlaceMaps/'+str(i)+'/'
        #print(source3)
        for root, dirs, filenames in os.walk(source3):
            for file in filenames:
                if str(i) in file and 'TuningMeasuresSelection' in file and not '~' in file:
                    fullpath = os.path.join(source3, file)
                    #print(fullpath)
                    mousex = re.search('PlaceMaps/(\w+)', source3)
                    dayx = re.search('-(\w+)/', source3)
                    sesx = re.search('_ses(\w+)_', fullpath)
                    ###
                    mouse = mousex.group(1)
                    day = dayx.group(1)
                    ses = sesx.group(1)

                    with open(fullpath, 'r') as f:
                        cluID=[]
                        Sparsityx=[]
                        Coherencex=[]
                        Informationx=[]
                        for co in f:
                            cluID.append(co.split()[0])
                            Sparsityx.append(co.split()[1])
                            Coherencex.append(co.split()[2])
                            Informationx.append(co.split()[3])

                        cluID=np.array(cluID).astype(np.int)

                        Sparsity=np.column_stack((cluID,Sparsityx))
                        Coherence=np.column_stack((cluID,Coherencex))
                        Information=np.column_stack((cluID,Informationx))

                        seq1=(mouse, '_', day, '_Sparsity_ses', ses)
                        pulse_dic[''.join (seq1)] = np.array(Sparsity).astype(np.float)
                        globals()[''.join (seq1)]  = np.array(Sparsity).astype(np.float)

                        seq2=(mouse, '_', day, '_Coherence_ses', ses)
                        pulse_dic[''.join (seq2)] = np.array(Coherence).astype(np.float)
                        globals()[''.join (seq2)]  = np.array(Coherence).astype(np.float)

                        seq3=(mouse, '_', day, '_SpatialInformation_ses', ses)
                        pulse_dic[''.join (seq3)] = np.array(Information).astype(np.float)
                        globals()[''.join (seq3)]  = np.array(Information).astype(np.float)
    #print(mme08_161020_Coherence_ses9)

#extracting Place Maps
def Place_Maps_predefined(all_recording_days):
    for i in all_recording_days:
        source3 = '/mnfs/ca3d2/data/melgaby_analysis/PlaceMaps/'+str(i)+'/'
        #print(source3)
        for root, dirs, filenames in os.walk(source3):
            for file in filenames:
                if str(i) in file and 'SmoothedMapClu' in file and not '~' in file and not '#' in file:
                    fullpath = os.path.join(source3, file)
                    #print(fullpath)
                    mousex = re.search('PlaceMaps/(\w+)', source3)
                    dayx = re.search('-(\w+)/', source3)
                    sesx = re.search('_ses(\w+)_', fullpath)
                    clux = re.search('MapClu(\w+)', fullpath)
                    ###
                    mouse = mousex.group(1)
                    day = dayx.group(1)
                    ses = sesx.group(1)
                    clu = clux.group(1)

                    with open(fullpath, 'r') as f:
                        Map=[]
                        for co in f:
                            Map.append(co.split())

                        seq1=(mouse, '_', day, '_SmoothenedMap_ses', ses, '_clu', clu)
                        pulse_dic[''.join (seq1)] = np.array(Map).astype(np.float)
                        globals()[''.join (seq1)]  = np.array(Map).astype(np.float)

    #print(mme08_161020_Coherence_ses9)    

#defining layout variables

    
def define_layout_variables(all_recording_days):
    for i in all_recording_days:
        mousex = re.search('(\w+)-', i)
        dayx = re.search('-(\w+)', i)
        ###
        mouse = mousex.group(1)
        day = dayx.group(1)
        #print(mouse)
        #exec('Strip_2_25_ses9_assemblyx = 0.00005*mme12_170407_Strip_2_25_ses9_assembly'+str(j)+'[:,0]')
        d_layout={"enclosure_max":0, "enclosure_min":1, "pumpA_max":2, "pumpA_min":3, "pumpB_max":4, "pumpB_min":5}
        for key, value in d_layout.items():
            exec(str(mouse)+'_'+str(day)+'_x_'+str(key)+'= '+str(mouse)+'_'+str(day)+'_x_coordinates'+'['+str(value)+']')
            exec(str(mouse)+'_'+str(day)+'_y_'+str(key)+'= '+str(mouse)+'_'+str(day)+'_y_coordinates'+'['+str(value)+']')
            
            
## Extracting pump crossings##
def extract_pump_crossing(all_Behaviour_recording_days):
    #if first column is less than max_x and greater than min_x and second column is less than max_y and greater than min_y:
    #    print column number * 512/20000 into a presence file
    #    c rossings file = two column file where values in presence file that differ from previous row by more than the sampling rate are considered entries and the last value to differ from the previous row by the sampling rate is considered an exit (similar to ttl pulses) 
    # mme08_161020_whl_ses13


    for i, value in whl_dic.items():
        mouse3x = re.search('(\w+)_', i)
        mouse3= mouse3x.group(1)
        mouse2x = re.search('(\w+)_', mouse3)
        mouse2 = mouse2x.group(1)
        mousex = re.search('(\w+)_', mouse2)
        dayx = re.search('_(\w+)_whl', i)
        sesx = re.search('_ses(\w+)', i)
        ###
        mouse = mousex.group(1)
        day = dayx.group(1)
        ses = sesx.group(1)

        recording_day=str(mouse)+'_'+str(day)
        #print(recording_day)
        #print(ses)
        if recording_day in all_Behaviour_recording_days:
            exec('x_pumpA_max='+str(mouse)+'_'+str(day)+'_x_pumpA_max')
            exec('x_pumpA_min='+str(mouse)+'_'+str(day)+'_x_pumpA_min')
            exec('y_pumpA_max='+str(mouse)+'_'+str(day)+'_y_pumpA_max')
            exec('y_pumpA_min='+str(mouse)+'_'+str(day)+'_y_pumpA_min')

            exec('x_pumpB_max='+str(mouse)+'_'+str(day)+'_x_pumpB_max')
            exec('x_pumpB_min='+str(mouse)+'_'+str(day)+'_x_pumpB_min')
            exec('y_pumpB_max='+str(mouse)+'_'+str(day)+'_y_pumpB_max')
            exec('y_pumpB_min='+str(mouse)+'_'+str(day)+'_y_pumpB_min')

            #print(y_pumpA_max)

            #presence_pumpA1=np.where((mme08_161020_whl_ses8[:,0] < mme08_161020_x_pumpA_max) & (mme08_161020_whl_ses8[:,0] > mme08_161020_x_pumpA_min) & (mme08_161020_whl_ses8[:,1] < mme08_161020_y_pumpA_max) & (mme08_161020_whl_ses8[:,1] > mme08_161020_y_pumpA_min))
            pumps = ['A', 'B']
            for pump in pumps:
                exec('presence_pump'+str(pump)+'1=np.where((value[:,0] < x_pump'+str(pump)+\
                     '_max) & (value[:,0] > x_pump'+str(pump)+'_min) & (value[:,1] < y_pump'+str(pump)+\
                     '_max) & (value[:,1] > y_pump'+str(pump)+'_min))')

            for j in presence_pumpA1:
                presence_pumpA=j*512

            if presence_pumpA.size:
                #print(presence_pumpA)
                entrances1=(presence_pumpA > (np.roll(presence_pumpA, 1)+(0.5*(20000))))
                entrances_indices=np.where(entrances1==True)

                exits1=(presence_pumpA < (np.roll(presence_pumpA, -1)-(0.5*(20000))))
                exits_indices=np.where(exits1==True)


                first_value=presence_pumpA[0]
                last_value=presence_pumpA[-1]

                entrances=np.insert(presence_pumpA[entrances_indices], 0, first_value)
                exits=np.append(presence_pumpA[exits_indices], last_value)

                exec(str(mouse)+'_'+str(day)+'_ses'+str(ses)+'_pumpA_crossings=np.column_stack((entrances, exits))')

            for j in presence_pumpB1:
                presence_pumpB=j*512

            if presence_pumpB.size:
                #print(presence_pumpA)
                entrances1=(presence_pumpB > (np.roll(presence_pumpB, 1)+(0.5*(20000))))
                entrances_indices=np.where(entrances1==True)

                exits1=(presence_pumpB < (np.roll(presence_pumpB, -1)-(0.5*(20000))))
                exits_indices=np.where(exits1==True)


                first_value=presence_pumpB[0]
                last_value=presence_pumpB[-1]

                entrances=np.insert(presence_pumpB[entrances_indices], 0, first_value)
                exits=np.append(presence_pumpB[exits_indices], last_value)

                exec(str(mouse)+'_'+str(day)+'_ses'+str(ses)+'_pumpB_crossings=np.column_stack((entrances, exits))')
                

def Rotate_xy(xy,angle):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))
    rotated=np.dot(xy,R)
    
    return(rotated)
                
class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return sp.ma.masked_array(sp.interp(value, x, y))

###takes mean of n bins to create new, more coarsely binned array
def binned_means(xx,bin_factor):
    xxdf=pd.DataFrame(xx.T)
    xxdfnew=xxdf.groupby(np.arange(len(xxdf))//bin_factor).mean()
    xxnew=xxdfnew.values.T
    return(xxnew)
    
def Place_map_minX(H,cmap,figname,interpolation):
    plt.imshow(H, cmap=cmap, interpolation=interpolation)
    #plt.colorbar()
    if figname!='none':
        plt.savefig(figname, bbox_inches='tight',pad_inches = 0)
    #plt.show()
    
def smooth_MAP_gaussian(ALL_map,sigma,occupancy):

    V=ALL_map.copy()
    V[np.isnan(ALL_map)]=0
    VV=sp.ndimage.gaussian_filter(V,sigma=sigma)

    W=0*ALL_map.copy()+1
    W[np.isnan(ALL_map)]=0
    WW=sp.ndimage.gaussian_filter(W,sigma=sigma)

    Z=VV/WW
    
    Z[occupancy==0]=np.nan
    
    return(Z)

def smooth_MAPX(image_array,smf):
    new_image = image_array.copy()
    n = 0
    average_sum = 0
    for i in range(0, len(image_array)):
        for j in range(0, len(image_array[i])):
            for k in range(-smf, smf):
                for l in range(-smf, smf):
                    if (len(image_array) > (i + k) >= 0) and (len(image_array[i]) > (j + l) >= 0) and\
                    np.isnan(image_array[i+k][j+l]) != True:
                        average_sum += image_array[i+k][j+l]
                        n += 1
            if n > 0:
                new_image[i][j] = average_sum/n#(int(round(average_sum/n)))
            average_sum = 0
            n = 0
    return(new_image)


def spatial_sparsity(occ,binned_map):
    ##see https://www.jneurosci.org/content/14/12/7347.long
    occX=np.concatenate(occ)
    mapX=np.concatenate(binned_map)

    prob_occ=occX/np.nansum(occX)

    sparsity=np.divide(np.square(np.nansum(mapX*prob_occ)),np.nansum(prob_occ*np.square(mapX)))
    return(sparsity)


def getSpikePhaseCoherence(spikePhases, minPhases=[False, 200]):
    ##From charlie
    if minPhases[0] and len(spikePhases) < minPhases[1]:
        return np.nan
    return (np.abs(np.mean(np.exp(1j*spikePhases))))

def polar_plot(x,nbins):
    from vBaseFunctions3 import hist
    count,bin_,_ = hist(x,nbins)
    plt.subplot(111, projection='polar')
    plt.bar(bin_,count,width=bin_[1]-bin_[0])
    
def polar_plot_state(xxx,ls='solid'):
    rx = list(xxx)
    theta = list(range(len(rx)))
    thetax = [2 * np.pi * (x/len(rx)) for x in theta]
    r = rx + [rx[0]]
    theta = thetax + [thetax[0]]
    
    ax = plt.subplot(111, projection='polar')
    ax.plot(theta, r,ls=ls)
    ax.set_rmax(np.max(r)+0.1*np.max(r))
    ax.grid(True)
    #ax.set_rorigin(-1)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticklabels(['A', '', 'B', '', 'C', '', 'D', ''],fontsize=20)

    #plt.show()
    
def polar_plot_state_replay(reactivated_angles_sig_sorted,mat):
    from scipy.interpolate import interp1d

    timestamps_replayx=[np.asarray(np.where(mat[ii]>0)[0]) for ii in range(len(mat))]
    timestamps_replay=np.concatenate(timestamps_replayx)

    timestamps_replay_=np.sort(timestamps_replay)

    angles_all=np.concatenate([np.repeat(reactivated_angles_sig_sorted[ii],len(timestamps_replayx[ii]))\
                               for ii in range(len(timestamps_replayx))])
    angles_all_rad=np.deg2rad(angles_all)

    angles_all_rad_=angles_all_rad[timestamps_replay.argsort()]




    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    ax.scatter(angles_all_rad, timestamps_replay,color='black')
    for xx,angle  in enumerate(angles_all_rad_[:-1]):
        curve=[angles_all_rad_[xx],angles_all_rad_[xx+1]],[timestamps_replay_[xx],timestamps_replay_[xx+1]]
        x = np.linspace( curve[0][0], curve[0][1], 500)
        y = interp1d( curve[0], curve[1])( x)
        ax.plot(x,y,color='blue')
        #ax.plot([angles_all_rad_[xx],angles_all_rad_[xx+1]],[timestamps_replay_[xx],timestamps_replay_[xx+1]], color='black')
    #ax.plot(angles_all_rad, timestamps_replay)
    ax.grid(True)
    #ax.set_rorigin(-1)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticklabels(['A', '', 'B', '', 'C', '', 'D', ''],fontsize=20)

    
def smooth_circ(xx,sigma):
    x_smoothed=gaussian_filter1d(np.hstack((xx,xx,xx)),sigma,axis=0)[len(xx):int(len(xx)*2)]
    return(x_smoothed)


def circular_angle(x,y):
    pref_diff_lin=x-y
    pref_diff_circ=(pref_diff_lin + 180) % 360 - 180
    return(pref_diff_circ)
def convert_angle(angle):
    if angle<0:
        angle_converted=360+angle
    else:
        angle_converted=angle
    return(angle_converted)

def plot_grouped_error(data,bins,groups,array_output=False):
    num_bins=len(bins)
    bin_value=np.digitize(np.argmax(groups,axis=1),bins)
    meanx=st.binned_statistic(bin_value,data,'mean',bins=np.arange(num_bins)+1)[0]
    semx=st.binned_statistic(bin_value,data,'std',bins=np.arange(num_bins)+1)[0]/np.sqrt(len(data))
    
    if array_output==True:
        return(bin_value)
    else:    
        plt.errorbar(bins[1:],meanx,semx)

def plot_ISI(xx,maxx,color):
    ALLhist=np.zeros((len(xx),maxx))
    for ii in range(len(xx)):
        xxx=xx[ii]
        xxx=xxx[np.where(xxx<maxx)]
        ALLhist[ii]=plt.hist(xxx,bins=maxx)[0]
        plt.close()
    x=np.linspace(0,maxx-1,maxx)
    noplot_timecourseB(x,ALLhist,color=color)
    
    
def equalize_rows(xxx):
    df = pd.DataFrame(xxx, dtype=float)
    return df.fillna(np.nan).values

def number_of_repeats(array):
    return(np.asarray([sum(1 for _ in group) for _, group in groupby(array)]))

def num_of_repeats2(MyList):
    my_dict = {i:list(MyList).count(i) for i in MyList}



#####################


def two_proportions_test(success_a, size_a, success_b, size_b):
    """
    A/B test for two proportions;
    given a success a trial size of group A and B compute
    its zscore and pvalue
    
    Parameters
    ----------
    success_a, success_b : int
        Number of successes in each group
        
    size_a, size_b : int
        Size, or number of observations in each group
    
    Returns
    -------
    zscore : float
        test statistic for the two proportion z-test

    pvalue : float
        p-value for the two proportion z-test
    """
    prop_a = success_a / size_a
    prop_b = success_b / size_b
    prop_pooled = (success_a + success_b) / (size_a + size_b)
    var = prop_pooled * (1 - prop_pooled) * (1 / size_a + 1 / size_b)
    zscore = np.abs(prop_b - prop_a) / np.sqrt(var)
    one_side = 1 - stats.norm(loc = 0, scale = 1).cdf(zscore)
    pvalue = one_side * 2
    return zscore, pvalue



    
def rearrange_matrix(x,indices):
    xx=x[indices]
    xxx=xx[:,indices]
    return(xxx)

def matrix_triangle(a,direction='upper',return_indices=False):
    if direction=='upper':
        indices=np.triu_indices(len(a), k = 1)
    if direction=='lower':
        indices=np.tril_indices(len(a), k = -1)
    triangle=a[indices]
    if return_indices==True:
        return(triangle,indices)
    else:
        return(triangle)

def scramble(a, axis=-1):
    """
    Return an array with the values of `a` independently shuffled along the
    given axis
    """ 
    b = a.swapaxes(axis, -1)
    n = a.shape[axis]
    idx = np.random.choice(n, n, replace=False)
    b = b[..., idx]
    return b.swapaxes(axis, -1)

def smooth_circular(x,sigma=10):
    return(gaussian_filter1d(np.hstack((x,x,x)),sigma,axis=0)[len(x):int(len(x)*2)])



def plot_spatial_maps(mouse_recday,neuron,per_state=False,save_fig=False,fignamex=None,figtype=None):
    mouse=mouse_recday[:4]

    print('')
    print('Mean Rate maps')
    ###ploting firing maps per state

    awake_sessions=np.arange(4)

    node_rate_matrices_state=[]
    for awake_session_ind in awake_sessions:
        node_rate_matrices_state.append(node_rate_matrices_dic['Per_state'][awake_session_ind][mouse_recday][neuron])


    rec_day_structure_numbers=recday_numbers_dic['structure_numbers'][mouse_recday]
    structure_nums=np.unique(rec_day_structure_numbers)
    rec_day_session_numbers=recday_numbers_dic['session_numbers'][mouse_recday]

    max_rate=np.nanmax(node_rate_matrices_state)
    min_rate=np.nanmin(node_rate_matrices_state)

    fig1, f1_axes = plt.subplots(figsize=(7.5, 7.5),ncols=len(awake_sessions), nrows=1, constrained_layout=True)  
    for awake_session_ind, timestamp in enumerate(awake_sessions):
        activity_allstates=node_rate_matrices_dic['All_states'][awake_session_ind][mouse_recday][neuron]

        structure=structure_dic[mouse]['ABCD'][rec_day_structure_numbers[awake_session_ind]]\
        [rec_day_session_numbers[awake_session_ind]]

        ax1=f1_axes[awake_session_ind]
        for state_port_ind, state_port in enumerate(states):
            node=structure[state_port_ind]-1
            ax1.text(Task_grid_plotting[node,0]-0.25, Task_grid_plotting[node,1]+0.25, state_port.lower(), fontsize=22.5)


        ax1.matshow(activity_allstates, cmap='coolwarm')
        ax1.axis('off')
    plt.axis('off')    
    
    if save_fig==True:
        plt.savefig(fignamex+figtype)  
        
    


    if per_state==True:
        ###per state plot

        print('')
        print('per state Rate maps')
        fig2, f2_axes = plt.subplots(figsize=(7.5, 7.5),ncols=len(awake_sessions), nrows=len(states),\
                                     constrained_layout=True)   
        for awake_session_ind, timestamp in enumerate(awake_sessions):   

            #print(awake_session_ind)
            for statename_ind, state in enumerate(states):
                #print(state)
                node_rate_matrix_state=node_rate_matrices_dic['Per_state'][awake_session_ind][mouse_recday][neuron]

                structure=structure_dic[mouse]['ABCD'][rec_day_structure_numbers[awake_session_ind]]\
                [rec_day_session_numbers[awake_session_ind]]

                ax=f2_axes[awake_session_ind,statename_ind]
                for state_port_ind, state_port in enumerate(states):
                    node=structure[state_port_ind]-1
                    ax.text(Task_grid_plotting[node,0]-0.25, Task_grid_plotting[node,1]+0.25,\
                            state_port.lower(), fontsize=22.5)

                ax.matshow(node_rate_matrix_state[statename_ind], cmap='coolwarm') #vmin=min_rate, vmax=max_rate
                ax.axis('off')
                #ax.savefig(str(neuron)+state+str(awake_session_ind)+'discmap.svg')
        plt.axis('off')
        if save_fig==True:
            plt.savefig(fignamex+'_perstate_'+figtype) 


def state_to_phase(angles,num_states):
    bins_per_state=int(360/num_states)
    return(np.asarray([angle-(angle//bins_per_state)*bins_per_state for angle in angles])*num_states)

def middle_value(arr):
    midpoint=len(arr)//2
    if len(arr)%2 == 0:
        middle_valuex=np.mean([arr[midpoint],arr[midpoint-1]])
    else:
        middle_valuex=arr[midpoint]
        
    return(middle_valuex)

#def max_bin_safe(xx,axisX=None): ##currently only works for 1st and 2nd dimensions
#    
#    if axisX==None:
#        xx_max=np.max(xx)
#        max_bins=np.where(xx==xx_max)[0]
#        if len(max_bins)==1:
#            max_bin=max_bins
#        else:
#            max_bin=[np.nan]
#        return(max_bin[0])
#
#    
#    else:
#        if axisX==0:
#            xx=xx.T
#        xx_max=np.max(xx,axis=1)
#
#        max_binsx=[np.where(xx[ii]==xx_max[ii])[0] for ii in range(len(xx_max))]
#        
#        max_bins=np.asarray([max_binsx[ii][0] if len(max_binsx[ii])==1 else np.nan for ii in range(len(max_binsx))])
#        return(max_bins)
    
def max_bin_safe(xx,axisX=None): ##currently only works for 1st and 2nd dimensions
    
    if axisX==None:
        xx_max=np.max(xx)
        max_bins=np.where(xx==xx_max)[0]
        if len(max_bins)==1:
            max_bin=max_bins
        else:
            max_bin=[np.nan]
        return(max_bin[0])

    
    else:
        if axisX==0:
            xx=xx.T
        xx_max=np.max(xx,axis=axisX)

        max_binsx=[np.where(xx[ii]==xx_max[ii])[0] for ii in range(len(xx_max))]
        
        max_bins=np.asarray([max_binsx[ii][0] if len(max_binsx[ii])==1 else np.nan for ii in range(len(max_binsx))])
        return(max_bins)
    

    
def random_rotation(length_array,angle_changes,sigma=10,noise_centre=5):
    rand_rotation=angle_changes[np.random.randint(0,len(angle_changes),size=length_array)]
    noise=np.random.normal(noise_centre,sigma,length_array) ##note, this is set to 5
    ##to make it equally likely to fall behind or ahead of cardinal axis when binned at 10 degree bins
    ##can just set to bin length/2
    return((rand_rotation+noise) % 360)

def non_cluster_indices(neurons_clusters):
    indices_all=[]
    for cluster_ind,cluster in enumerate(neurons_clusters):
        if cluster_ind!=(len(neurons_clusters)-1):
            subsequent_cluster_members=np.concatenate(neurons_clusters[int(cluster_ind+1):])
            indices=np.vstack([[[cluster[ii],subsequent_cluster_members[jj]] for ii in range(len(cluster))]\
                     for jj in range(len(subsequent_cluster_members))])
            indices_all.append(indices)
    indices_all=np.vstack(indices_all)
    return(indices_all)

def cumulativeDist_plot(x,y,colorx,colory,name):
    fig, ax = plt.subplots(figsize=(3,3))
    hfont = {'fontname':'Arial'}
    x = x[x>-1E38]
    x = x[x<1E38]
    values, base = np.histogram(x, bins=40)
    cumulative = np.cumsum(values)
    plt.plot(base[:-1], cumulative, c=colorx)

    y = y[y>-1E38]
    y = y[y<1E38]
    values, base = np.histogram(y, bins=40)
    cumulative = np.cumsum(values)
    plt.plot(base[:-1], cumulative, c=colory)
    
    if name != 'none':
        plt.savefig(name)
    plt.show()
    
def cumulativeDist_plot_norm(x,y,colorx,colory,binsize,name,xmin,xmax):
    fig, ax = plt.subplots(figsize=(3,3))
    hfont = {'fontname':'Arial'}
    x = x[x>-1E38]
    x = x[x<1E38]
    
    y = y[y>-1E38]
    y = y[y<1E38]
    xy=np.hstack((x,y))
    bins=np.arange(np.min(xy)-binsize,np.max(xy)+binsize,binsize)
    
    values, base = np.histogram(x, bins=bins)
    cumulative = np.cumsum(values)/len(x)
    plt.plot(base[:-1], cumulative, c=colorx)

   
    
    #bins=np.arange(np.min(y)-binsize,np.max(y)+binsize,binsize)
    values, base = np.histogram(y, bins=bins)
    cumulative = np.cumsum(values)/len(y)
    plt.plot(base[:-1], cumulative, c=colory)
    
    plt.xlim(xmin,xmax)
    if name != 'none':
        plt.savefig(name)
    plt.show()
    
def angle_to_distance(xx):
    return(np.asarray([1-math.cos(math.radians(xx[ii])) for ii in range(len(xx))]))

#def rotate(p, origin=(0, 0), degrees=0):
#    angle = np.deg2rad(degrees)
#    R = np.array([[np.cos(angle), -np.sin(angle)],
#                  [np.sin(angle),  np.cos(angle)]])
#    o = np.atleast_2d(origin)
#    p = np.atleast_2d(p)
#    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def angle_to_stateX(angle,states=np.asarray(['A','B','C','D']),measure='state'):
    n_states=len(states)
    state_length=360/n_states
    state_angle_array=positive_angle(np.asarray([(state_ind)*(state_length) for state_ind in range(n_states)]))
    
    
    state=states[np.argmin(positive_angle(circular_angle(angle,state_angle_array)))]
    if measure=='state':
        return(state)
    elif measure=='ratio':
        ratio=(positive_angle(circular_angle(angle,state_angle_array[states==state]))/state_length)[0]
        return(ratio)

def range_ratio_peaks(field_session,states,measure):
    range_ratios_allfields=[]
    peak_states_allfields=[]
    for field_ind in np.arange(len(field_session)):
        fieldx=field_session[field_ind]
        peak_states=np.asarray([angle_to_stateX(fieldx[ii],states) for ii in range(len(fieldx))])
        peak_ratios=np.asarray([angle_to_stateX(fieldx[ii],states,measure='ratio') for ii in range(len(fieldx))])

        peak_states_unique=np.unique(peak_states)

        range_ratios_field=[]
        if len(peak_states_unique)==1:
            range_ratios=[peak_ratios[0],peak_ratios[-1]]
            range_ratios_field.append(range_ratios)
        elif len(peak_states_unique)==2:
            for ii in range(len(peak_states_unique)):
                peak_ratiosX=peak_ratios[peak_states==peak_states_unique[ii]]
                range_ratiosX=[peak_ratiosX[0],peak_ratiosX[-1]]
                range_ratios_field.append(range_ratiosX)

        range_ratios_allfields.append(np.asarray(range_ratios_field))
        peak_states_allfields.append(np.asarray(peak_states_unique))
    
    if measure=='ratios':
        return(np.asarray(range_ratios_allfields))
    elif measure=='states':
        return(np.asarray(peak_states_allfields))

def equalize_rowsX(xxx):
    len_min=int(np.min([len(xxx[ii]) for ii in range(len(xxx))]))
    xx=np.asarray([xxx[ii][:len_min] for ii in range(len(xxx))])
    return(xx)

def cross_corr_fast(x,y):
    data_length = len(x)

    b = x
    a = np.zeros(data_length * 2)

    a[data_length//2:data_length//2+data_length] = y # This works for data_length being even

    # Do an array flipped convolution, which is a correlation.
    c = sp.signal.fftconvolve(b, a[::-1], mode='valid')
    return(c/100)

from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris



def plot_dendrogram(model, return_linkage=False, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    if return_linkage==True:
        return(linkage_matrix)
    
def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def rank_repeat(a):
    arr=np.zeros(len(a))
    for n in np.unique(a):
        count=0
        for ii in range(len(a)):
            if a[ii]==n:
                arr[ii]=count
                count+=1

    arr=arr.astype(int)
    return(arr)

def number_of_repeats(array):
    return(np.asarray([sum(1 for _ in group) for _, group in groupby(array)]))

def number_of_repeats_ordered(array):
    return(np.asarray([sum(1 for _ in group) for _, group in groupby(array)]))

def number_of_repeats_ALL(array):
    unique_rows=np.unique(array,axis=0)
    return(np.asarray([sum((array == unique_rows[ii]).all(1)) for ii in range(len(unique_rows))]))

###counts num of repeats for each stretch of numbers
def rank_repeat2(a):
    num_repeats=number_of_repeats(a)
    arr=[]
    for n_ind, n in enumerate(unique_adjacent(a)):
        count=0
        indices=np.arange(num_repeats[n_ind])
        arr.append(indices)
    arr=np.concatenate(arr)
    arr=arr.astype(int)
    return(arr)

def edge_node_fill(edge_mat,node_mat):
    new_mat=np.copy(edge_mat)
    for ii in [0,2,4]:
        new_mat[ii,0]=node_mat[int(ii/2),0]
        new_mat[ii,2]=node_mat[int(ii/2),1]
        new_mat[ii,4]=node_mat[int(ii/2),2]
        
    return(new_mat)


def concatenate_states(xx):
    xx_concatenated=np.column_stack((xx[0][:len(xx[-1])],xx[1][:len(xx[-1])],xx[2][:len(xx[-1])],xx[3][:len(xx[-1])]))
    return(xx_concatenated)










##Defining Task grid
from scipy.spatial import distance_matrix
from itertools import product
x=(0,1,2)
Task_grid=np.asarray(list(product(x, x)))
Task_grid_plotting=np.column_stack((Task_grid[:,1],Task_grid[:,0]))
Task_grid_plotting2=[]
for yy in np.arange(3):
    y=int(yy*2)
    for xx in np.arange(3):
        x=int(xx*2)    
        Task_grid_plotting2.append([x,y])
Task_grid_plotting2=np.asarray(Task_grid_plotting2)    
Task_grid2=np.column_stack((Task_grid_plotting2[:,1],Task_grid_plotting2[:,0]))

Edge_grid=np.asarray([[1,2],[2,3],[1,4],[2,5],[3,6],[4,5],[5,6],[4,7],[5,8],[6,9],[7,8],[8,9]]) ###
Edge_grid_=Edge_grid-1
Edge_grid_coord_x=[Task_grid[Edge_grid_[ii][0]][0]+Task_grid[Edge_grid_[ii][1]][0] for ii in range(len(Edge_grid_))]
Edge_grid_coord_y=rank_repeat(Edge_grid_coord_x)
Edge_grid_coord=np.column_stack((Edge_grid_coord_x,Edge_grid_coord_y))
Edge_grid_coord2=np.asarray([[0,1],[0,3],[1,0],[1,2],[1,4],[2,1],[2,3],[3,0],[3,2],[3,4],[4,1],[4,3]])

direction_dic={'N':[1,0],'S':[-1,0],'W':[0,1],'E':[0,-1]}
direction_dic_plotting={'N': [0, -1], 'S': [0, 1], 'W': [-1, 0], 'E': [1, 0]}

##Defining state-action pairs (for policy calculation)
#node_one_step_coord=np.asarray([np.asarray(remove_empty([Task_grid[jj]-Task_grid[ii]\
#                                                    if np.sum(abs(Task_grid[jj]-Task_grid[ii]))==1\
#                                                    else [] for ii in range(len(Task_grid))]))\
#                           for jj in range(len(Task_grid))])

#node_states=np.asarray([str(ii+1)+'_'+list(direction_dic.keys())[list(direction_dic.values()).\
#                           index(list(node_one_step_coord[ii][jj]))] for ii in range(len(node_one_step_coord))\
#for jj in range(len(node_one_step_coord[ii]))])#


#edge_one_step_coord=np.asarray([np.asarray(remove_empty([Edge_grid_coord2[jj]-Task_grid2[ii]\
#                                                    if np.sum(abs(Edge_grid_coord2[jj]-Task_grid2[ii]))==1\
#                                                    else [] for ii in range(len(Task_grid2))]))\
#                           for jj in range(len(Edge_grid_coord2))])

#edge_states=np.asarray([str(ii+10)+'_'+list(direction_dic.keys())[list(direction_dic.values()).\
#                           index(list(edge_one_step_coord[ii][jj]))] for ii in range(len(edge_one_step_coord))\
#for jj in range(len(edge_one_step_coord[ii]))])#

#State_action_grid=np.concatenate((node_states,edge_states))



########



direction_dic={'N':[1,0],'S':[-1,0],'W':[0,1],'E':[0,-1]}
def find_direction(start_,end_,direction_dicX=direction_dic,node_grid=Task_grid2,edge_grid=Edge_grid_coord2,\
                  node_grid_onlynodes=Task_grid):
    if np.logical_or(np.isnan(start_),np.isnan(end_))==True:
        return(np.nan)
    
    start_=int(start_)
    end_=int(end_)
    Task_grid_start=node_grid
    Task_grid_end=node_grid
    if start_<=9 and end_<=9:
        Task_grid_start=node_grid_onlynodes
        Task_grid_end=node_grid_onlynodes
    
    if start_>9:
        start_=int(start_-9)
        Task_grid_start=edge_grid
    if end_>9:
        end_=int(end_-9)
        Task_grid_end=edge_grid

    start=start_-1
    end=end_-1
    try:
        if start>=0 and end>=0:
            direction=list(direction_dicX.keys())[list(direction_dicX.values()).\
                                                  index(list(Task_grid_start[start]-Task_grid_end[end]))]
        else:
            direction=np.nan
    except:
        direction=np.nan
    return(direction)

def predict_task_map(occupancy_state_map_,node_edge_mat,nodes=Task_grid2,edges=Edge_grid_coord2):
    predicted_task_map=np.asarray([node_edge_mat[nodes[node_edge-1,0],nodes[node_edge-1,1]]\
                                   if node_edge<10 else node_edge_mat[edges[node_edge-10,0],\
                                                                      edges[node_edge-10,1]]\
                                   for node_edge in occupancy_state_map_])
    return(predicted_task_map)


def predict_task_map_policy(policy_all_,mean_policy_FR):
    predicted_task_map=np.asarray([mean_policy_FR[int(state_action)] if np.isnan(state_action)==False
                                   else np.nan for state_action in policy_all_])
    return(predicted_task_map)

def create_binary(a):
    ax=np.copy(a)
    a_values=np.unique(ax)

    try:
        len(a_values)==2
    except ValueError:
        print ("Not a valid Sync file")
    else:
        ax[a==np.max(a)]=1
        ax[a==np.min(a)]=0
    return(ax)

def unique_adjacent(a):
    return(np.asarray([k for k,g in groupby(a)]))

def num_routes(start,end):
    num_steps=int(distance_mat[start][end])
    node=[start]
    nodes=[]
    for num_step in range(num_steps):
        next_nodes=[]
        for ii in node:
            next_nodes.append(np.where(distance_mat[ii]==1)[0])
        node=np.concatenate(next_nodes)
        nodes.append(node)
    prob_chance=len(np.where(node==end)[0])/len(node)
    return(prob_chance,len(node),nodes)

def std_complex2(x):
    stds=[]
    for i in x:
        stdsx=np.nanstd(i)
        stds.append(stdsx)
    stds=np.asarray(stds)
    return(stds)
def mean_std(stds):
    std=np.sqrt(np.sum(np.asarray([(len(stds)-1)*stds[ii]**2 for ii in range(len(stds))]))/(2*(len(stds)-1)))
    return(std)

def positive_angle(xx):
    xxx=np.asarray([int(xx[ii]+360) if xx[ii]<0 else int(xx[ii]) for ii in range(len(xx))])
    xxx[xxx==360]=0
    return(xxx)
def circular_angle(x,y):
    pref_diff_lin=x-y
    pref_diff_circ=(pref_diff_lin + 180) % 360 - 180
    return(pref_diff_circ)


def polar_plot_stateX(meanx,upperx,lowerx,color='black',labels='states',plot_type='line',Marker=False,\
                      fields_booleanx=None):
    rx = list(meanx)
    theta = list(range(len(rx)))
    thetax = [2 * np.pi * (x/len(rx)) for x in theta]
    r = rx + [rx[0]]
    theta = thetax + [thetax[0]]
    
    if Marker==True:
        fields_booleanx=fields_booleanx*(np.max(upperx)+0.1*np.max(upperx))
        fields_boolean=list(fields_booleanx)+[list(fields_booleanx)[0]]

    upper=list(upperx)+[list(upperx)[0]]
    lower=list(lowerx)+[list(lowerx)[0]]
    
    ax = plt.subplot(111, projection='polar')
    
    if plot_type=='line':
        ax.plot(theta, r,color=color)
        ax.fill_between(theta, upper, lower, alpha=0.2,color=color)
        ax.set_rmax(np.max(upper)+0.01*np.max(upper))
        if Marker==True:
            ax.plot(theta, fields_boolean,color='black',linestyle='None',marker='.')

    elif plot_type=='bar':
        ax.bar(theta,r,width=5/len(r),color=color)
    elif plot_type=='marker':
        ax.plot(theta, r,color=color)
        
    
    ax.grid(True)
    #ax.set_rorigin(-1)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    if labels=='states':
        ax.set_xticklabels(['A', '', 'B', '', 'C', '', 'D', ''],fontsize=20)
    elif labels == 'angles':
        ax.set_xticklabels(['0', '', '90', '', '180', '', '270', ''],fontsize=20)

    #plt.show()


                
def polar_plot_stateX2(meanx,upperx,lowerx,ax,repeated,color='black',labels='states',plot_type='line',Marker=False,\
                      fields_booleanx=[], structure_abstract='ABCD',fontsize=20,set_max=False,max_val=1):
    rx = list(meanx)
    theta = list(range(len(rx)))
    thetax = [2 * np.pi * (x/len(rx)) for x in theta]
    r = rx + [rx[0]]
    theta = thetax + [thetax[0]]
    
    #ax=plt.subplot(111, projection='polar')
    
    if Marker==True:
        fields_booleanx=fields_booleanx*(np.max(upperx)+0.1*np.max(upperx))
        fields_boolean=list(fields_booleanx)+[list(fields_booleanx)[0]]

    upper=list(upperx)+[list(upperx)[0]]
    lower=list(lowerx)+[list(lowerx)[0]]
    
    if plot_type=='line':
        ax.plot(theta, r,color=color)
        ax.fill_between(theta, upper, lower, alpha=0.2,color=color)
        if set_max==False:
            ax.set_rmax(np.max(upper)+0.01*np.max(upper))
        else:
            ax.set_rmax(max_val)
            
        if Marker==True:
            ax.plot(theta, fields_boolean,color='black',linestyle='None',marker='.')

    elif plot_type=='bar':
        ax.bar(theta,r,width=5/len(r),color=color)
    elif plot_type=='marker':
        ax.plot(theta, r,color=color)
        
    
    ax.grid(True)
    #ax.set_rorigin(-1)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    if labels=='states':
        if structure_abstract=='ABCD':
            ax.set_xticklabels(['A', '', 'B', '', 'C', '', 'D', ''],fontsize=fontsize)
        elif structure_abstract=='AB' and repeated==False:
            ax.set_xticklabels(['A', '', '', '', 'B', '', '', ''],fontsize=fontsize)
        elif structure_abstract=='AB' and repeated==True:
            ax.set_xticklabels(['A', '', 'B', '', 'A', '', 'B', ''],fontsize=fontsize)
    elif labels == 'angles':
        ax.set_xticklabels(['0', '', '90', '', '180', '', '270', ''],fontsize=fontsize)
        
        
    
def arrange_plot_statecells_persession(mouse_recday,neuron,awake_sessions,sessions_included=None\
                                       ,fignamex=False,sigma=10,\
                                       save=False,plot=False,figtype='.svg',array_output=False, Marker=False,\
                                       fields_booleanx=[],measure_type='mean', abstract_structures=[],\
                                      repeated=False):
    
    colors=np.repeat('blue',len(awake_sessions))
    plot_boolean=np.repeat(False,len(awake_sessions))
    plot_boolean[sessions_included]=True
    
    standardized_FR_alltrials=[]
    standardized_FR_smoothed_all=[]
    standardized_FR_smoothed_upper_all=[]
    standardized_FR_smoothed_lower_all=[]
    fig= plt.figure(figsize=plt.figaspect(1)*4.5)
    fig.tight_layout()
    for awake_session_ind, timestamp in enumerate(awake_sessions):
        structure_abstract=abstract_structures[awake_session_ind]
        if measure_type=='mean':
            xx=np.asarray(standardized_spike_events_dic[awake_session_ind][mouse_recday][neuron])
            if structure_abstract=='AB' and repeated==False:
                xx=np.asarray(standardized_spike_events_dic['AB_nonrepeated'][awake_session_ind][mouse_recday][neuron])

        else:
            xx=np.asarray(standardized_spike_events_dic[measure_type][awake_session_ind][mouse_recday][neuron])
        
        if len(np.shape(xx))==0:
            print('Empty: Possibly No trials completed')
            continue
        xxx=np.asarray([xx[ii][:len(xx[-1])] for ii in range(len(xx))])
        standardized_FR_alltrials.append(xxx)
        standardized_FR=np.hstack([np.nanmean(xxx[ii],axis=0) for ii in range(len(xxx))])*\
        behaviour_oversampling_factor*behaviour_rate/tracking_oversampling_factor
        #standardized_FR=np.concatenate(standardized_spike_events_dic[awake_session_ind][mouse_recday][neuron])\

        standardized_FR_smoothed=gaussian_filter1d(np.hstack((standardized_FR,standardized_FR,standardized_FR))\
                                                   ,sigma,axis=0)[len(standardized_FR):int(len(standardized_FR)*2)]

        
        standardized_FR_smoothed_all.append(standardized_FR_smoothed)


        standardized_FR_sem=np.hstack([st.sem(xxx[ii],axis=0) for ii in range(len(xxx))])*\
        behaviour_oversampling_factor*behaviour_rate/tracking_oversampling_factor
        standardized_FR_upper=standardized_FR+standardized_FR_sem
        standardized_FR_lower=standardized_FR-standardized_FR_sem
        standardized_FR_smoothed_upper=gaussian_filter1d(np.hstack((standardized_FR_upper,standardized_FR_upper\
                                                                    ,standardized_FR_upper))\
                                                         ,sigma,axis=0)[len(standardized_FR_upper):\
                                                                        int(len(standardized_FR_upper)*2)]
        standardized_FR_smoothed_upper_all.append(standardized_FR_smoothed_upper)

        standardized_FR_smoothed_lower=gaussian_filter1d(np.hstack((standardized_FR_lower,standardized_FR_lower\
                                                                    ,standardized_FR_lower))\
                                                         ,sigma,axis=0)[len(standardized_FR_lower):\
                                                                        int(len(standardized_FR_lower)*2)]
        standardized_FR_smoothed_lower_all.append(standardized_FR_smoothed_lower)
        
        color=colors[awake_session_ind]
        
        ax = fig.add_subplot(1, len(awake_sessions), awake_session_ind+1, projection='polar')
        if len(fields_booleanx)>0:
            polar_plot_stateX2(standardized_FR_smoothed,standardized_FR_smoothed_upper,standardized_FR_smoothed_lower,\
                              ax,color=color, Marker=Marker,fields_booleanx=fields_booleanx[awake_session_ind],\
                             structure_abstract=structure_abstract,repeated=repeated)
        else:
            polar_plot_stateX2(standardized_FR_smoothed,standardized_FR_smoothed_upper,standardized_FR_smoothed_lower,\
                              ax,color=color, Marker=False,structure_abstract=structure_abstract,repeated=repeated)
    plt.margins(0,0)
    plt.tight_layout()
    if save==True:
        plt.savefig(fignamex+str(awake_session_ind)+figtype)
    if plot==True & plot_boolean[awake_session_ind]==True:
        plt.show()
    else:
        plt.close() 

    if array_output==True:
        return(standardized_FR_smoothed_all,standardized_FR_smoothed_upper_all,standardized_FR_smoothed_lower_all)
    
def arrange_plot_statecells(mouse_recday,neuron,awake_sessions,rec_day_structure_numbers,structure_nums,fignamex=False,\
                            sigma=10,colors=['blue','blue'],save=False,plot=False,figtype='.svg',array_output=False,\
                           measure_type='mean'):
    standardized_FR_alltrials=[]
    standardized_FR_smoothed_all=[]
    standardized_FR_smoothed_upper_all=[]
    standardized_FR_smoothed_lower_all=[]
    for awake_session_ind, timestamp in enumerate(awake_sessions):
        if measure_type=='mean':
            xx=np.asarray(standardized_spike_events_dic[awake_session_ind][mouse_recday][neuron])
        else:
            xx=np.asarray(standardized_spike_events_dic[measure_type][awake_session_ind][mouse_recday][neuron])
        xxx=np.asarray([xx[ii][:len(xx[-1])] for ii in range(len(xx))])
        standardized_FR_alltrials.append(xxx)
        standardized_FR=np.hstack([np.nanmean(xxx[ii],axis=0) for ii in range(len(xxx))])*\
        behaviour_oversampling_factor*behaviour_rate/tracking_oversampling_factor
        #standardized_FR=np.concatenate(standardized_spike_events_dic[awake_session_ind][mouse_recday][neuron])\

        standardized_FR_smoothed=gaussian_filter1d(np.hstack((standardized_FR,standardized_FR,standardized_FR))\
                                                   ,sigma,axis=0)[len(standardized_FR):int(len(standardized_FR)*2)]
        standardized_FR_smoothed_all.append(standardized_FR_smoothed)


        standardized_FR_sem=np.hstack([st.sem(xxx[ii],axis=0) for ii in range(len(xxx))])*\
        behaviour_oversampling_factor*behaviour_rate/tracking_oversampling_factor
        standardized_FR_upper=standardized_FR+standardized_FR_sem
        standardized_FR_lower=standardized_FR-standardized_FR_sem
        standardized_FR_smoothed_upper=gaussian_filter1d(np.hstack((standardized_FR_upper,standardized_FR_upper\
                                                                    ,standardized_FR_upper))\
                                                         ,sigma,axis=0)[len(standardized_FR_upper):\
                                                                        int(len(standardized_FR_upper)*2)]
        standardized_FR_smoothed_upper_all.append(standardized_FR_smoothed_upper)

        standardized_FR_smoothed_lower=gaussian_filter1d(np.hstack((standardized_FR_lower,standardized_FR_lower\
                                                                    ,standardized_FR_lower))\
                                                         ,sigma,axis=0)[len(standardized_FR_lower):\
                                                                        int(len(standardized_FR_lower)*2)]
        standardized_FR_smoothed_lower_all.append(standardized_FR_smoothed_lower)
        
    if array_output==True:
        return(standardized_FR_smoothed_all,standardized_FR_smoothed_upper_all,standardized_FR_smoothed_lower_all)

    standardized_FR_alltrials_structures=np.hstack((standardized_FR_alltrials[0],standardized_FR_alltrials[1])),\
    np.hstack((standardized_FR_alltrials[2],standardized_FR_alltrials[3]))

    standardized_FR_smoothed_all=np.asarray(standardized_FR_smoothed_all)
    for structure_num_ind, structure_num in enumerate(structure_nums):
        standardized_FR_smoothed_structure=\
        np.nanmean(standardized_FR_smoothed_all[np.where(rec_day_structure_numbers==structure_num)[0]],axis=0)

        xxx=standardized_FR_alltrials_structures[structure_num_ind]

        standardized_FR_structure=np.hstack([np.nanmean(xxx[ii],axis=0) for ii in range(len(xxx))])*\
        behaviour_oversampling_factor*behaviour_rate/tracking_oversampling_factor
        FR_smoothed_structure=gaussian_filter1d(np.hstack((standardized_FR_structure,\
                                                                        standardized_FR_structure,\
                                                                        standardized_FR_structure))\
                                                   ,sigma,axis=0)[len(standardized_FR_structure):\
                                                               int(len(standardized_FR_structure)*2)]

        standardized_FR_sem=np.hstack([st.sem(xxx[ii],axis=0) for ii in range(len(xxx))])*\
        behaviour_oversampling_factor*behaviour_rate/tracking_oversampling_factor

        standardized_FR_structure_upper=standardized_FR_structure+standardized_FR_sem
        standardized_FR_structure_lower=standardized_FR_structure-standardized_FR_sem
        FR_smoothed_structure_upper=gaussian_filter1d(np.hstack((standardized_FR_structure_upper,\
                                                                    standardized_FR_structure_upper\
                                                                    ,standardized_FR_structure_upper))\
                                                         ,sigma,axis=0)[len(standardized_FR_structure_upper):\
                                                                        int(len(standardized_FR_structure_upper)*2)]

        FR_smoothed_structure_lower=gaussian_filter1d(np.hstack((standardized_FR_structure_lower\
                                                                    ,standardized_FR_structure_lower\
                                                                    ,standardized_FR_structure_lower))\
                                                         ,sigma,axis=0)[len(standardized_FR_structure_lower):\
                                                                        int(len(standardized_FR_structure_lower)*2)]

        
        color=colors[structure_num_ind]
        polar_plot_stateX(FR_smoothed_structure,FR_smoothed_structure_upper,FR_smoothed_structure_lower,color=color)
        if save==True:
            plt.savefig(fignamex+str(structure_num)+figtype)
        if plot==True:
            plt.show()
        else:
            plt.close()

            
def arrange_plot_statecells_persessionX(mouse_recday,neuron,awake_sessions,Activity_dicX,sessions_included=None\
                                       ,fignamex=False,sigma=10,\
                                       save=False,plot=False,figtype='.svg', Marker=False,\
                                       fields_booleanx=[],measure_type='mean', abstract_structures=[],\
                                      repeated=False):
    
    colors=np.repeat('blue',len(awake_sessions))
    plot_boolean=np.repeat(False,len(awake_sessions))
    plot_boolean[sessions_included]=True
    

    fig= plt.figure(figsize=plt.figaspect(1)*4.5)
    fig.tight_layout()
    for awake_session_ind, timestamp in enumerate(awake_sessions):
        structure_abstract=abstract_structures[awake_session_ind]
        standardized_FR_smoothed=Activity_dicX['Mean'][mouse_recday][neuron][awake_session_ind]
        if len(standardized_FR_smoothed)==0:
            print('Empty: Possibly No trials completed')
            continue
        standardized_FR_sem=Activity_dicX['SEM'][mouse_recday][neuron][awake_session_ind]
        
        standardized_FR_smoothed_upper=standardized_FR_smoothed+standardized_FR_sem
        standardized_FR_smoothed_lower=standardized_FR_smoothed-standardized_FR_sem
       
        
        color=colors[awake_session_ind]
        
        ax = fig.add_subplot(1, len(awake_sessions), awake_session_ind+1, projection='polar')
        if len(fields_booleanx)>0:
            polar_plot_stateX2(standardized_FR_smoothed,standardized_FR_smoothed_upper,standardized_FR_smoothed_lower,\
                              ax,color=color, Marker=Marker,fields_booleanx=fields_booleanx[awake_session_ind],\
                             structure_abstract=structure_abstract,repeated=repeated)
        else:
            polar_plot_stateX2(standardized_FR_smoothed,standardized_FR_smoothed_upper,standardized_FR_smoothed_lower,\
                              ax,color=color, Marker=False,structure_abstract=structure_abstract,repeated=repeated)
    plt.margins(0,0)
    plt.tight_layout()
    if save==True:
        plt.savefig(fignamex+str(awake_session_ind)+figtype)
    if plot==True & plot_boolean[awake_session_ind]==True:
        plt.show()
    else:
        plt.close() 
        
def arrange_plot_statecells_persessionX2(mouse_recday,neuron,Data_folder,sessions_included=None\
                                       ,fignamex=False,sigma=10,\
                                       save=False,plot=False,figtype='.svg', Marker=False,\
                                       fields_booleanx=[],measure_type='mean', abstract_structures=[],\
                                      repeated=False,behaviour_oversampling_factor=3,behaviour_rate=1000,\
                                       tracking_oversampling_factor=50):

    awake_sessions_behaviour= np.load(Intermediate_object_folder_dropbox+'awake_session_behaviour_'+mouse_recday+'.npy')
    awake_sessions=np.load(Intermediate_object_folder_dropbox+'awake_session_'+mouse_recday+'.npy')
    
    colors=np.repeat('blue',len(awake_sessions_behaviour))
    plot_boolean=np.repeat(False,len(awake_sessions_behaviour))
    plot_boolean[sessions_included]=True
    
    
    
    num_trials_day=np.load(Intermediate_object_folder+'Num_trials_'+mouse_recday+'.npy')

    fig= plt.figure(figsize=plt.figaspect(1)*4.5)
    fig.tight_layout()
    for awake_session_ind, timestamp in enumerate(awake_sessions_behaviour):
        structure_abstract=abstract_structures[awake_session_ind]
        
        if num_trials_day[awake_session_ind]<2:
            print('Not enough trials session'+str(awake_session_ind))
            continue
        if timestamp not in awake_sessions:
            print('Ephys not used for session'+str(awake_session_ind))
            continue
            
            
        try:
            norm_activity_all=np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(awake_session_ind)+'.npy')
        except:
            print('No file found session'+str(awake_session_ind))
            continue
        
        norm_activity_neuron=norm_activity_all[neuron]
        
        xxx=np.asarray(norm_activity_neuron).T
        standardized_FR=np.hstack([np.nanmean(xxx[ii],axis=0) for ii in range(len(xxx))])*\
        behaviour_oversampling_factor*behaviour_rate/tracking_oversampling_factor
        standardized_FR_sem=np.hstack([st.sem(xxx[ii],axis=0,nan_policy='omit') for ii in range(len(xxx))])*\
        behaviour_oversampling_factor*behaviour_rate/tracking_oversampling_factor
        standardized_FR_smoothed=smooth_circular(standardized_FR,sigma=sigma)            
        standardized_FR_sem_smoothed=smooth_circular(standardized_FR_sem,sigma=sigma)                    

        
        standardized_FR_smoothed_upper=standardized_FR_smoothed+standardized_FR_sem_smoothed
        standardized_FR_smoothed_lower=standardized_FR_smoothed-standardized_FR_sem_smoothed
       
        
        color=colors[awake_session_ind]
        
        ax = fig.add_subplot(1, len(awake_sessions_behaviour), awake_session_ind+1, projection='polar')
        if len(fields_booleanx)>0:
            polar_plot_stateX2(standardized_FR_smoothed,standardized_FR_smoothed_upper,standardized_FR_smoothed_lower,\
                              ax,color=color, Marker=Marker,fields_booleanx=fields_booleanx[awake_session_ind],\
                             structure_abstract=structure_abstract,repeated=repeated)
        else:
            polar_plot_stateX2(standardized_FR_smoothed,standardized_FR_smoothed_upper,standardized_FR_smoothed_lower,\
                              ax,color=color, Marker=False,structure_abstract=structure_abstract,repeated=repeated)
    plt.margins(0,0)
    #plt.tight_layout()
    if save==True:
        plt.savefig(fignamex+str(awake_session_ind)+figtype)
    if plot==True & plot_boolean[awake_session_ind]==True:
        plt.show()
    else:
        plt.close() 

def plot_spatial_mapsX(mouse_recday,neuron,sessions_used, plot_edge=True, per_state=False,save_fig=False,\
                       fignamex=None,figtype=None,sessions_custom=False):
    mouse=mouse_recday[:4]
    
    Num_trials_completed_=dict_to_array(Num_trials_dic2[mouse_recday])
    All_sessions=session_dic['All'][mouse_recday]  
    if sessions_custom==False:
        awake_sessions=session_dic['awake'][mouse_recday][Num_trials_completed_>0]
        awake_ses_inds=np.arange(awake_sessions)
    else:
        awake_ses_inds=sessions_used
    rec_day_structure_numbers=recday_numbers_dic['structure_numbers'][mouse_recday]
    rec_day_session_numbers=recday_numbers_dic['session_numbers'][mouse_recday]
    structure_nums=np.unique(rec_day_structure_numbers)

    print('')
    print('Mean Rate maps')
    ###ploting firing maps per state

    fig1, f1_axes = plt.subplots(figsize=(7.5, 7.5),ncols=len(awake_ses_inds), constrained_layout=True)

    node_rate_matrices=[]
    for awake_session_ind in awake_ses_inds:
        node_rate_matrices.append(node_rate_matrices_dic['All_states'][awake_session_ind][mouse_recday][neuron])

    #max_rate=np.nanmax(node_rate_matrices)
    #min_rate=np.nanmin(node_rate_matrices)
    for awake_ses_ind_ind, awake_session_ind in enumerate(awake_ses_inds):
        node_rate_mat=node_rate_matrices_dic['All_states'][awake_session_ind][mouse_recday][neuron]
        edge_rate_mat=edge_rate_matrices_dic['All_states'][awake_session_ind][mouse_recday][neuron]

        node_edge_mat=edge_node_fill(edge_rate_mat,node_rate_mat)
        
        if plot_edge==True:
            mat_used=node_edge_mat
            gridX=Task_grid_plotting2
        else:
            mat_used=node_rate_mat
            gridX=Task_grid_plotting
            
        arrow_length=0.2
        prop_scaling=0.3
        adjustment=0.25
            

        max_rate=np.nanmax(mat_used)
        min_rate=np.nanmin(mat_used)
        #exec('node_rate_matrix'+str(awake_session_ind)+'=node_rate_matrix')
        ax=f1_axes[awake_ses_ind_ind]

        structure=structure_dic[mouse]['ABCD'][rec_day_structure_numbers[awake_session_ind]]\
        [rec_day_session_numbers[awake_session_ind]]
        for state_port_ind, state_port in enumerate(states):
            node=structure[state_port_ind]-1
            ax.text(gridX[node,0]-adjustment, gridX[node,1]+adjustment,\
                    state_port.lower(), fontsize=22.5)
            
            
        ###Policy
        directions=Policy_dic['Mean'][mouse_recday][awake_session_ind]
        for node in np.arange(9):
            for dir_ind, (direction, coords_) in enumerate(direction_dic_plotting.items()):
                prop_=len(np.where(directions[node]==direction)[0])/len(directions[node])
                if prop_>0:
                    ax.arrow(gridX[node,0], gridX[node,1],\
                              coords_[0]*arrow_length,coords_[1]*arrow_length,width=prop_scaling*prop_/2,\
                              head_width=prop_scaling*prop_,color='white')
        ax.axis('off')
        ax.matshow(mat_used,vmin=min_rate, vmax=max_rate, cmap='coolwarm')
    plt.axis('off')   
    if save_fig==True:
        plt.savefig(fignamex+figtype, bbox_inches = 'tight', pad_inches = 0)  
    plt.show()


    if per_state==True:
        ###per state plot

        print('')
        print('per state Rate maps')
        fig2, f2_axes = plt.subplots(figsize=(7.5, 7.5),ncols=len(awake_ses_inds), nrows=len(states),\
                                     constrained_layout=True)   
        for awake_ses_ind_ind, awake_session_ind in enumerate(awake_ses_inds):   

            #print(awake_session_ind)
            for statename_ind, state in enumerate(states):
                #print(state)
                node_rate_matrix_state=node_rate_matrices_dic['Per_state'][awake_session_ind][mouse_recday][neuron]\
                [statename_ind]
                edge_rate_matrix_state=edge_rate_matrices_dic['Per_state'][awake_session_ind][mouse_recday][neuron]\
                [statename_ind]
                node_edge_mat_state=edge_node_fill(edge_rate_matrix_state,\
                                                       node_rate_matrix_state)
                
                if plot_edge==True:
                    mat_used=node_edge_mat_state
                    gridX=Task_grid_plotting2
                else:
                    mat_used=node_rate_matrix_state[state_port_ind]
                    gridX=Task_grid_plotting

                structure=structure_dic[mouse]['ABCD'][rec_day_structure_numbers[awake_session_ind]]\
                [rec_day_session_numbers[awake_session_ind]]
                
                
                ax=f2_axes[awake_ses_ind_ind,statename_ind]
                for state_port_ind, state_port in enumerate(states):
                    node=structure[state_port_ind]-1
                    ax.text(gridX[node,0]-0.25, gridX[node,1]+0.25,\
                            state_port.lower(), fontsize=22.5)

                ax.matshow(mat_used, cmap='coolwarm') #vmin=min_rate, vmax=max_rate
                ax.axis('off')
                #ax.savefig(str(neuron)+state+str(awake_session_ind)+'discmap.svg')
        plt.axis('off')
        if save_fig==True:
            plt.savefig(fignamex+'_perstate_'+figtype, bbox_inches = 'tight', pad_inches = 0)
            
def data_matrix(data, concatenate=False):
    data_mat=np.asarray([data[ii][:len(data[-1])] for ii in range (len(data))])
    if concatenate==True:
        data_mat=np.concatenate(np.hstack(data_mat))
    return(data_mat)

def continguous_field(array,num_bins,cont_thr=2):
    if len(array)==0:
        field=[]
    else:
        bool_xx=np.diff(array)<=cont_thr
        xx=0
        field=np.zeros(len(bool_xx))
        for ii in range(len(bool_xx)):
            if bool_xx[ii]==False:
                xx+=1
            field[ii]=xx
        field=np.hstack((0,field))
        if array[0]+(num_bins-1)-array[-1]<cont_thr:
            field[field==unique_adjacent(field)[-1]]=0

    return(field)

from collections import Counter
def most_common(aa):
    counts=list(Counter(aa).values())
    max_count=np.max(counts)
    return(np.asarray(list(Counter(aa).keys()))[counts==max_count],max_count)

def demean(x):
    return(x-np.nanmean(x))

from collections import Counter
from itertools import combinations

def most_common_pair(a_):
    a=np.copy(a_)
    d  = Counter()
    for sub in a:
        if len(a) < 2:
            continue
        #sub.sort()
        for comb in combinations(sub,2):
            d[comb] += 1

    return([d.most_common()[0][0][0],d.most_common()[0][0][1]], d.most_common()[0][1]/len(a))

def fill_diagonal(source_array, diagonal):
    copy = source_array.copy()
    np.fill_diagonal(copy, diagonal)
    return copy

###Smoothing functions
def partition(alist, indices):
    return np.asarray([np.asarray(alist[i:j]) for i, j in zip(indices[:-1], indices[1:])])

def normalise(xx,num_bins=90,take_max=False):
    lenxx=len(xx)
    if lenxx<num_bins:
        xx=np.repeat(xx,10)/10
        lenxx=lenxx*10
    indices_polar=np.arange(lenxx)
    if take_max==True:
        normalized_xx=st.binned_statistic(indices_polar,xx, 'max', bins=num_bins)[0]
    else:
        normalized_xx=st.binned_statistic(indices_polar,xx, 'mean', bins=num_bins)[0]
    return(normalized_xx)

def raw_to_norm(raw_neuron,Trial_times_conc,num_states=4,return_mean=True,smoothing=True,\
                take_max=False,smoothing_sigma=10):
    raw_neuron_split=remove_empty(partition(list(raw_neuron),list(Trial_times_conc)))
    if len(raw_neuron_split)%num_states!=0:
        raw_neuron_split=raw_neuron_split[:len(raw_neuron_split)-len(raw_neuron_split)%num_states]
    
    if take_max==True:
        raw_neuron_split_norm=np.asarray([normalise(raw_neuron_split[ii],take_max=True)\
                                          for ii in np.arange(len(raw_neuron_split))])
    else:
        raw_neuron_split_norm=np.asarray([normalise(raw_neuron_split[ii]) for ii in np.arange(len(raw_neuron_split))])
    
    Actual_norm=(raw_neuron_split_norm.reshape(len(raw_neuron_split_norm)//num_states,\
                                               len(raw_neuron_split_norm[0])*num_states))
    
    if return_mean==True:
        Actual_norm_mean=np.nanmean(Actual_norm,axis=0)
        if smoothing==True:
            Actual_norm_smoothed=smooth_circular(Actual_norm_mean,sigma=smoothing_sigma)
            return(Actual_norm_smoothed)
        else:
            return(Actual_norm_mean)
    else:
        return(Actual_norm)
    
def _nanargmin(arr, axis=0):
    try:
        if len(np.shape(arr))==1:
            return np.nanargmin(arr)
        else:
            return np.nanargmin(arr, axis)
    except ValueError:
        return np.nan
    
def _nanargmax(arr, axis=0):
    try:
        if len(np.shape(arr))==1:
            return np.nanargmax(arr)
        else:
            return np.nanargmax(arr, axis)
    except ValueError:
        return np.nan
    
    
def indep_roll(arr, shifts, axis=1):
    """Apply an independent roll for each dimensions of a single axis.

    Parameters
    ----------
    arr : np.ndarray
        Array of any shape.

    shifts : np.ndarray
        How many shifting to use for each dimension. Shape: `(arr.shape[axis],)`.

    axis : int
        Axis along which elements are shifted. 
    """
    arr = np.swapaxes(arr,axis,-1)
    all_idcs = np.ogrid[[slice(0,n) for n in arr.shape]]

    # Convert to a positive shift
    shifts[shifts < 0] += arr.shape[-1] 
    all_idcs[-1] = all_idcs[-1] - shifts[:, np.newaxis]

    result = arr[tuple(all_idcs)]
    arr = np.swapaxes(result,-1,axis)
    return arr

def partition(alist, indices):
    return np.asarray([np.asarray(alist[i:j]) for i, j in zip(indices[:-1], indices[1:])])

def normalise(xx,num_bins=90,take_max=False):
    lenxx=len(xx)
    if lenxx<num_bins:
        xx=np.repeat(xx,10)/10
        lenxx=lenxx*10
    indices_polar=np.arange(lenxx)
    if take_max==True:
        normalized_xx=st.binned_statistic(indices_polar,xx, 'max', bins=num_bins)[0]
    else:
        normalized_xx=st.binned_statistic(indices_polar,xx, 'mean', bins=num_bins)[0]
    return(normalized_xx)

def raw_to_norm(raw_neuron,Trial_times_conc,num_states=4,return_mean=True,smoothing=True,\
                take_max=False,smoothing_sigma=10):
    raw_neuron_split=remove_empty(partition(list(raw_neuron),list(Trial_times_conc)))
    if len(raw_neuron_split)%num_states!=0:
        raw_neuron_split=raw_neuron_split[:len(raw_neuron_split)-len(raw_neuron_split)%num_states]
    
    if take_max==True:
        raw_neuron_split_norm=np.asarray([normalise(raw_neuron_split[ii],take_max=True)\
                                          for ii in np.arange(len(raw_neuron_split))])
    else:
        raw_neuron_split_norm=np.asarray([normalise(raw_neuron_split[ii]) for ii in np.arange(len(raw_neuron_split))])
    
    Actual_norm=(raw_neuron_split_norm.reshape(len(raw_neuron_split_norm)//num_states,\
                                               len(raw_neuron_split_norm[0])*num_states))
    
    if return_mean==True:
        Actual_norm_mean=np.nanmean(Actual_norm,axis=0)
        if smoothing==True:
            Actual_norm_smoothed=smooth_circular(Actual_norm_mean,sigma=smoothing_sigma)
            return(Actual_norm_smoothed)
        else:
            return(Actual_norm_mean)
    else:
        return(Actual_norm)
    
def unique_nosort(a):
    indexes = np.unique(a, return_index=True)[1]
    return(np.asarray([a[index] for index in sorted(indexes)]))

def one_hot_encode(x,length):
    array=np.zeros((len(x),length))
    for entry in np.arange(len(x)):
        if ~np.isnan(x[entry]):
            array[entry,int(x[entry])]=1
    print(array)
    
    
def non_repeat_ses_maker_old(mouse_recday):
    sessions=np.load(Intermediate_object_folder_dropbox+'Task_num_'+mouse_recday+'.npy')
    num_trials_day=np.load(Intermediate_object_folder_dropbox+'Num_trials_'+mouse_recday+'.npy',\
                                        allow_pickle=True)
    non_repeat_ses=[]
    for session_unique in np.unique(sessions):
        sessions_ses=np.where(sessions==session_unique)[0]
        num_sessions=len(sessions_ses)
        if num_sessions==1:
            non_repeat_ses.append(sessions_ses[0])
        else:
            sessions_ses_trials=sessions_ses[np.where(num_trials_day[sessions_ses]!=0)]
            non_repeat_ses.append(sessions_ses_trials[0])
    non_repeat_ses=np.sort(non_repeat_ses)
    return(non_repeat_ses)

def non_repeat_ses_maker(mouse_recday):
    Tasks=np.load(Intermediate_object_folder_dropbox+'Task_data_'+mouse_recday+'.npy',allow_pickle=True)
    #sessions=np.load(Intermediate_object_folder_dropbox+'Task_num_'+mouse_recday+'.npy')
    num_trials_day=np.load(Intermediate_object_folder_dropbox+'Num_trials_'+mouse_recday+'.npy',\
                                        allow_pickle=True)

    non_repeat_bool_all=[]
    for ses_ind in np.arange(len(Tasks)):
        if ses_ind==0:
            non_repeat_bool=True
        else:
            num_prev_repeats=np.sum([np.array_equal(Tasks[ses_ind],Tasks[:ses_ind][jj])\
                                     for jj in range(len(Tasks[:ses_ind]))])
            if num_prev_repeats==0:
                non_repeat_bool=True
            else:
                non_repeat_bool=False

        non_repeat_bool_all.append(non_repeat_bool)
    non_repeat_bool_all=np.hstack((non_repeat_bool_all))
    num_trials_bool=num_trials_day>0
    non_repeat_ses_bool=np.logical_and(non_repeat_bool_all,num_trials_bool)

    non_repeat_ses=np.where(non_repeat_ses_bool==True)[0]
    return(non_repeat_ses)

def is_invertible(a):
     return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]
    
def circular_sem(a):
    if len(np.shape(a))==2:
        sem_=np.rad2deg(np.hstack(([st.circvar(remove_nan(a[:,ii]))/np.sqrt(len(remove_nan(a[:,ii])))\
                               for ii in range(len(a.T))])))
    elif len(np.shape(a))==1:
        sem_=np.rad2deg(st.circvar(remove_nan(a))/np.sqrt(len(remove_nan(a))))
        
    return(sem_)