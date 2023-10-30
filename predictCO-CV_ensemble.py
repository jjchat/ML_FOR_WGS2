#! /usr/bin/env python

#--------------------------------
# Authors: Joyjit Chattoraj et al
# Date: October 2023
#--------------------------------
from platform import python_version
import subprocess
import time
import sys
import os
import numpy as np
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

#print('Current Python version: {0}'.format(python_version())) # 3.8.8
#print((subprocess.check_output("lscpu", shell=True).strip()).decode()) # Intel(R) Core(TM) i9-9900KF CPU @ 3.60GHz
#print('Current Envronment: {0}'.format(sys.executable)) # anaconda3
#print(pd.__version__) # 1.2.4
#print('sklearn version: ', sklearn.__version__) # 0.24.1
#print(xgb.__version__) # 1.3.3
start=time.time()
###############################################################################

###############################################################################
#------------------------------------ 
# Reading CO conversion cv values predicted by XGBoost
#------------------------------------
filexgb="WGS_XGBoostpredicted_CO.xlsx" 
dfxgb=pd.read_excel(filexgb)
display(dfxgb)

#------------------------------------ 
# Reading CO conversion cv values predicted by NN with custom loss and activation
#------------------------------------
filenn="WGS_ANNpredicted_CO.xlsx"
dfnn=pd.read_excel(filenn)
display(dfnn)
###############################################################################

###############################################################################
#------------------------------------ 
# compute thermodynamic equilibrium CO conversion
#------------------------------------ 
def computeEqCO(Ta,H2a,COa,H2Oa,CO2a):
    nbxs=len(Ta)
    #print(Ta.shape, np.min(Ta),np.max(Ta),np.count_nonzero(Ta<=100))
    EqCO=np.zeros([nbxs]) 
    countUp=0
    countNv=0
    count_Tlo=0
    #print("# row T H2 CO  H2O CO2 k_eq | root1 root2 root\n")
    for ii in range(nbxs): 
        k=1e+18 
        if (Ta[ii]+273)>100: # K 
            k=np.exp(4577.8/(Ta[ii]+273) - 4.33)
            count_Tlo += 1
            
        coeffa=(1-k)*COa[ii]*COa[ii]
        coeffb=COa[ii]*( H2a[ii] + CO2a[ii] + k*(COa[ii] + H2Oa[ii]) )
        coeffc=H2a[ii]*CO2a[ii] - k*H2Oa[ii]*COa[ii]
        root2=coeffb**2 - 4*coeffa*coeffc
        if  root2<0:
            root2=0  
            countUp+=1
                
        root    = np.sqrt(root2) 
        root1   = (0.5/coeffa)*(- coeffb - root)
        root2   = (0.5/coeffa)*(- coeffb + root)
        root    = np.max([root1,root2])
        if root>1:
            root = np.min([root1,root2,1])
        if root<0:
            countNv+=1
                         
        EqCO[ii]=root

    EqCO=EqCO*100 # convert in percentage
    return EqCO


eqCO=computeEqCO(dfxgb['Temperature (C)'].values,dfxgb['H2'].values,
                 dfxgb['CO'].values,dfxgb['H2O'].values,
                 dfxgb['CO2'].values)

#------------------------------------ 
# compare thermodynamic equilibrium CO
# and experimental CO conversion %
#------------------------------------ 
def compareEqExpCO(eq, exp, plot=False):
    deltaCO=eq-exp
    if plot:
        plt.hist(deltaCO)
        plt.xlabel("thermo-experi")
        plt.show()
        nbs=np.count_nonzero(deltaCO<0)
        print("# Nb. of exp_CO > eq_CO: "+str(nbs)+", "+str(nbs*100/len(eq))+"%\n")
    return  np.count_nonzero(deltaCO<0)

compareEqExpCO(eqCO, dfxgb['CO_xgboost'].values, plot=False)
compareEqExpCO(eqCO, dfnn['CO_ensemble'].values, plot=False)
if np.count_nonzero(dfxgb['Eq_CO_Conversion'].values-dfnn['Eq_CO_Conversion'].values)!=0:
    print("Error, Two CO prediction files are not identically shuffled")
    exit()

#------------------------------------ 
# read X and y variables
#------------------------------------
nb_xs=len(dfxgb.columns)-3
allx=dfxgb[dfxgb.columns[2:nb_xs]].to_numpy()
print(allx.shape)
allyX=dfxgb.loc[:,['CO Conversion (%)','Eq_CO_Conversion','CO_xgboost']].to_numpy()
allyN=dfnn.loc[:,['CO_ensemble']].to_numpy()
ally=np.concatenate((allyX.T,allyN.T), axis=0).T
print(allyX.shape, allyN.shape, ally.shape)
dids=dfxgb['id'].values
refs=dfxgb['Reference'].values
###############################################################################

###############################################################################
# build Ensemble model  
###############################################################################
def predictEnsemble(_eqCO, pred_yxgb, pred_ynn):
    nbys=len(_eqCO)
    pred_y=[]

    #------------------------------------ 
    # definition of ensemble (XGBoost+NN) method
    #------------------------------------ 
    for ii in range(nbys):
        yval=0.5*(pred_yxgb[ii]+pred_ynn[ii]) 
        if yval < 0  or yval > 100 or yval > _eqCO[ii]:
            yval=pred_ynn[ii]
        pred_y.append(yval)

    pred_y = np.array(pred_y)
    #print(_eqCO.shape, pred_yxgb.shape, pred_ynn.shape, pred_y.shape)
    return pred_y
###############################################################################

###############################################################################
# evaluate the cross-validation performance
# predict and save the test data points
###############################################################################
#------------------------------------ 
# set and initialize parameters for cross validation 
#------------------------------------ 
nbCV=10 # number of cross validation

train_score =[]
train_rmse  =[]
test_score  =[]
test_rmse   =[]
ctrain_score=[]
ctrain_rmse =[]
ctest_score =[]
ctest_rmse  =[]
ctest_gtr1  =[]
ctest_les0  =[]
ctest_gtrEq =[]
ctest_gtrEqp=[]
predictionXs=np.inf*np.ones(allx.shape)
predictionYs=np.inf*np.ones([ally.shape[0],5])

nbrows=len(ally)
nb_test_pts = [nbrows//nbCV + (1 if nn < nbrows%nbCV else 0) for nn in range(nbCV)]
#print(nb_test_pts)
testyid=0

#------------------------------------ 
# split data points into train and test
#------------------------------------ 
def train_test_split_forcv(_X, _y, foldid, testpt_array):
    test_stindex = int(np.sum(testpt_array[:foldid]))
    test_enindex = int(np.sum(testpt_array[:foldid+1]))
    _trainx=np.delete(_X,np.arange(test_stindex,test_enindex),axis=0)
    _trainy=np.delete(_y,np.arange(test_stindex,test_enindex),axis=0)
    _testx=_X[test_stindex:test_enindex,:]
    _testy=_y[test_stindex:test_enindex,:]
    #print(s,testpt_array[s],test_stindex,test_enindex,_trainx.shape,_testx.shape) 
    return _trainx, _testx, _trainy, _testy 

#------------------------------------ 
# loop over k-fold cross validation
#------------------------------------ 
for s in range(nbCV):
    print("####################################################")
    train_x, test_x, train_y, test_y = train_test_split_forcv(allx, ally, s, 
                                                              nb_test_pts)

    print("# cross-validation id: ",s+1)
    print("# shape train-test: ", train_x.shape, train_y.shape, test_x.shape, test_y.shape) 
    ###########################################################

    ##################################
    test_yeq    = test_y[:,1]
    pred_yxgb   = test_y[:,2]
    pred_ynn    = test_y[:,3]
    test_y      = test_y[:,0] 
    pred_y      = predictEnsemble(test_yeq,pred_yxgb,pred_ynn)
    testSc      = r2_score(test_y, pred_y)
    rmsetest    = np.sqrt(np.mean((test_y-pred_y)**2)) 
    gtr1test    = np.count_nonzero(pred_y>100)
    les0test    = np.count_nonzero(pred_y<0)
    gtrEqtestp  = compareEqExpCO(test_yeq, pred_y)
    ##################################

    print("#######################")
    print("# R2     : ", testSc)
    print("# RMSE   : ", rmsetest)
    print("# < 0    : ", les0test)
    print("# > 100  : ", gtr1test)
    print("# Exp>Eq.: ", gtrEqtestp)
    print("###################################")
 
    test_score.append([testSc])
    test_rmse.append([rmsetest])
    ctest_gtr1.append([gtr1test])
    ctest_les0.append([les0test])
    ctest_gtrEqp.append([gtrEqtestp])

    for rr in range(len(test_y)):
        predictionXs[testyid]=test_x[rr]
        predictionYs[testyid]=np.array([test_y[rr],test_yeq[rr],
                                        pred_yxgb[rr], pred_ynn[rr],
                                        pred_y[rr]]) 
        testyid += 1
print("####################################################")

#------------------------------------ 
# print overall performance
#------------------------------------ 
print("# Overall performance:")
print("# test_score (avg.)", 
      np.mean(np.array(test_score), axis=0))
print("# test_score (std.)", 
      np.std(np.array(test_score), axis=0))
print("# test_rmse  (avg.)", np.mean(np.array(test_rmse),axis=0))
print("# test_rmse  (std.)", np.std(np.array(test_rmse),axis=0))
print("# test < 0   (avg.)", np.mean(np.array(ctest_les0),axis=0))
print("# test > 100 (avg.)", np.mean(np.array(ctest_gtr1),axis=0))
print("# testp > Eq.(avg.)", np.mean(np.array(ctest_gtrEqp),axis=0))
print("# out-of-bound (%)    ", 100*nbCV*(np.mean(np.array(ctest_les0),axis=0)+np.mean(np.array(ctest_gtr1),axis=0))/nbrows)
print("# th. violation (%)   ", 100*nbCV*(np.mean(np.array(ctest_gtrEqp),axis=0))/nbrows)
#------------------------------------ 

#------------------------------------ 
# save prediction values in an excel sheet
#------------------------------------
columnpred=dfxgb.columns[2:].to_list()+['CO_ann']+['CO_xgboost_ann']
predictionXsYs=np.concatenate((predictionXs.T,predictionYs.T),axis=0).T
dfpred=pd.DataFrame(data=predictionXsYs,columns=columnpred)
dfpred['id']=dids
dfpred['Reference']=refs
dfpred=dfpred[dfxgb.columns.to_list()+['CO_ann']+['CO_xgboost_ann']]
writer = pd.ExcelWriter("WGS_XGBoostANNpredicted_CO.xlsx", engine='xlsxwriter')
dfpred.to_excel(writer,index=False)
writer.save()
###############################################################################

print(f'# Time (s): {time.time() - start}')
print("# Programme Ends Successfully")
