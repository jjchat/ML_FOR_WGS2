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
# Reading WGS data file
#------------------------------------
filename="./WGS_dataset.xlsx"
dforig=pd.read_excel(filename, sheet_name='data')
dforig=dforig.rename(columns={"Unnamed: 0": "id"})
dforig['id'] = np.arange(dforig.shape[0])
display(dforig)
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
        k=1e+18 #np.exp(4577.8/100. - 4.33)
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
                
        root    = np.sqrt(root2) #(coeffb**2 - 4*coeffa*coeffc)
        root1   = (0.5/coeffa)*(- coeffb - root)
        root2   = (0.5/coeffa)*(- coeffb + root)
        root    = np.max([root1,root2])
        if root>1:
            root = np.min([root1,root2,1])
        if root<0:
            countNv+=1
                         
        EqCO[ii]=root

    EqCO=EqCO*100 # convert in percentage
    #print("# Nb. of negative conversion : "+str(countNv)+"\n")
    #print("# EqCo (min, max, avg.): ",np.min(EqCO), np.max(EqCO), np.mean(EqCO))
    #plt.hist(EqCO)
    #plt.show()
    return EqCO


eqCO=computeEqCO(dforig['Temperature (C)'].values,dforig['H2'].values,
                 dforig['CO'].values,dforig['H2O'].values,
                 dforig['CO2'].values)

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

compareEqExpCO(eqCO, dforig['CO Conversion (%)'].values, plot=False)
###############################################################################

###############################################################################
#------------------------------------ 
# add Equil. CO in the dataframe
#------------------------------------ 
dforig['Eq_CO_Conversion']=eqCO
display(dforig)

#------------------------------------ 
# shuffle the dataframe
#------------------------------------
df=dforig.copy()
for shid in [1,2,3]:
    df=shuffle(df,random_state=shid)
#display(df)
nb_xs=len(df.columns)-2
nb_ys=2
allx=df[df.columns[2:nb_xs]].to_numpy()
#print(allx.shape)
ally=df.loc[:,['CO Conversion (%)','Eq_CO_Conversion']].to_numpy()
#print(ally.shape)
dids=df['id'].values
refs=df['Reference'].values

#------------------------------------ 
# delete data points 
# key 0 --> no deletion
# key 1 --> deletion of Eq_y<0
# key 2 --> deletion of y > Eq_y
#------------------------------------ 
def deleteData(arx,ary,arid,arref,key=0):
    if key==0:
        return [arx,ary,arid,arref]
    
    alx=[]
    aly=[]
    aid=[]
    arf=[]
    
    if key==1:
        for ii in range(len(ary)):
            if ary[ii][1]>=0: # Eq_CO 
                alx.append(arx[ii])
                aly.append(ary[ii])
                aid.append(arid[ii])
                arf.append(arref[ii])
    if key==2:
        for ii in range(len(ary)):
            if (ary[ii][1]-ary[ii][0])>=0:
                alx.append(arx[ii])
                aly.append(ary[ii])
                aid.append(arid[ii])
                arf.append(arref[ii])

    return [np.array(alx), np.array(aly), aid, arf]

[allx,ally,dids,refs]=deleteData(allx,ally,dids,refs,key=2) 
#print(allx.shape,ally.shape)

#------------------------------------ 
# normalize ip/op features
#------------------------------------ 
class MaxScaler:
    def __init__(self,_Variables,verbose=True):
        self.colmax = np.max(np.abs(_Variables), axis=0)
        if verbose:
            print("# Abs_maximum of features: ",self.colmax)
    def transform(self,_Variables):
        return _Variables/self.colmax
    def inverse_transform(self,_Variables,copy=None):
        return _Variables*self.colmax
    
scalerX=MaxScaler(allx, verbose=False)
allx=scalerX.transform(allx)
print("# X:", allx.shape)
scalerY=MaxScaler(np.array([[100.,100],[0.,0.]]), verbose=False)
ally=scalerY.transform(ally)
print("# Y:", ally.shape)
###############################################################################

###############################################################################
# set XGBoost hyper-parameters  
###############################################################################
para_xgb = {'objective':'reg:squarederror', 'colsample_bytree': 0.9,
            'learning_rate': 0.1, 'min_split_loss':0, 'max_depth': 6,
            'subsample': 0.9, 'min_child_weight': 1, 'alpha':0,
            'n_estimators': 4000}
###############################################################################

###############################################################################
# perform K-fold cross validation
# evaluate the cross-validation performance
# predict and save the test data points
###############################################################################
#------------------------------------ 
# set and initialize parameters for cross validation 
#------------------------------------ 
nbCV=10 # number of cross validation

train_pts    =[]
train_score  =[]
train_rmse   =[]
ctrain_score =[]
ctrain_rmse  =[]
ctrain_gtr1  =[]
ctrain_les0  =[]
ctrain_gtrEq =[]
ctrain_gtrEqp=[]
test_score   =[]
test_rmse    =[]
ctest_score  =[]
ctest_rmse   =[]
ctest_gtr1   =[]
ctest_les0   =[]
ctest_gtrEq  =[]
ctest_gtrEqp =[]
predictionXs=np.inf*np.ones(allx.shape)
predictionYs=np.inf*np.ones([ally.shape[0],3])

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
    train_pts.append(train_y.shape[0])
    print("# cross-validation id: ",s+1)
    print("# shape train-test: ", train_x.shape, train_y.shape, test_x.shape, test_y.shape) 
    ###########################################################
    modelX = xgb.XGBRegressor(**para_xgb)
    modelX.fit(train_x,np.copy(train_y[:,0]))
    ###########################################################

    #############################################
    ##################################
    train_y     = scalerY.inverse_transform(train_y) # transfer to original values
    train_yeq   = train_y[:,1]
    train_y     = train_y[:,0].astype(np.float32) 
    pred_y      = modelX.predict(train_x)
    pred_y      = scalerY.colmax[0]*pred_y #.detach().numpy().reshape(-1) 
    trainSc     = r2_score(train_y, pred_y)
    rmsetrain   = np.sqrt(np.mean((train_y-pred_y)**2)) 
    gtr1train   = np.count_nonzero(pred_y>100)
    les0train   = np.count_nonzero(pred_y<0)
    gtrEqtrainp = compareEqExpCO(train_yeq, pred_y)
    ##################################

    print("#######################")
    print("# Train  : ")
    print("# R2     : ", trainSc)
    print("# RMSE   : ", rmsetrain)
    print("# < 0    : ", les0train)
    print("# > 100  : ", gtr1train)
    print("# Exp>Eq.: ", gtrEqtrainp)
    print("#######################")
 
    train_score.append([trainSc])
    train_rmse.append([rmsetrain])
    ctrain_gtr1.append([gtr1train])
    ctrain_les0.append([les0train])
    ctrain_gtrEqp.append([gtrEqtrainp])
    #############################################

    #############################################
    ##################################
    test_y      = scalerY.inverse_transform(test_y) # transfer to original values
    test_yeq    = test_y[:,1]
    test_y      = test_y[:,0].astype(np.float32) 
    pred_y      = modelX.predict(test_x)
    pred_y      = scalerY.colmax[0]*pred_y 
    testSc      = r2_score(test_y, pred_y)
    rmsetest    = np.sqrt(np.mean((test_y-pred_y)**2)) 
    gtr1test    = np.count_nonzero(pred_y>100)
    les0test    = np.count_nonzero(pred_y<0)
    gtrEqtestp  = compareEqExpCO(test_yeq, pred_y)
    ##################################

    print("#######################")
    print("# Test   : ")
    print("# R2     : ", testSc)
    print("# RMSE   : ", rmsetest)
    print("# < 0    : ", les0test)
    print("# > 100  : ", gtr1test)
    print("# Exp>Eq.: ", gtrEqtestp)
    print("#######################")
 
    test_score.append([testSc])
    test_rmse.append([rmsetest])
    ctest_gtr1.append([gtr1test])
    ctest_les0.append([les0test])
    ctest_gtrEqp.append([gtrEqtestp])
    #############################################
    
    test_x=scalerX.inverse_transform(test_x)
    for rr in range(len(test_y)):
        predictionXs[testyid]=test_x[rr]
        predictionYs[testyid]=np.array([test_y[rr],test_yeq[rr],pred_y[rr]]) 
        testyid += 1

print("####################################################")

#------------------------------------ 
# print overall performance
#------------------------------------ 
#------------------------------------ 
print("# Overall training performance:")
print("# r2_score avg. & std.", np.mean(np.array(train_score), axis=0),
      np.std(np.array(train_score), axis=0))
print("# rmse avg. & std.    ", np.mean(np.array(train_rmse),axis=0),
      np.std(np.array(train_rmse),axis=0))
print("# train < 0           ", np.mean(np.array(ctrain_les0),axis=0))
print("# train > 100         ", np.mean(np.array(ctrain_gtr1),axis=0))
print("# train > Eq.         ", np.mean(np.array(ctrain_gtrEqp),axis=0))
print("# out-of-bound (%)    ", 100*(np.sum(np.array(ctrain_les0),axis=0)+np.sum(np.array(ctrain_gtr1),axis=0))/np.sum(np.array(train_pts)))
print("# th. violation (%)   ", 100*(np.sum(np.array(ctrain_gtrEqp),axis=0))/np.sum(np.array(train_pts)))
#------------------------------------ 

#------------------------------------ 
print("# Overall testing performance:")
print("# r2_score avg. & std.", np.mean(np.array(test_score), axis=0),
      np.std(np.array(test_score), axis=0))
print("# rmse avg. & std.    ", np.mean(np.array(test_rmse),axis=0),
      np.std(np.array(test_rmse),axis=0))
print("# test < 0            ", np.mean(np.array(ctest_les0),axis=0))
print("# test > 100          ", np.mean(np.array(ctest_gtr1),axis=0))
print("# testp > Eq.         ", np.mean(np.array(ctest_gtrEqp),axis=0))
print("# out-of-bound (%)    ", 100*nbCV*(np.mean(np.array(ctest_les0),axis=0)+np.mean(np.array(ctest_gtr1),axis=0))/nbrows)
print("# th. violation (%)   ", 100*nbCV*(np.mean(np.array(ctest_gtrEqp),axis=0))/nbrows)
#------------------------------------

#------------------------------------ 
# save prediction values in an excel sheet
#------------------------------------
columnpred=df.columns[2:].to_list()+['CO_xgboost']
predictionXsYs=np.concatenate((predictionXs.T,predictionYs.T),axis=0).T
dfpred = pd.DataFrame(data=predictionXsYs,columns=columnpred)
dfpred['id']=dids
dfpred['Reference']=refs
dfpred=dfpred[df.columns.to_list()+['CO_xgboost']]
writer = pd.ExcelWriter("WGS_XGBoostpredicted_CO.xlsx", engine='xlsxwriter')
dfpred.to_excel(writer,index=False)
writer.save()
###############################################################################

print(f'# Time (s): {time.time() - start}')
print("# Programme Ends Successfully")
