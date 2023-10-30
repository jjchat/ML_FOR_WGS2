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
import csv
import sklearn
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(1)

#print('Current Python version: {0}'.format(python_version())) # 3.8.8
#print((subprocess.check_output("lscpu", shell=True).strip()).decode()) # Intel(R) Core(TM) i9-9900KF CPU @ 3.60GHz
#print('Current Envronment: {0}'.format(sys.executable)) # anaconda3
#print(pd.__version__) # 1.2.4
#print('sklearn version: ', sklearn.__version__) # 0.24.1
#print(torch.__version__) # 1.10.0
start = time.time()
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
# building Neural Networks model with pyTorch 
# with a custom loss and
# additional activation function
###############################################################################
#------------------------------------ 
# create the custom loss function 
#------------------------------------ 
def computeMSEAndEqLoss(predy,y,y1,beta=0):
    delta = y1-predy
    delta[delta>0]=0
    loss = torch.mean((predy-y)**2 + beta*(delta**2))
    return loss

#------------------------------------ 
# create the class Data
#------------------------------------ 
class Data(Dataset):
    
    # Constructor
    def __init__(self,_X, _Y, _Y1):
        _X     = _X.astype(np.float32)
        _Y     = _Y.astype(np.float32)
        _Y1    = _Y1.astype(np.float32)
        self.x = torch.from_numpy(_X) 
        self.y = torch.from_numpy(_Y)
        self.y1= torch.from_numpy(_Y1)
        self.len = self.x.shape[0]
        #print(self.x.shape)
        
    # Getter
    def __getitem__(self, index):    
        return self.x[index], self.y[index], self.y1[index]
    
    # Get length
    def __len__(self):
        return self.len

#------------------------------------ 
# define the NN architecture 
#------------------------------------ 
class NetCatalyst(nn.Module):
    
    # Constructor
    def __init__(self, Layers):
        super(NetCatalyst, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            torch.nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            self.hidden.append(linear)

    # Prediction
    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.relu(linear_transform(x))
            else:
                x = linear_transform(x)
                x=torch.sigmoid(x) # additional activation function
        return x


#------------------------------------ 
# define the NN training module
#------------------------------------ 
def train(model, train_loader, valid_loader, optimizer, epochs=100,_beta=0):
    i = 0
    cost  = []
    vcost = []
    tolerance=1e-3  # tolerance for the optimization
    for epoch in range(epochs):
        total=0
        for i, (x, y, y1) in enumerate(train_loader):
            optimizer.zero_grad()
            z = model(x)
            #print(x.shape,y.shape,z.view(-1).shape)
            loss = computeMSEAndEqLoss(z.view(-1), y, y1, beta=_beta)
            loss.backward()
            optimizer.step()
            total += loss.item()
        
        cost.append(total)
        if epoch % 1000 == 0 or epoch == epochs-1:    
            print("# train:", epoch, cost[epoch])      
            # early stops if deviations of last 10 consecutive epochs are less than tolerance  
            if epoch>0.5*epochs:
                Stop=True
                for iepoch in range(10):
                    dev=abs(cost[epoch-1-iepoch]-cost[epoch-iepoch])
                    if dev > tolerance:
                        Stop=False
                        break
                
                if Stop==True:
                    print("# Fulfilling stopping criteria at epoch ", epoch)
                    break 
        
        total=0
        for x, y, y1 in valid_loader:     
            z = model(x)
            #print(x.shape,y.shape,z.view(-1).shape)
            loss = computeMSEAndEqLoss(z.view(-1), y, y1, beta=_beta)
            total += loss.item()
        
        vcost.append(total)
        if epoch % 1000 == 0  or epoch == epochs-1:    
            print("# valid:", epoch, vcost[epoch])      
            # early stops if deviations of last 10 consecutive epochs are less than tolerance  
            if epoch>0.5*epochs:
                Stop=True
                for iepoch in range(10):
                    dev=abs(vcost[epoch-1-iepoch]-vcost[epoch-iepoch])
                    if dev > tolerance:
                        Stop=False
                        break
                
                if Stop==True:
                    print("# valid: Fulfilling stopping criteria at epoch ", epoch)
                    break 
        
    return 
###############################################################################

###############################################################################
# perform K-fold cross validation
# evaluate the cross-validation performance
# predict and save the test data points
###############################################################################
#------------------------------------ 
# set and initialize parameters for cross validation 
#------------------------------------ 
np.random.seed(674532)
nbCV      = 10 # number of cross validation
rands     = [np.random.randint(10,1000) for ii in range(nbCV)] 
coeffLoss = 0.1 # coefficient of Thermodynalic Loss function, 0--> no thermodynamic loss
mit       = 7000      # max iteration. 
mom       = 0.99      # momentumfloat, default=0, optional
input_dim = len(allx[0])
output_dim= 1
layers    =[input_dim, 40, 40, 40, 40, 40, output_dim] # NN layers and nodes 

train_vals   = []
train_pts    = []
train_score  = []
train_rmse   = []
ctrain_score = []
ctrain_rmse  = []
ctrain_gtr1  = []
ctrain_les0  = []
ctrain_gtrEq = []
ctrain_gtrEqp= []
test_score   = []
test_rmse    = []
ctest_score  = []
ctest_rmse   = []
ctest_gtr1   = []
ctest_les0   = []
ctest_gtrEq  = []
ctest_gtrEqp = []
predictionXs= np.inf*np.ones(allx.shape)
predictionYs= np.inf*np.ones([ally.shape[0],5])

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

    fit_x, valid_x, fit_y, valid_y = train_test_split(train_x, train_y, 
                                                        test_size=0.1, 
                                                        random_state=rands[s])
    train_pts.append(train_y.shape[0])
    print("# cross-validation id: ",s+1)
    print("# shape train-test: ", train_x.shape, train_y.shape, test_x.shape, test_y.shape) 
    print("# shape fit-valid: ", fit_x.shape, fit_y.shape, valid_x.shape, valid_y.shape)     
    ###########################################################
    modelS         = NetCatalyst(layers)
    optimizerS     = torch.optim.SGD(modelS.parameters(), lr=0.001, momentum=mom)
    train_datasetS = Data(fit_x,fit_y[:,0],fit_y[:,1])
    valid_datasetS = Data(valid_x,valid_y[:,0],valid_y[:,1])
    train_loaderS  = DataLoader(dataset=train_datasetS, batch_size=200, shuffle=True) 
    valid_loaderS  = DataLoader(dataset=valid_datasetS, batch_size=len(valid_y), shuffle=False) 
    train(modelS, train_loaderS, valid_loaderS, optimizerS, epochs=mit, _beta=coeffLoss)

    modelA         = NetCatalyst(layers)
    optimizerA     = torch.optim.Adam(modelA.parameters(), lr=0.001, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=0, amsgrad=False)
    train_datasetA = Data(fit_x,fit_y[:,0],fit_y[:,1])
    valid_datasetA = Data(valid_x,valid_y[:,0],valid_y[:,1])
    train_loaderA  = DataLoader(dataset=train_datasetA, batch_size=200, shuffle=True) 
    valid_loaderA  = DataLoader(dataset=valid_datasetA, batch_size=len(valid_y), shuffle=False) 
    train(modelA, train_loaderA, valid_loaderA, optimizerA, epochs=mit, _beta=coeffLoss)
    ###########################################################
    
    ###########################################################
    ##################################
    test_y      = scalerY.inverse_transform(test_y) # transfer to original values
    test_yeq    = test_y[:,1]
    test_y      = test_y[:,0].astype(np.float32) 
    pred_y      = modelS(torch.from_numpy(test_x.astype(np.float32)))
    pred_yS     = scalerY.colmax[0]*pred_y.detach().numpy().reshape(-1) 
    testScS     = r2_score(test_y, pred_yS)
    rmsetestS   = np.sqrt(np.mean((test_y-pred_yS)**2)) 
    gtr1testS   = np.count_nonzero(pred_yS>100)
    les0testS   = np.count_nonzero(pred_yS<0)
    gtrEqtestpS = compareEqExpCO(test_yeq, pred_yS)
    ##################################
    
    ##################################
    pred_y      = modelA(torch.from_numpy(test_x.astype(np.float32)))
    pred_yA     = scalerY.colmax[0]*pred_y.detach().numpy().reshape(-1)
    testScA     = r2_score(test_y, pred_yA)
    rmsetestA   = np.sqrt(np.mean((test_y-pred_yA)**2)) 
    gtr1testA   = np.count_nonzero(pred_yA>100)
    les0testA   = np.count_nonzero(pred_yA<0)
    gtrEqtestpA = compareEqExpCO(test_yeq, pred_yA)
    ##################################
    
    ##################################
    pred_yE     = 0.5*(pred_yS+pred_yA)
    testScE     = r2_score(test_y, pred_yE)
    rmsetestE   = np.sqrt(np.mean((test_y-pred_yE)**2)) 
    gtr1testE   = np.count_nonzero(pred_yE>100)
    les0testE   = np.count_nonzero(pred_yE<0)
    gtrEqtestpE = compareEqExpCO(test_yeq, pred_yE)
    ##################################
       
    print("#######################")
    print("# Test   : ")
    print("# R2     (test: sgd, adam, ensemble): ", testScS, testScA, testScE)
    print("# RMSE   (test: sgd, adam, ensemble): ", rmsetestS, rmsetestA, rmsetestE)
    print("# < 0    (test: sgd, adam, ensemble): ", les0testS, les0testA, les0testE)
    print("# > 100  (test: sgd, adam, ensemble): ", gtr1testS, gtr1testA, gtr1testE)
    print("# Exp>Eq.(test: sgd, adam, ensemble): ", gtrEqtestpS, gtrEqtestpA, gtrEqtestpE)
    print("#######################")
 
    test_score.append([testScS,testScA,testScE])
    test_rmse.append([rmsetestS,rmsetestA,rmsetestE])
    ctest_gtr1.append([gtr1testS,gtr1testA,gtr1testE])
    ctest_les0.append([les0testS,les0testA,les0testE])
    ctest_gtrEqp.append([gtrEqtestpS,gtrEqtestpA,gtrEqtestpE])
    ###########################################################
    
    test_x=scalerX.inverse_transform(test_x)
    for rr in range(len(test_y)):
        predictionXs[testyid]=test_x[rr]
        predictionYs[testyid]=np.array([test_y[rr],test_yeq[rr],pred_yS[rr],pred_yA[rr],pred_yE[rr]]) 
        testyid += 1

print("####################################################")

#------------------------------------ 
# print overall performance
#------------------------------------ 
#------------------------------------ 
print("# Overall testing performance:")
print("# test_score (avg: sgd, adam, ensemble)", 
      np.mean(np.array(test_score), axis=0))
print("# test_score (std: sgd, adam, ensemble)", 
      np.std(np.array(test_score), axis=0))
print("# test_rmse  (avg: sgd, adam, ensemble)", np.mean(np.array(test_rmse),axis=0))
print("# test_rmse  (std: sgd, adam, ensemble)", np.std(np.array(test_rmse),axis=0))
print("# test < 0   (sgd, adam, ensemble)", np.mean(np.array(ctest_les0),axis=0))
print("# test > 100 (sgd, adam, ensemble)", np.mean(np.array(ctest_gtr1),axis=0))
print("# testp > Eq.(sgd, adam, ensemble)", np.mean(np.array(ctest_gtrEqp),axis=0))
print("# out-of-bound (%)    ", 100*nbCV*(np.mean(np.array(ctest_les0),axis=0)+np.mean(np.array(ctest_gtr1),axis=0))/nbrows)
print("# th. violation (%)   ", 100*nbCV*(np.mean(np.array(ctest_gtrEqp),axis=0))/nbrows)
#------------------------------------ 

#------------------------------------ 
# save prediction values in an excel sheet
#------------------------------------
columnpred=df.columns[2:].to_list()+['CO_sgd','CO_adam','CO_ensemble']
predictionXsYs=np.concatenate((predictionXs.T,predictionYs.T),axis=0).T
dfpred = pd.DataFrame(data=predictionXsYs,columns=columnpred)
dfpred['id']=dids
dfpred['Reference']=refs
dfpred=dfpred[df.columns.to_list()+['CO_sgd','CO_adam','CO_ensemble']]
writer = pd.ExcelWriter("WGS_ANNpredicted_CO.xlsx", engine='xlsxwriter')
dfpred.to_excel(writer,index=False)
writer.save()
###############################################################################

print(f'# Time (s): {time.time() - start}')
print("# Programme Ends Successfully")
