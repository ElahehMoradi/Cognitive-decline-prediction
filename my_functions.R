#R Script for Cross-Validated Regression to Predict Cognitive Decline by combining MRI scores derived from ridge linear regression
#with additional information such as plasma measures, demographic data, and cognitive assessments using Random Forest regression.
# Author: Elaheh Moradi, University of Eastern Finland, Kuopio ,Finland (elaheh.moradi@uef.fi)
#last updated 30.12.2024
#
# Requirements:
# R version 4.1.1 or later
#install.packages("glmnet")
#install.packages("Metrics")
#install.packages("randomForest")
#install.packages("caret", dependencies = c("Depends", "Suggests"))


#Parameters: 
#Xdata: input matrix (additional information such as plasma measures, demographic data, and cognitive assessments), of dimension nobs x nvars; each row is an observation vector, the rownames of Xdata are the subjects IDs.
#MRI: input MRI matrix, of dimension nobs x nvars; each row is an observation vector
#label: response variable 
#seed: seed number for reproducing the results

#Returned values:
#yhat: predicted label
#MRIscores: predicted MRI score by RLR
#R
#MAE
#imp_all: a matrix of importance derived from K fold experiments of random forest regression
#coef_all_mri: a matrix of RLR coefficients derived from K fold experiments from MRI part

#
# Usage: 
#Set working directory to directory containing R script
#source('my_functions.R')
#results= Regression_RLR_RFR(Xdata,MRI,label,seed )
##################################

#####################################
newfunc_regression= function(Xtrain, Ytrain,  RID_train){
  library(glmnet)
  library(caret)
  
  folds1<- createFolds(Ytrain, k = 10, list = TRUE, returnTrain = FALSE)
  allAct1<- vector()
  allPred1= vector()
  RID_pred1= vector()
  
  for (j in 1:length(folds1)){
    print(j)
    ind1<- folds1[[j]]
    Xtrain1= Xtrain[-ind1,]
    Ytrain1=Ytrain[-ind1]
    
    Xtest1= Xtrain[ind1,]
    Ytest1= Ytrain[ind1]
    
    RID_pred1= c(RID_pred1,RID_train[ind1])
    model=cv.glmnet(Xtrain1, Ytrain1,alpha= 0)
    
    pred1<- predict(model, Xtest1,  s="lambda.min")
    allAct1= c(allAct1, Ytest1)
    allPred1= c(allPred1,pred1)
  }
  
  ii= match( RID_train,RID_pred1 )
  RID_pred1= RID_pred1[ii]
  print(identical(RID_pred1, RID_train))
  allAct1= allAct1[ii]
  allPred1= allPred1[ii]
  
  
  results= list(allAct1= allAct1, allPred1= allPred1,RID_pred1= RID_pred1)
  return(results)
}
###########################################


##############################################
Regression_RLR_RFR= function(Xdata, MRI, label,seed){
  library(glmnet)
  library(caret)
  library(Metrics)
  library(randomForest)
  
  RID = rownames(Xdata)
  set.seed(seed)
  folds<- createFolds(label, k = 10, list = TRUE, returnTrain = FALSE)
  
  allAct<- vector()
  allPred= vector()
  RID_pred= vector()
  all_imp= matrix(0, nrow = (ncol(Xdata)+1), ncol= 10)
  allMRI= vector()
  all_coef_mri= matrix(0, nrow = ncol(MRI), ncol= 10)
  
  for (i in 1:length(folds)){
    ind<- folds[[i]]
    RID_pred= c(RID_pred,RID[ind])
    Xtrain= Xdata[-ind,]
    Ytrain=label[-ind]
    MRI_train= MRI[-ind,]
    Xtest= Xdata[ind,]
    Ytest= label[ind]
    MRI_test= MRI[ind,]
    RID_train= RID[-ind]
    
    normParam <- preProcess(MRI_train)
    MRI_train <- predict(normParam, MRI_train)
    MRI_test<- predict(normParam, MRI_test)
    
    res= newfunc_regression(MRI_train, Ytrain,RID_train)
    RID_pred1= res$RID_pred1
    pred1= res$allPred1
    act1= res$allAct1
    
    
    model=cv.glmnet(MRI_train, Ytrain,alpha= 0)
    tmp_mri <- as.vector(coef(model, s = "lambda.min"))[-1]
    all_coef_mri[,i]=tmp_mri
    pred_mri<- predict(model, MRI_test,  s="lambda.min")
    
    
    cnames= c(colnames(Xtrain), "mri")
    Xtrain= cbind(Xtrain, pred1)
    Xtest= cbind(Xtest, pred_mri)
    colnames(Xtrain)= cnames
    colnames(Xtest)= cnames
    
    rf= randomForest(Xtrain,Ytrain, ntree= 1000, importance = T)
    tmp= importance(rf)[,1]
    all_imp[,i]=tmp
    ypred= predict(rf, Xtest)
    
    allMRI= c(allMRI, pred_mri)
    allAct= c(allAct, Ytest)
    allPred= c(allPred,ypred)
    
  }
  rownames(all_imp)= colnames(Xtrain)
  
  rownames(all_coef_mri)= colnames(MRI)
  
  ind= match(RID, RID_pred)
  yhat= allPred[ind]
  MRIscores= allMRI[ind]
  
  R=  cor(allAct,allPred)
  MAE=mae(allAct,allPred)
  
  
  results= list(yhat = yhat, MRIscores= MRIscores,R= R, MAE = MAE, all_imp = all_imp, all_coef_mri= all_coef_mri)
  return(results)
}