#!/usr/bin/env Rscript
Args <- commandArgs()
getTrainingModel <- function(Type,method, trainData,  fileTile, times, ArgsFile) {
  trainData <- read.csv(trainData, sep=",", header = TRUE)
  
  lightlabel=mapvalues(trainData$Class,from=c(-1),to=c(0))
  colNum <- ncol(trainData)-1
  # 把原始训练集中的Class列交换到第一列（如果原始数据是从weka转来???那么class在最后一列）
  if(names(trainData)[length(trainData)] == "Class")
  {
    names<- names(trainData)
    newNames <- names[1:colNum]
    newNames = c("Class",newNames)
    trainData <- trainData[,newNames]
  }
  
  trainData$Class <- mapvalues(trainData$Class,from=c("-1","1"),to=c("F","T"))
  trainData$Class <- as.factor(trainData$Class)
  
  trainData.y <- trainData$Class
  #trainData.y=mapvalues(trainData.y,from=c(-1),to=c(0))
  trainData.x <- trainData[,c(-1)]
  
  # Args <- read.csv(file = ArgsFile, header = T, sep = "\t", quote = "\"")
  # print(Args)
  #RFArgs <- read.csv(file = ArgsFile, header = T, sep = "\t", quote = "\"" )
  #RFMtry <- RFArgs$mtry[which(RFArgs==fileTile,arr.ind=T)[[1]]]
  
  
  
  if(method=="svm"){
    Args <- read.csv(file = ArgsFile, header = T, sep = "\t", quote = "\"")
    #参数
    # SVMGamma <- Args$gamma[which(Args==fileTileVecs[fi],arr.ind=T)[[1]]]
    # SVMCost <-  Args$cost[which(Args==fileTileVecs[fi],arr.ind=T)[[1]]]
    SVMGamma <- Args$gamma[which(Args==fileTile,arr.ind=T)[[1]]]
    print(SVMGamma)
    SVMCost <-  Args$cost[which(Args==fileTile,arr.ind=T)[[1]]]
    trainModel <- svm(trainData.x, trainData.y, kernel = "radial", gamma =SVMGamma , cost =SVMCost , probability = TRUE)
    saveRDS(trainModel,sprintf("model/%s_%s_%s_%s.rds",Type,fileTile,method,times))
    
  } else if(method=="rf"){
    Args <- read.csv(file = ArgsFile, header = T, sep = "\t", quote = "\"")
    #参数
    RFmtry <- Args$mtry[which(Args==fileTile,arr.ind=T)[[1]]]
    trainModel <- randomForest(trainData.x, trainData.y, mtry=RFmtry, ntree=1000, keep.forest=TRUE, importance=TRUE)
    saveRDS(trainModel,sprintf("model/%s_%s_%s_%s.rds",Type,fileTile,method,times))
  } else if(method=="lightgbm"){
    #注意lightgbm的参数的分隔符是","
    Args <- read.csv(file = ArgsFile, header = T, sep = ",", quote = "\"")
    lgb_train <- lgb.Dataset(
      data = data.matrix(trainData.x), 
      label = lightlabel, 
      free_raw_data = FALSE
    )
    # 参数
    params <- list(
      learning_rate = Args$learningRate[which(Args==fileTile,arr.ind=T)[[1]]],
      num_leaves = as.integer(Args$numLeaves[which(Args==fileTile,arr.ind=T)[[1]]]),
      max_depth = as.integer(Args$maxDepth[which(Args==fileTile,arr.ind=T)[[1]]]),
      min_data_in_leaf = as.integer(Args$minDataInLeaf[which(Args==fileTile,arr.ind=T)[[1]]]),
      max_bin = as.integer(Args$maxBin[which(Args==fileTile,arr.ind=T)[[1]]]),
      feature_fraction = Args$featureFraction[which(Args==fileTile,arr.ind=T)[[1]]],
      min_sum_hessian = Args$minSumHessian[which(Args==fileTile,arr.ind=T)[[1]]],
      lambda_l1 = Args$lambdaL1[which(Args==fileTile,arr.ind=T)[[1]]],
      lambda_l2 = Args$lambdaL2[which(Args==fileTile,arr.ind=T)[[1]]],
      drop_rate = Args$dropRate[which(Args==fileTile,arr.ind=T)[[1]]],
      max_drop = as.integer(Args$maxDrop[which(Args==fileTile,arr.ind=T)[[1]]])
    )
    # 模型
    trainModel <- lightgbm(
      params = params,
      data = lgb_train,
      nrounds = 300,
      early_stopping_rounds = 10,
      num_threads = 16,
      objective = "binary"
    )
    lgb.save(trainModel, sprintf("model/%s_%s_%s_%s.model",Type,fileTile,method,times))
  }else if(method=="nb"){
    
    trainModel <- naiveBayes(Class~.,data=trainData,type="raw")
    saveRDS(trainModel,sprintf("model/%s_%s_%s_%s.rds",Type,fileTile,method,times))
   }else if(method=="lr"){
    trainModel=glm(Class~.,data=trainData,family=binomial(link="logit"),control=list(maxit=100))
    saveRDS(trainModel,sprintf("model/%s_%s_%s_%s.rds",Type,fileTile,method,times))
      }

  # else if(method=="rsnns"){
  #   trainModel <- mlp(trainData.x, lightlabel, size=c(32,16), maxit=1000, initFunc="Randomize_Weights", initFuncParams=c(-0.3,0.3), learnFunc="Rprop",learnFuncParams=c(0.1,0),hiddenActFunc="Act_Logistic",linOut=FALSE)
  #   saveRDS(trainModel,sprintf("featureExtraction/H-model/H_%s_%s.rds",fileTile,"model_rsnns"))
 # }
    else if (method=="xgboost"){
    
    
    
    
    trainData$Class=mapvalues(trainData$Class,from=c("F","T"),to=c(0,1))
    # trainData$Class <- as.factor( trainData$Class)
    data.y <- trainData$Class
    data.x <- trainData[,c(-1)]
    
    XGArgs <- read.csv(file = ArgsFile, header = T, sep = "\t", quote = "\"")
    #bestmtry <- RFArgs$mtry[which(RFArgs==fileTileVecs,arr.ind=T)[[1]]]
    param = list(objective = XGArgs$objective[which(XGArgs==fileTile,arr.ind=T)[[1]]],
                 eval_metric = XGArgs$eval_metric[which(XGArgs==fileTile,arr.ind=T)[[1]]],
                 max_depth = XGArgs$max_depth[which(XGArgs==fileTile,arr.ind=T)[[1]]],
                 eta = XGArgs$eta[which(XGArgs==fileTile,arr.ind=T)[[1]]],
                 gamma = XGArgs$gamma[which(XGArgs==fileTile,arr.ind=T)[[1]]],
                 subsample = XGArgs$subsample[which(XGArgs==fileTile,arr.ind=T)[[1]]],
                 colsample_bytree = XGArgs$colsample_bytree[which(XGArgs==fileTile,arr.ind=T)[[1]]],
                 min_child_weight = XGArgs$min_child_weight[which(XGArgs==fileTile,arr.ind=T)[[1]]],
                 max_delta_step = XGArgs$max_delta_step[which(XGArgs==fileTile,arr.ind=T)[[1]]]
    )
    
    
    
    dtune = xgb.DMatrix(data = as.matrix(data.x), label = as.matrix(data.y))
    
    xgb.valid.fit = xgb.train(data = dtune, params = param, nrounds = XGArgs$nrounds, verbose = T, maximize = F)
    
    xgb.save(xgb.valid.fit,sprintf("model/%s_%s_%s_%s.h5",Type,fileTile,method,times))
    
    # xgb.valid.fit = xgb.load(sprintf('model/VF_G-_xgboost_predict_%s.h5',fileVecs[fi]))
    
  }
  
}


library(e1071)
library(plyr)
library(ROCR)
library(caret)
#library(RSNNS)
library(lightgbm)
library("randomForest")
library("xgboost")


Type = "SPNG"
times = 10
fileTileVecs = c("stacking")
fileVecs = fileTileVecs
for(i in 1:times){
  
  for(fi in 1:length(fileVecs)) {
    
   file = sprintf("feature_train/%s/%s_train_%s_label.csv",i,Type,fileVecs[fi])
    
    # if(fileTileVecs[fi] == "S-FPSSM" || fileTileVecs[fi] == "SOCN" || fileTileVecs[fi] == "MORAN"){
    #   next
    # }
    print(fileTileVecs[fi])
    getTrainingModel(Type=Type,method = "svm", trainData=file,  fileTile=fileTileVecs[fi], times=i, ArgsFile=sprintf("table/%s_stacking_tune_svm_%s.csv",Type,i))
    # getTrainingModel(method = "rf", trainData=fileVecs[fi],  fileTile=fileTileVecs[fi], times=i, ArgsFile=sprintf("table/phage_rf_normal_try_features_tunedArgs_%s.csv",i))
    
    getTrainingModel(Type=Type,method = "rf", trainData=file,  fileTile=fileTileVecs[fi],times = i,ArgsFile = sprintf("table/%s_stacking_tune_rf_%s.csv",Type,i))
    # getTrainingModel(Type=Type,method = "lightgbm", trainData=file,  fileTile=fileTileVecs[fi], times=i, ArgsFile=sprintf("table/phage_stacking_tune_lightgbm_%s.csv",i))
    getTrainingModel(Type=Type,method = "xgboost", trainData=file,  fileTile=fileTileVecs[fi],time=i,ArgsFile =sprintf("table/%s_stacking_tune_xgboost_%s.csv",Type,i) )
    getTrainingModel(Type=Type,method = "nb", trainData=file,  fileTile=fileTileVecs[fi], times=i, ArgsFile="")
    #getTrainingModel(method = "rsnns", trainData=fileVecs[fi],  fileTile=fileTileVecs[fi])
    getTrainingModel(Type=Type,method = "lr", trainData=file,  fileTile=fileTileVecs[fi], times=i, ArgsFile="")


    

  }
}


#总共生成模型10*10  每个特征生成10个模型



















