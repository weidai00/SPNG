#!/usr/bin/env Rscript
# source ("T3ROC-style.R")
# source ("utility.R")

# Rscript protein_stacking_model_cross_validation.R svm 10 5
library(e1071)
library(ROCR)
library("randomForest")
library(plyr)
#library(caret)
#start with Args[6]
# Args = commandArgs()
library(xgboost)
library(lightgbm)
Args = commandArgs() 


cv<-function(Type,model,times,fold,inputFile,tuneFile,fileTileVecs){
  
  

group1PrecisionList = list()
group2PrecisionList = list()
group3PrecisionList = list()
precisionList_sum = list()
group1LabelList = list()
group2LabelList = list()
group3LabelList = list()
labelList_sum = list()

roc.x.values=vector()
roc.y.values=vector()
auc.legends=vector()
mat = vector()
matt = vector()
mattt = vector()

for(fi in 1:length(fileTileVecs))          #代表多少个csv文件
{
  
  testNumber <- times
  testAUCNumber <- 1;
  
  test_accs=vector()
  test_precisions=vector()
  test_recalls = vector()
  test_accs = vector()
  test_mccs = vector()
  test_sns = vector()
  test_sps = vector()
  test_Fvalues=vector()
  test_specificitys=vector()
  test_aucs=vector()
  precisionList = list()
  labelList = list()
  foldList = list()
  
  
  for(ti in 1:testNumber)
  {
    # inputDir = sprintf("../feature_full/%s",ti)
    file_index = (fi-1)*testNumber+ti
    fileVecs = inputFile[file_index]
    cat("Read data : ")
    cat(fileVecs)
    cat("\n")
    data = read.csv(fileVecs, sep=",", header = TRUE)
	  data[data =="nan"]<-0
    data[data =="inf"]<-0
    data[data =="Inf"]<-0
    data[data =="-nan"]<-0
    data[data =="-inf"]<-0
    data[data =="-Inf"]<-0
    data[is.na(data)] <- 0
	
	
    colNum <- ncol(data)-1
    # 把原始训练集中的Class列交换到第一列（如果原始数据是从weka转来???,那么class在最后一列）
    if(names(data)[length(data)] == "Class")
    {
      names<- names(data)
      newNames <- names[1:colNum]
      newNames = c("Class",newNames)
      data <- data[,newNames]
    }
    
    if(model=="lr"){
      data$Class=mapvalues(data$Class,from=c("-1","1"),to=c("F","T"))
    ## 因为randomforest的因变量y需要时factor类型
    data$Class <- as.factor(data$Class)
    data.y <- data$Class
    data.x <- data[,c(-1)]
    #cross validation
    #good: 123 125
    #set.seed(ti*123)
    set.seed(ti*123)
    
    cvNumber <- fold
    n=nrow(data)
    
    ## generate an array containing 1000 number in random, which varies from 1 to 10
    ## This is used to identify the fold number during process cross validation.
    folds=floor(cvNumber*runif(n))+1
    data$folds=folds
    data$score=0
    ## ==========================
    ## use randomforest to do 5 cross validation
    
    accs=vector()
    precisions=vector()
    recalls = vector()
    accs = vector()
    mccs = vector()
    sns = vector()
    sps = vector()
    Fvalues=vector()
    specificitys=vector()
    aucs=vector()
    
    for(i in 1:cvNumber){
      
      data.train <- data[data$folds!=i,]
      data.test <- data[data$folds==i,]
      
      data.train.y <- data.train$Class
      colNum <- ncol(data.train)
      data.train.x <- data.train[,c(2:(colNum-2))]
      
      glm.fit=glm(Class~.,data=data.train[,c(1:(colNum-2))],family=binomial(link="logit"),control=list(maxit=100))
      # p<-predict(glm.fit,type='response')
      # qplot(seq(-2,2,length=80),sort(p),col="response")
     if (dir.exists("model")){
          
          # if (dir.exists(sprintf("model/%s",model))){
            print("exist")
          # }
          # else{
          #   dir.create(sprintf("model/%s",model))
          # }
        } 
        else{
          dir.create("model")
          # dir.create(sprintf("model/%s",model))
        }
      # saveRDS(glm.fit,sprintf("model/%s_%s_cv_%s_%s_%s.rds",Type,model,fileTileVecs[fi],i,ti))
      rfNormal.predictions<-predict(glm.fit, data.test[,c(2:(colNum-2))],type='response')
      cat("rfNormal.predictions\n")
      print(rfNormal.predictions)
      
      data[folds==i,]$score=rfNormal.predictions
      ##使用caret包中的confusionMatrix()计算各种指标
      rf.class <- as.integer(rfNormal.predictions > 0.5)
      rf.class = mapvalues(rf.class,from=c("0","1"),to=c("F","T"))
      
      confMat=table(rf.class,data.test$Class)
      cat("confMat of rf:\n")
      print(confMat)
      
      if(nrow(confMat)<2)
      {
        cat("%%%$$$###\n")
        next
      }
      
      #precisionList[[i]] <- rfNormal.predictions[,2]
      #labelList[[i]] <- data.test$Class
      # precisionList[[testAUCNumber]] <- rfNormal.predictions
      # labelList[[testAUCNumber]] <- data.test$Class
      # foldList[[testAUCNumber]] <- data[data$folds==i,]$fold
      
      testAUCNumber <- testAUCNumber+1
      
      
      # compute AUC
      rf.pred = ROCR::prediction(rfNormal.predictions,data.test$Class)
      rf.perf = ROCR::performance(rf.pred,"tpr","fpr")
      roc.rf.x=unlist(attr(rf.perf,"x.values"))
      roc.rf.y=unlist(attr(rf.perf,"y.values"))
      
      rf.auc <- ROCR::performance(rf.pred, "auc")@y.values
      aucs = c(aucs,unlist(rf.auc))
      
      precision=confMat[2,2]/(confMat[2,2]+confMat[2,1])
      recall=confMat[2,2]/(confMat[2,2]+confMat[1,2])
      acc = (confMat[1,1]+confMat[2,2])/(confMat[2,2]+confMat[1,2]+confMat[1,1]+confMat[2,1])
      sp = confMat[1,1]/(confMat[1,1]+confMat[2,1])
      mcc = (confMat[1,1]*confMat[2,2]-confMat[1,2]*confMat[2,1])/sqrt((confMat[2,2]+confMat[1,2])*(confMat[2,2]+confMat[2,1])*(confMat[1,1]+confMat[1,2])*(confMat[1,1]+confMat[2,1]))
      Fvalue = 2.0*precision*recall/(precision+recall)
      
      precisions = c(precisions,precision)
      recalls = c(recalls,recall)
      accs = c(accs,acc)
      sps = c(sps,sp)
      mccs = c(mccs,mcc)
      Fvalues = c(Fvalues,Fvalue)
    }
    # subcolset <- names(data)
    }
    if(model=="rf"){
      data$Class=mapvalues(data$Class,from=c("-1","1"),to=c("F","T"))
      
      ## 因为randomforest的因变量y需要时factor类型
      data$Class <- as.factor(data$Class)
      data.y <- data$Class
      data.x <- data[,c(-1)]
      
      RFArgs <- read.csv(file = tuneFile[ti], header = T, sep = "\t", quote = "\"" )
      bestmtry <- RFArgs$mtry[which(RFArgs==fileTileVecs[fi],arr.ind=T)[[1]]] #change
      cat("Best mtry value: \n")
      print(bestmtry)
      #cross validation
      #good: 123 125
      set.seed(ti*123)
      # set.seed(123)
      cvNumber <- fold #change
      n=nrow(data)
      
      ## generate an array containing 1000 number in random, which varies from 1 to 10
      ## This is used to identify the fold number during process cross validation.
      folds=floor(cvNumber*runif(n))+1
      data$folds=folds
      data$score = 0
      ## ==========================
      ## use randomforest to do 5 cross validation
      
      accs=vector()
      precisions=vector()
      recalls = vector()
      accs = vector()
      mccs = vector()
      sns = vector()
      sps = vector()
      Fvalues=vector()
      specificitys=vector()
      aucs=vector()
      
      for(i in 1:cvNumber){
        
        data.train <- data[data$folds!=i,]
        data.test <- data[data$folds==i,]
        
        data.train.y <- data.train$Class
        colNum <- ncol(data.train)
        data.train.x <- data.train[,c(2:(colNum-2))]
        
        rfNormal <-randomForest(data.train.x,data.train.y, mtry=bestmtry, ntree=1000, 
                                keep.forest=TRUE, importance=TRUE)
        
        if (dir.exists("model")){
          
          # if (dir.exists(sprintf("model/%s",model))){
            print("exist")
          # }
          # else{
          #   dir.create(sprintf("model/%s",model))
          # }
        } 
        else{
          dir.create("model")
          # dir.create(sprintf("model/%s",model))
        }
        # saveRDS(rfNormal,sprintf("model/%s_%s_cv_%s_%s_%s.rds",Type,model,fileTileVecs[fi],i,ti))
        rfNormal.predictions <- predict(rfNormal,type="prob",newdata=data.test[,c(2:(colNum-2))])[,2]#change
        cat("rfNormal.predictions\n")
        print(rfNormal.predictions)
        
        data[folds==i,]$score=rfNormal.predictions
        ##使用caret包中的confusionMatrix()计算各种指标
        rf.class <- as.integer(rfNormal.predictions > 0.5)
        rf.class = mapvalues(rf.class,from=c("0","1"),to=c("F","T"))
        
        confMat=table(rf.class,data.test$Class)
        cat("confMat of rf:\n")
        print(confMat)
        
        if(nrow(confMat)<2)
        {
          cat("%%%$$$###\n")
          next
        }
        
        #precisionList[[i]] <- rfNormal.predictions[,2]
        #labelList[[i]] <- data.test$Class
        # precisionList[[testAUCNumber]] <- rfNormal.predictions
        # labelList[[testAUCNumber]] <- data.test$Class
        # foldList[[testAUCNumber]] <- data[data$folds==i,]$fold
        
        testAUCNumber <- testAUCNumber+1
        
        
        # compute AUC
        rf.pred = ROCR::prediction(rfNormal.predictions,data.test$Class)
        rf.perf = ROCR::performance(rf.pred,"tpr","fpr")
        roc.rf.x=unlist(attr(rf.perf,"x.values"))
        roc.rf.y=unlist(attr(rf.perf,"y.values"))
        
        rf.auc <- ROCR::performance(rf.pred, "auc")@y.values
        aucs = c(aucs,unlist(rf.auc))
        
        precision=confMat[2,2]/(confMat[2,2]+confMat[2,1])
        recall=confMat[2,2]/(confMat[2,2]+confMat[1,2])
        acc = (confMat[1,1]+confMat[2,2])/(confMat[2,2]+confMat[1,2]+confMat[1,1]+confMat[2,1])
        sp = confMat[1,1]/(confMat[1,1]+confMat[2,1])
        mcc = (confMat[1,1]*confMat[2,2]-confMat[1,2]*confMat[2,1])/sqrt((confMat[2,2]+confMat[1,2])*(confMat[2,2]+confMat[2,1])*(confMat[1,1]+confMat[1,2])*(confMat[1,1]+confMat[2,1]))
        Fvalue = 2.0*precision*recall/(precision+recall)
        
        precisions = c(precisions,precision)
        recalls = c(recalls,recall)
        accs = c(accs,acc)
        sps = c(sps,sp)
        mccs = c(mccs,mcc)
        Fvalues = c(Fvalues,Fvalue)
      
    }
    
    
    
    
    
    
      
      
      
    }
    if(model=="nb"){
      
      data$Class=mapvalues(data$Class,from=c("-1","1"),to=c("F","T"))
      
      ## 因为randomforest的因变量y需要时factor类型
      data$Class <- as.factor(data$Class)
      data.y <- data$Class
      data.x <- data[,c(-1)]
      
      #cross validation
      #good: 123 125
      #set.seed(ti*123)
      set.seed(ti*123)
      
      cvNumber <- fold #change
      n=nrow(data)
      
      ## generate an array containing 1000 number in random, which varies from 1 to 10
      ## This is used to identify the fold number during process cross validation.
      folds=floor(cvNumber*runif(n))+1
      data$folds=folds
      data$score=0
      ## ==========================
      ## use randomforest to do 5 cross validation
      
      accs=vector()
      precisions=vector()
      recalls = vector()
      accs = vector()
      mccs = vector()
      sns = vector()
      sps = vector()
      Fvalues=vector()
      specificitys=vector()
      aucs=vector()
      
      for(i in 1:cvNumber){
        
        data.train <- data[data$folds!=i,]
        data.test <- data[data$folds==i,]
        
        data.train.y <- data.train$Class
        colNum <- ncol(data.train)
        data.train.x <- data.train[,c(2:(colNum-2))]
        
        glm.fit=naiveBayes(Class~.,data=data.train[,c(1:(colNum-2))])
        # p<-predict(glm.fit,type='raw')
        # qplot(seq(-2,2,length=80),sort(p),col="response")
        if (dir.exists("model")){
          
          # if (dir.exists(sprintf("model/%s",model))){
            print("exist")
          # }
          # else{
          #   dir.create(sprintf("model/%s",model))
          # }
        } 
        else{
          dir.create("model")
          # dir.create(sprintf("model/%s",model))
        }
        # saveRDS(glm.fit,sprintf("model/%s_%s_cv_%s_%s_%s.rds",Type,model,fileTileVecs[fi],i,ti))
        
       
        rfNormal.predictions<-predict(glm.fit, data.test[,c(2:(colNum-2))],type = "raw")[,2]
        cat("rfNormal.predictions\n")
        print(rfNormal.predictions)
        
        data[folds==i,]$score=rfNormal.predictions
        ##使用caret包中的confusionMatrix()计算各种指标
        rf.class <- as.integer(rfNormal.predictions > 0.5)
        rf.class = mapvalues(rf.class,from=c("0","1"),to=c("F","T"))
        
        confMat=table(rf.class,data.test$Class)
        cat("confMat of rf:\n")
        print(confMat)
        
        if(nrow(confMat)<2)
        {
          cat("%%%$$$###\n")
          next
        }
        
        #precisionList[[i]] <- rfNormal.predictions[,2]
        #labelList[[i]] <- data.test$Class
        # precisionList[[testAUCNumber]] <- rfNormal.predictions
        # labelList[[testAUCNumber]] <- data.test$Class
        # foldList[[testAUCNumber]] <- data[data$folds==i,]$fold
        
        testAUCNumber <- testAUCNumber+1
        
        
        # compute AUC
        rf.pred = ROCR::prediction(rfNormal.predictions,data.test$Class)
        rf.perf = ROCR::performance(rf.pred,"tpr","fpr")
        roc.rf.x=unlist(attr(rf.perf,"x.values"))
        roc.rf.y=unlist(attr(rf.perf,"y.values"))
        
        rf.auc <- ROCR::performance(rf.pred, "auc")@y.values
        aucs = c(aucs,unlist(rf.auc))
        
        precision=confMat[2,2]/(confMat[2,2]+confMat[2,1])
        recall=confMat[2,2]/(confMat[2,2]+confMat[1,2])
        acc = (confMat[1,1]+confMat[2,2])/(confMat[2,2]+confMat[1,2]+confMat[1,1]+confMat[2,1])
        sp = confMat[1,1]/(confMat[1,1]+confMat[2,1])
        mcc = (confMat[1,1]*confMat[2,2]-confMat[1,2]*confMat[2,1])/sqrt((confMat[2,2]+confMat[1,2])*(confMat[2,2]+confMat[2,1])*(confMat[1,1]+confMat[1,2])*(confMat[1,1]+confMat[2,1]))
        Fvalue = 2.0*precision*recall/(precision+recall)
        
        precisions = c(precisions,precision)
        recalls = c(recalls,recall)
        accs = c(accs,acc)
        sps = c(sps,sp)
        mccs = c(mccs,mcc)
        Fvalues = c(Fvalues,Fvalue)
      }
      
    } 
      
    
    
    if(model=="svm"){
      data$Class=mapvalues(data$Class,from=c("-1","1"),to=c("F","T"))
      
      ## 因为randomforest的因变量y需要时factor类型
      data$Class <- as.factor(data$Class)
      data.y <- data$Class
      data.x <- data[,c(-1)]
      
      # SVMArgsFile= sprintf("table/ecoli_svm_Args.csv",ti)
      SVMArgs <- read.csv(file = tuneFile[ti],header = T,sep = "\t", quote = "\"")
      print(SVMArgs)
      SVMGamma <- SVMArgs$gamma[which(SVMArgs==fileTileVecs[fi],arr.ind=T)[[1]]]
      SVMCost <- SVMArgs$cost[which(SVMArgs==fileTileVecs[fi],arr.ind=T)[[1]]]
      
      tune.cost = SVMCost
      tune.gamma = SVMGamma
      
      #cross validation
      #good: 123 125
      #set.seed(ti*123)
      set.seed(ti*123)
      
      cvNumber <- fold
      n=nrow(data)
      
      ## generate an array containing 1000 number in random, which varies from 1 to 10
      ## This is used to identify the fold number during process cross validation.
      folds=floor(cvNumber*runif(n))+1
      data$folds=folds
      data$score = 0
      ## ==========================
      ## use randomforest to do 5 cross validation
      
      accs=vector()
      precisions=vector()
      recalls = vector()
      accs = vector()
      mccs = vector()
      sns = vector()
      sps = vector()
      Fvalues=vector()
      specificitys=vector()
      aucs=vector()
      
      for(i in 1:cvNumber){
        
        data.train <- data[data$folds!=i,]
        data.test <- data[data$folds==i,]
        
        data.train.y <- data.train$Class
        colNum <- ncol(data.train)
        data.train.x <- data.train[,c(2:(colNum-2))]
        
        svmModel <- svm(x=data.train.x,y=data.train.y,kernel ="radial", gamma = tune.gamma, cost = tune.cost, probability=TRUE)
        
       if (dir.exists("model")){
          
          # if (dir.exists(sprintf("model/%s",model))){
            print("exist")
          # }
          # else{
          #   dir.create(sprintf("model/%s",model))
          # }
        } 
        else{
          dir.create("model")
          # dir.create(sprintf("model/%s",model))
        }
        # saveRDS(svmModel,sprintf("model/%s_%s_cv_%s_%s_%s.rds",Type,model,fileTileVecs[fi],i,ti))
        
        
        rfNormal.predictions<-predict(svmModel, data.test[,c(2:(colNum-2))], probability = TRUE)
        rfNormal.predictions <-attr(rfNormal.predictions,"probabilities")[,1]
        cat("rfNormal.predictions\n")
        print(rfNormal.predictions)
        
        data[folds==i,]$score=rfNormal.predictions
        ##使用caret包中的confusionMatrix()计算各种指标
        rf.class <- as.integer(rfNormal.predictions > 0.5)
        rf.class = mapvalues(rf.class,from=c("0","1"),to=c("F","T"))
        
        confMat=table(rf.class,data.test$Class)
        cat("confMat of rf:\n")
        print(confMat)
        
        if(nrow(confMat)<2)
        {
          cat("%%%$$$###\n")
          next
        }
        
        #precisionList[[i]] <- rfNormal.predictions[,2]
        #labelList[[i]] <- data.test$Class
        # precisionList[[testAUCNumber]] <- rfNormal.predictions
        # labelList[[testAUCNumber]] <- data.test$Class
        # foldList[[testAUCNumber]] <- data[data$folds==i,]$fold
        
        testAUCNumber <- testAUCNumber+1
        
        
        # compute AUC
        rf.pred = ROCR::prediction(rfNormal.predictions,data.test$Class)
        rf.perf = ROCR::performance(rf.pred,"tpr","fpr")
        roc.rf.x=unlist(attr(rf.perf,"x.values"))
        roc.rf.y=unlist(attr(rf.perf,"y.values"))
        
        rf.auc <- ROCR::performance(rf.pred, "auc")@y.values
        aucs = c(aucs,unlist(rf.auc))
        
        precision=confMat[2,2]/(confMat[2,2]+confMat[2,1])
        recall=confMat[2,2]/(confMat[2,2]+confMat[1,2])
        acc = (confMat[1,1]+confMat[2,2])/(confMat[2,2]+confMat[1,2]+confMat[1,1]+confMat[2,1])
        sp = confMat[1,1]/(confMat[1,1]+confMat[2,1])
        mcc = (confMat[1,1]*confMat[2,2]-confMat[1,2]*confMat[2,1])/sqrt((confMat[2,2]+confMat[1,2])*(confMat[2,2]+confMat[2,1])*(confMat[1,1]+confMat[1,2])*(confMat[1,1]+confMat[2,1]))
        Fvalue = 2.0*precision*recall/(precision+recall)
        
        precisions = c(precisions,precision)
        recalls = c(recalls,recall)
        accs = c(accs,acc)
        sps = c(sps,sp)
        mccs = c(mccs,mcc)
        Fvalues = c(Fvalues,Fvalue)
      }
        
    }
    if(model=="xgboost"){
      
      data$Class=mapvalues(data$Class,from=c("-1","1"),to=c(0,1))
      
      ## 因为randomforest的因变量y需要时factor类型
      data$Class <- as.factor(data$Class)
      data.y <- data$Class
      data.x <- data[,c(-1)]
      
      cat("begin to read parameters:")
      XGArgs <- read.csv(file = tuneFile[ti], header = T, sep = "\t", quote = "\"" )
      
      param = list(objective = XGArgs$objective[which(XGArgs==fileTileVecs[fi],arr.ind = T)[[1]]],
                   eval_metric = XGArgs$eval_metric[which(XGArgs==fileTileVecs[fi],arr.ind = T)[[1]]],
                   max_depth = XGArgs$max_depth[which(XGArgs==fileTileVecs[fi],arr.ind = T)[[1]]],
                   eta = XGArgs$eta[which(XGArgs==fileTileVecs[fi],arr.ind = T)[[1]]],
                   gamma = XGArgs$gamma[which(XGArgs==fileTileVecs[fi],arr.ind = T)[[1]]],
                   subsample = XGArgs$subsample[which(XGArgs==fileTileVecs[fi],arr.ind = T)[[1]]],
                   colsample_bytree = XGArgs$colsample_bytree[which(XGArgs==fileTileVecs[fi],arr.ind = T)[[1]]],
                   min_child_weight = XGArgs$min_child_weight[which(XGArgs==fileTileVecs[fi],arr.ind = T)[[1]]],
                   max_delta_step = XGArgs$max_delta_step[which(XGArgs==fileTileVecs[fi],arr.ind = T)[[1]]]
      )
      
      
      #cross validation
      #good: 123 125
      # set.seed(XGArgs$seed)
      set.seed(ti*123)
      
      cvNumber <- fold
      n=nrow(data)
      
      ## generate an array containing 1000 number in random, which varies from 1 to 10
      ## This is used to identify the fold number during process cross validation.
      folds=floor(cvNumber*runif(n))+1
      data$folds=folds
      data$score=0
      
      ## ==========================
      ## use randomforest to do 5 cross validation
      
      accs=vector()
      precisions=vector()
      recalls = vector()
      accs = vector()
      mccs = vector()
      sns = vector()
      sps = vector()
      Fvalues=vector()
      specificitys=vector()
      aucs=vector()
      
      for(i in 1:cvNumber){
        
        data.train <- data[data$folds!=i,]
        data.test <- data[data$folds==i,]
        
        data.train.y <- data.train$Class
        colNum <- ncol(data.train)
        data.train.x <- data.train[,c(2:(colNum-2))]
        
        dtune = xgb.DMatrix(data = as.matrix(data.train.x), label = as.matrix(data.train.y))
        
        xgb.valid.fit = xgb.train(data = dtune, params = param, nrounds = XGArgs$nrounds, verbose = T, maximize = F)
       if (dir.exists("model")){
          
          # if (dir.exists(sprintf("model/%s",model))){
            print("exist")
          # }
          # else{
          #   dir.create(sprintf("model/%s",model))
          # }
        } 
        else{
          dir.create("model")
          # dir.create(sprintf("model/%s",model))
        }
        # xgb.save(xgb.valid.fit,sprintf("model/%s_%s_cv_%s_%s_%s.h5",Type,model,fileTileVecs[fi],i,ti))
        
        
        rfNormal.predictions = predict(xgb.valid.fit, as.matrix(data.test[,c(2:(colNum-2))]))
        cat("rfNormal.predictions\n")
        print(rfNormal.predictions)
        
        data[folds==i,]$score=rfNormal.predictions
        ##使用caret包中的confusionMatrix()计算各种指标
        rf.class <- as.integer(rfNormal.predictions > 0.5)
        rf.class = mapvalues(rf.class,from=c("0","1"),to=c("F","T"))
        
        confMat=table(rf.class,data.test$Class)
        cat("confMat of rf:\n")
        print(confMat)
        
        if(nrow(confMat)<2)
        {
          cat("%%%$$$###\n")
          next
        }
        
        #precisionList[[i]] <- rfNormal.predictions[,2]
        #labelList[[i]] <- data.test$Class
        # precisionList[[testAUCNumber]] <- rfNormal.predictions
        # labelList[[testAUCNumber]] <- data.test$Class
        # foldList[[testAUCNumber]] <- data[data$folds==i,]$fold
        
        testAUCNumber <- testAUCNumber+1
        
        
        # compute AUC
        rf.pred = ROCR::prediction(rfNormal.predictions,data.test$Class)
        rf.perf = ROCR::performance(rf.pred,"tpr","fpr")
        roc.rf.x=unlist(attr(rf.perf,"x.values"))
        roc.rf.y=unlist(attr(rf.perf,"y.values"))
        
        rf.auc <- ROCR::performance(rf.pred, "auc")@y.values
        aucs = c(aucs,unlist(rf.auc))
        
        precision=confMat[2,2]/(confMat[2,2]+confMat[2,1])
        recall=confMat[2,2]/(confMat[2,2]+confMat[1,2])
        acc = (confMat[1,1]+confMat[2,2])/(confMat[2,2]+confMat[1,2]+confMat[1,1]+confMat[2,1])
        sp = confMat[1,1]/(confMat[1,1]+confMat[2,1])
        mcc = (confMat[1,1]*confMat[2,2]-confMat[1,2]*confMat[2,1])/sqrt((confMat[2,2]+confMat[1,2])*(confMat[2,2]+confMat[2,1])*(confMat[1,1]+confMat[1,2])*(confMat[1,1]+confMat[2,1]))
        Fvalue = 2.0*precision*recall/(precision+recall)
        
        precisions = c(precisions,precision)
        recalls = c(recalls,recall)
        accs = c(accs,acc)
        sps = c(sps,sp)
        mccs = c(mccs,mcc)
        Fvalues = c(Fvalues,Fvalue)
      }
    }
    if(model=="lightgbm"){
      Args <- read.csv(file = tuneFile[ti], header = T, sep = ",", quote = "\"")
      
      
      #cross validation
      #good: 123 125
      set.seed(ti*123)
      #set.seed(123)
      cvNumber <- fold
      n=nrow(data)
      ## generate an array containing 1000 number in random, which varies from 1 to 10
      ## This is used to identify the fold number during process cross validation.
      folds=floor(cvNumber*runif(n))+1
      data$folds=folds
      data$score=0
      ## ==========================
      ## use randomforest to do 5 cross validation
      
      accs=vector()
      precisions=vector()
      recalls = vector()
      accs = vector()
      mccs = vector()
      sns = vector()
      sps = vector()
      Fvalues=vector()
      specificitys=vector()
      aucs=vector()
      
      for(i in 1:cvNumber){
        
        data.train <- data[data$folds!=i,]
        data.test <- data[data$folds==i,]
        
        data.train.y <- data.train$Class
        data.train.y=mapvalues(data.train.y,from=c(-1),to=c(0))
        colNum <- ncol(data.train)
        data.train.x <- data.train[,c(2:(colNum-2))]#attention
        
        # lgb_weight <- (data.train.y * Args$weight[which(Args==fileTitle,arr.ind=T)[[1]]] + 1) / sum(data.train.y * Args$weight[which(Args==fileTitle,arr.ind=T)[[1]]] + 1)
        # lightgbm
        
        lgb_train <- lgb.Dataset(
          data = data.matrix(data.train.x), 
          label = data.train.y, 
          free_raw_data = TRUE
          # weight = lgb_weight
        )
        # 参数,注意是Args$learing
        fileTitle = fileTileVecs[fi]
        params <- list(
          learning_rate = Args$learningRate[which(Args==fileTitle,arr.ind=T)[[1]]],#attention
          num_leaves = as.integer(Args$numLeaves)[which(Args==fileTitle,arr.ind=T)[[1]]],
          max_depth = Args$maxDepth[which(Args==fileTitle,arr.ind=T)[[1]]],
          min_data_in_leaf = as.integer(Args$minDataInLeaf[which(Args==fileTitle,arr.ind=T)[[1]]]),
          max_bin = as.integer(Args$maxBin[which(Args==fileTitle,arr.ind=T)[[1]]]),
          #min_data_in_bin = as.integer(Args$minDataInBin),
          feature_fraction = Args$featureFraction[which(Args==fileTitle,arr.ind=T)[[1]]],
          min_sum_hessian = Args$minSumHessian[which(Args==fileTitle,arr.ind=T)[[1]]],
          lambda_l1 = Args$lambdaL1[which(Args==fileTitle,arr.ind=T)[[1]]],
          lambda_l2 = Args$lambdaL2[which(Args==fileTitle,arr.ind=T)[[1]]],
          drop_rate = Args$dropRate[which(Args==fileTitle,arr.ind=T)[[1]]],
          max_drop = as.integer(Args$maxDrop[which(Args==fileTitle,arr.ind=T)[[1]]])
        )
        # 模型
        
        lgb_mod <- lightgbm(
          data =lgb_train,
          params = params,
          nrounds = 300,
          early_stopping_rounds = 10,
          num_threads = 16,
          objective = "binary"
          
        )
        
        if (dir.exists("model")){
          
          # if (dir.exists(sprintf("model/%s",model))){
            print("exist")
          # }
          # else{
          #   dir.create(sprintf("model/%s",model))
          # }
        } 
        else{
          dir.create("model")
          # dir.create(sprintf("model/%s",model))
        }
        # lgb.save(lgb_mod,sprintf("model/%s_%s_cv_%s_%s_%s.model",Type,model,fileTileVecs[fi],i,ti))
        rfNormal.predictions <- predict(lgb_mod, as.matrix(data.test[,c(2:(colNum-2))]))
        cat("rfNormal.predictions\n")
        print(rfNormal.predictions)
        
        data[folds==i,]$score=rfNormal.predictions
        ##使用caret包中的confusionMatrix()计算各种指标
        rf.class <- as.integer(rfNormal.predictions > 0.5)
        rf.class = mapvalues(rf.class,from=c("0","1"),to=c("F","T"))
        
        confMat=table(rf.class,data.test$Class)
        cat("confMat of rf:\n")
        print(confMat)
        
        if(nrow(confMat)<2)
        {
          cat("%%%$$$###\n")
          next
        }
        
        #precisionList[[i]] <- rfNormal.predictions[,2]
        #labelList[[i]] <- data.test$Class
        # precisionList[[testAUCNumber]] <- rfNormal.predictions
        # labelList[[testAUCNumber]] <- data.test$Class
        # foldList[[testAUCNumber]] <- data[data$folds==i,]$fold
        
        testAUCNumber <- testAUCNumber+1
        
        
        # compute AUC
        rf.pred = ROCR::prediction(rfNormal.predictions,data.test$Class)
        rf.perf = ROCR::performance(rf.pred,"tpr","fpr")
        roc.rf.x=unlist(attr(rf.perf,"x.values"))
        roc.rf.y=unlist(attr(rf.perf,"y.values"))
        
        rf.auc <- ROCR::performance(rf.pred, "auc")@y.values
        aucs = c(aucs,unlist(rf.auc))
        
        precision=confMat[2,2]/(confMat[2,2]+confMat[2,1])
        recall=confMat[2,2]/(confMat[2,2]+confMat[1,2])
        acc = (confMat[1,1]+confMat[2,2])/(confMat[2,2]+confMat[1,2]+confMat[1,1]+confMat[2,1])
        sp = confMat[1,1]/(confMat[1,1]+confMat[2,1])
        mcc = (confMat[1,1]*confMat[2,2]-confMat[1,2]*confMat[2,1])/sqrt((confMat[2,2]+confMat[1,2])*(confMat[2,2]+confMat[2,1])*(confMat[1,1]+confMat[1,2])*(confMat[1,1]+confMat[2,1]))
        Fvalue = 2.0*precision*recall/(precision+recall)
        
        precisions = c(precisions,precision)
        recalls = c(recalls,recall)
        accs = c(accs,acc)
        sps = c(sps,sp)
        mccs = c(mccs,mcc)
        Fvalues = c(Fvalues,Fvalue)
      } 
    }
    

    
    precisionList[[ti]] <- data$score   #change
    labelList[[ti]] <- data$Class
    foldList[[ti]] <- data$fold
    
    
    # if(ti==1){
    #   test_precisions =precisions
    #   test_recalls = recalls
    #   test_sps =sps
    #   test_Fvalues = Fvalues
    #   test_accs = accs
    #   test_mccs = mccs
    #   test_aucs = aucs
    # }
    # else{
    test_precisions = c(test_precisions,mean(precisions))
    test_recalls = c(test_recalls,mean(recalls))
    test_sps = c(test_sps,mean(sps))
    test_Fvalues = c(test_Fvalues,mean(Fvalues))
    test_accs = c(test_accs,mean(accs))
    test_mccs = c(test_mccs,mean(mccs))
    test_aucs = c(test_aucs,mean(aucs))
    # }
    
  }
  
  cat("**********************************\n");
  
  cat(fileTileVecs[fi])
  
  cat(":\n")
  
  test_precisions_mean = mean(test_precisions)
  test_precisions_mean_format = sprintf("%.3f",test_precisions_mean)
  test_precisions_sd = sd(test_precisions)
  test_precisions_sd_format = sprintf("%.3f",test_precisions_sd)
  test_precisions_se = test_precisions_sd/sqrt(length(test_precisions))
  test_precisions_se_format = sprintf("%.3f",test_precisions_se)
  
  test_recalls_mean = mean(test_recalls)
  test_recalls_mean_format = sprintf("%.3f",test_recalls_mean)
  test_recalls_sd = sd(test_recalls)
  test_recalls_sd_format = sprintf("%.3f",test_recalls_sd)
  test_recalls_se = test_recalls_sd/sqrt(length(test_recalls))
  test_recalls_se_format = sprintf("%.3f",test_recalls_se)
  
  
  test_sps_mean = mean(test_sps)
  test_sps_mean_format = sprintf("%.3f",test_sps_mean)
  test_sps_sd = sd(test_sps)
  test_sps_sd_format = sprintf("%.3f",test_sps_sd)
  test_sps_se = test_sps_sd/sqrt(length(test_sps))
  test_sps_se_format = sprintf("%.3f",test_sps_se)
  
  
  test_Fvalues_mean = mean(test_Fvalues)
  test_Fvalues_mean_format = sprintf("%.3f",test_Fvalues_mean)
  test_Fvalues_sd = sd(test_Fvalues)
  test_Fvalues_sd_format = sprintf("%.3f",test_Fvalues_sd)
  test_Fvalues_se = test_Fvalues_sd/sqrt(length(test_Fvalues))
  test_Fvalues_se_format = sprintf("%.3f",test_Fvalues_se)
  
  
  test_accs_mean = mean(test_accs)
  test_accs_mean_format = sprintf("%.3f",test_accs_mean)
  test_accs_sd = sd(test_accs)
  test_accs_sd_format = sprintf("%.3f",test_accs_sd)
  test_accs_se = test_accs_sd/sqrt(length(test_accs))
  test_accs_se_format = sprintf("%.3f",test_accs_se)
  
  test_mccs_mean = mean(test_mccs)
  test_mccs_mean_format = sprintf("%.3f",test_mccs_mean)
  test_mccs_sd = sd(test_mccs)
  test_mccs_sd_format = sprintf("%.3f",test_mccs_sd)
  test_mccs_se = test_mccs_sd/sqrt(length(test_mccs))
  test_mccs_se_format = sprintf("%.3f",test_mccs_se)
  
  
  test_aucs_mean = mean(test_aucs)
  test_aucs_mean_format = sprintf("%.3f",test_aucs_mean)
  test_aucs_sd = sd(test_aucs)
  test_aucs_sd_format = sprintf("%.3f",test_aucs_sd)
  test_aucs_se = test_aucs_sd/sqrt(length(test_aucs))
  test_aucs_se_format = sprintf("%.3f",test_aucs_se)
  
  
  line = c(model,fileTileVecs[fi],
           paste(test_precisions_mean_format,test_precisions_sd_format, sep = "±"),
           paste(test_recalls_mean_format,test_recalls_sd_format, sep = "±"),
           paste(test_sps_mean_format,test_sps_sd_format, sep = "±"),
           paste(test_Fvalues_mean_format,test_Fvalues_sd_format, sep = "±"),
           paste(test_accs_mean_format,test_accs_sd_format, sep = "±"),
           paste(test_mccs_mean_format,test_mccs_sd_format, sep = "±"),
           test_aucs_mean)
  
  mat = rbind(mat,line)
  matt = cbind(matt,as.vector(unlist(precisionList)))
  mattt = cbind(mattt,as.vector(as.integer(unlist(precisionList)>0.5)))
}
  
matt = cbind(matt,as.vector(unlist(labelList)))
matt = cbind(matt,as.vector(unlist(foldList)))

mattt = cbind(mattt,as.vector(unlist(labelList)))
mattt = cbind(mattt,as.vector(unlist(foldList))) 

feature_name = paste(fileTileVecs,model,sep = "_")

colnames(matt) <- c(feature_name,"LabelList","foldList")
if(dir.exists("result")){
  
  write.csv(matt,file=sprintf("result/%s_stacking_%s_cv_perc.csv",Type,model),row.names = F)
  
  colnames(mat) <- c("Model","Encoding","PRE","SN","SP","F-value","ACC","MCC","AUC")
  write.csv(mat,file=sprintf("result/%s_stacking_%s_cv_perf.csv",Type,model),row.names = F)
  colnames(mattt)<-c(feature_name,"LabelList","foldList")
  write.csv(mattt,file=sprintf("result/%s_stacking_%s_cv_label.csv",Type,model),row.names = F)
}else{
  dir.create("result")
  write.csv(matt,file=sprintf("result/%s_stacking_%s_cv_perc.csv",Type,model),row.names = F)
  
  colnames(mat) <- c("Model","Encoding","PRE","SN","SP","F-value","ACC","MCC","AUC")
  write.csv(mat,file=sprintf("result/%s_stacking_%s_cv_perf.csv",Type,model),row.names = F)
  colnames(mattt)<-c(feature_name,"LabelList","foldList")
  write.csv(mattt,file=sprintf("result/%s_stacking_%s_cv_label.csv",Type,model),row.names = F)
}


  
}



##
Type = "SPNG"
 

model=Args[6]
times=as.numeric(Args[7])
fold=as.numeric( Args[8])
# fileTileVecs = c("aac","aac_pssm","aadp_pssm","aatp","ab_pssm","ac","acc","apaac","cc","cksaagp","cksaap","ctdc","ctdt","ctriad","d_fpssm","dde","dp","dp_pssm","dpc","dpc_pssm","dr","edp","eedp","gaac","gdpc","geary","gtpc","k_separated_bigrams_pssm","ksctriad","medp","moran","nmbroto","paac","pdt","pse_pssm","pssm_ac","pssm_composition","qsorder","rpssm","rpm_pssm","tpc","tpc_pssm","s_fpssm","socnumber")
fileTileVecs = c("stacking")
#fileTileVecs = c("dpc")

inputFile = vector()
tuneFile = vector()

  for (i in 1:length(fileTileVecs)){
    
    tuneFile = vector()

    for(fnum in 1:times){

      inputFile = c(inputFile,sprintf("feature_train/%s/%s_train_stacking_label.csv",fnum,Type))

      tuneFile = c(tuneFile,sprintf("table/%s_stacking_tune_%s_%s.csv",Type,model,fnum))
    }
  
    
  }


cv(Type = Type,time=times,inputFile = inputFile,fold = fold,model = model,tuneFile = tuneFile,fileTileVecs = fileTileVecs)



























