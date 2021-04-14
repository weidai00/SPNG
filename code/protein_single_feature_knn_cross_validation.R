#!/usr/bin/env Rscript
# source ("T3ROC-style.R")
# source ("utility.R")

library(e1071)
library(ROCR)
library("randomForest")
library(plyr)
#library(caret)



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
mattt= vector()
mat = vector()
matt = vector()
# fileTileVecs = c("AAC-PSSM","AADP-PSSM","AATP","AB-PSSM","D-FPSSM","DPC-PSSM","DP-PSSM","EDP","EEDP","k-separated-bigrams-PSSM","MEDP","Pse-PSSM","PSSM-AC","PSSM-composition","RPM-PSSM","RPSSM","S-FPSSM","TPC","AAC","DPC","MOREAU","MORAN","CTDC","CTDT","CTDD","GEARY","CTRIAD","QSO","SOCN","APAAC","PAAC","BLOSUM");
fileTileVecs = c("paac","qsorder","ctriad","ctdt","tpc","pse_pssm","aatp")
fileNum=10
Type = "SPNG"
cvNum = 5
# fileTileVecs = c("pssm_composition")
for(fi in 1:length(fileTileVecs))          #代表多少个csv文件
{
  
 
  testNumber <- 1
  testAUCNumber <- 1;
  testAUCNumber_new <- 1;

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

  for (ti in 1:fileNum)
  {
    inputDir = c("feature_train")
    
    #32个特征编码方法
    fileVecs = sprintf("%s/%s/%s_train_%s.csv",inputDir,ti,Type,fileTileVecs[fi])
    fileVecs1 = sprintf("feature_test/%s_test_%s.csv",Type,fileTileVecs[fi])


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
    # testData = read.csv(fileVecs1,header = T)
    colNum <- ncol(data)-1
    # 把原始训练集中的Class列交换到第一列（如果原始数据是从weka转来???,那么class在最后一列）
    if(names(data)[length(data)] == "Class")
    {
      names<- names(data)
      newNames <- names[1:colNum]
      newNames = c("Class",newNames)
      data <- data[,newNames]
    }

    subcolset <- names(data)
    data$Class=mapvalues(data$Class,from=c("-1","1"),to=c("F","T"))

    ## 因为randomforest的因变量y需要时factor类型
    #data$Class <- as.factor(data$Class)
    data.y <- data$Class
    data.x <- data[,c(-1)]

    Args <- read.csv(file = sprintf("table/%s_tune_knn_%s.csv",Type,ti), header = T, sep = "\t", quote = "\"" )
    kValue <- Args$k[which(Args==fileTileVecs[fi],arr.ind=T)[[1]]]
    
    cat("Best k value: \n")
    print(kValue)

    #cross validation
    #good: 123 125
    set.seed(ti*123)
    # set.seed(123)

    cvNumber <- cvNum
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
      data.train.x <- data.train[,c(2:(colNum-2))]#change

      library (class)
      myknn <- knn(data.train.x,data.test[,c(2:(colNum-2))],data.train.y,k = kValue, prob=TRUE)#change
      #<test>
      # myknn_test<-knn(data.train.x,testData[,c(-1)],data.train.y,k=kValue,prob=TRUE)
      # test_pred<-myknn_test
      # test_pred.label<-as.array(test_pred)
      # test_pred<-attr( test_pred,"prob")
      # test_pred[( test_pred.label == "F")] <- c(1)- test_pred[( test_pred.label == "F")] 
      # print(test_pred)
      #</test>
      
      
      rfNormal.predictions<-myknn
      cat("rfNormal.predictions\n")
      print(rfNormal.predictions)
      rfNormal.predictions.label <- as.array(rfNormal.predictions)
      cat("rfNormal.predictions.label\n")
      print(rfNormal.predictions.label == "F")
      rfNormal.predictions<-attr(rfNormal.predictions,"prob")
      rfNormal.predictions[(rfNormal.predictions.label == "F")] <- c(1)-rfNormal.predictions[(rfNormal.predictions.label == "F")] 
      cat("rfNormal.predictions\n")
      print(rfNormal.predictions)

      data[data$folds==i,]$score = rfNormal.predictions#change
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
    precisionList[[ti]] <- data$score   #change
    labelList[[ti]] <- data$Class
    foldList[[ti]] <- data$fold


    test_precisions = c(precisions,test_precisions)
    test_recalls = c(recalls,test_recalls)
    test_sps =c(sps,test_sps)
    test_Fvalues = c(Fvalues,test_Fvalues)
    test_accs = c(accs,test_accs)
    test_mccs = c(mccs,test_mccs)
    test_aucs = c(aucs,test_accs)
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


  line = c("KNN",fileTileVecs[fi],
    paste(test_precisions_mean_format,test_precisions_sd_format, sep = "±"),
    paste(test_recalls_mean_format,test_recalls_sd_format, sep = "±"),
    paste(test_sps_mean_format,test_sps_sd_format, sep = "±"),
    paste(test_Fvalues_mean_format,test_Fvalues_sd_format, sep = "±"),
    paste(test_accs_mean_format,test_accs_sd_format, sep = "±"),
    paste(test_mccs_mean_format,test_mccs_sd_format, sep = "±"))

  mat = rbind(mat,line)
  matt = cbind(matt,unlist(precisionList))
  mattt = cbind(mattt,as.integer(unlist(precisionList)>0.5))
}
matt = cbind(matt,unlist(labelList))
matt = cbind(matt,unlist(foldList))

mattt = cbind(mattt,unlist(labelList))
mattt = cbind(mattt,unlist(foldList))


feature_name = paste(fileTileVecs,"knn",sep = "_")

colnames(matt) <-c(feature_name,"LabelList","foldList")
write.table(matt,file=sprintf("result/%s_single_feature_knn_cv_perc.csv",Type),sep=",",row.names = F, col.names = T)
## 存储指标数据的表???
colnames(mat) <- c("Model","Encoding","PRE","SN","SP","F-value","ACC","MCC")
#write.table(mat,file="train_performance/T6SE_P_70_multiple_train_singleSVMs_cv5_performance_sd_total_100.csv",sep=",",row.names = F, col.names = T)
write.csv(mat,file=sprintf("result/%s_single_feature_knn_cv_perf.csv",Type),row.names = F)

colnames(mattt)<-c(feature_name,"LabelList","foldList")
write.csv(mattt,file = sprintf("result/%s_single_feature_knn_cv_label.csv",Type),row.names = F)
