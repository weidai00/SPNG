library(e1071)
library(ROCR)
library("randomForest")
library(plyr)
library(xgboost)
library(lightgbm)

#Rscipt protein_single_feature_model_predict.R svm 10
Args = commandArgs()
predict_model<-function(Type,model,times){

  fileTileVecs = c("paac","qsorder","ctriad","ctdt","tpc","pse_pssm","aatp")
 
  dp.data = read.csv(sprintf("feature_test/%s_test_paac.csv",Type),header = T)
  matt =vector() 
  mat=vector()
  mattt = vector()
  matt = cbind(dp.data$Class,matt)
  mattt = cbind(dp.data$Class,mattt)
  for (fi in 1:length(fileTileVecs)){
    line = vector()
   precisionList = list()
   precisionList_sum=0
   predict_cv = list()
  
  for (ti in 1:times){
  

      testData = read.csv(sprintf("feature_test/%s_test_%s.csv",Type,fileTileVecs[fi]),header = T)
      #<model>
      if(model=="lr"){
      modelDir = sprintf("model/%s_%s_%s_%s.rds",Type,fileTileVecs[fi],model,ti)
      trainModel = readRDS(modelDir)
      rfNormal.predictions<-predict(trainModel, testData[,c(-1)],type='response')
      }
      if(model=="svm"){
        modelDir = sprintf("model/%s_%s_%s_%s.rds",Type,fileTileVecs[fi],model,ti)
        trainModel = readRDS(modelDir)
        rfNormal.predictions <- predict(trainModel, testData[,c(-1)], probability = TRUE)
        rfNormal.predictions <- attr(rfNormal.predictions,"probabilities")[,1]
      }
      if(model=="nb"){
        modelDir =sprintf("model/%s_%s_%s_%s.rds",Type,fileTileVecs[fi],model,ti)
        trainModel = readRDS(modelDir)
        rfNormal.predictions<-predict(trainModel, testData[,c(-1)],type='raw')[,2]
      }
      if(model=="rf"){
        modelDir = sprintf("model/%s_%s_%s_%s.rds",Type,fileTileVecs[fi],model,ti)
        trainModel = readRDS(modelDir)
        rfNormal.predictions<-predict(trainModel,type="prob",newdata=testData[,c(-1)])[,2]
      }
      if(model=="lightgbm"){
        modelDir = sprintf("model/%s_%s_%s_%s.model",Type,fileTileVecs[fi],model,ti)
        trainModel = lgb.load(modelDir)
        rfNormal.predictions<-predict(trainModel, as.matrix(testData[,c(-1)]))
      }
      if(model=="xgboost"){
        modelDir = sprintf("model/%s_%s_%s_%s.h5",Type,fileTileVecs[fi],model,ti)
        trainModel = xgb.load(modelDir)
        rfNormal.predictions<-predict(trainModel, as.matrix(testData[,c(-1)]))
      }
      #</model>
      pro = rfNormal.predictions
      # if(cv==1){
      #   precisionList <- pro/10
      # }
      # else{
      #   precisionList <- pro/10+precisionList
      # }  
     
       if(ti==1){
            precisionList  = pro/times
        }else{
            precisionList = precisionList+pro/times
     
        }
    
  
    
    # precisionList_sum = precisionList_sum+(precisionList[[ti]]/10)
    
  }
    
    #<confu>
    results.number <- as.integer(precisionList > 0.5)
    rf.class <- mapvalues(results.number, from=c("0","1"), to=c("F","T"))
    testData$Class <- mapvalues(testData$Class,from=c("-1","1"),to=c("F","T"))

    confMat<- table(rf.class,testData$Class)
    # colnames(predictResultCSV) <- c("Pred","Real")
    # 
    # dep.330 <- as.data.frame(predictResultCSV)
    # dep.330.pos <- dep.330[dep.330[,2]=="T",]
    # dep.330.neg <- dep.330[dep.330[,2]=="F",]
    # dep.bal <- rbind(dep.330.pos,dep.330.neg)
    # 先预测值，后真实???
    # confMat <- table(dep.bal[,1],dep.bal[,2])
    
    
    
    
    # sn = sum(precisionList>0.5)/length(precisionList)
   
   
   
    matt = cbind(matt,unlist(precisionList))
    mattt = cbind(mattt,as.integer(unlist(precisionList)>0.5))
  
  
  }
  
  feature_name = paste(fileTileVecs,model,sep = "_")
  
  colnames(matt) <- c("Class",feature_name)
  write.table(matt,file=sprintf("result/%s_single_feature_%s_test_perc.csv",Type,model),sep=",",row.names = F)
  

  
  
  colnames(mattt)<-c("Class",feature_name)
  
  write.table(mattt,file=sprintf("result/%s_single_feature_%s_test_label.csv",Type,model),sep=",",row.names = F)

}
# 




Type = "SPNG"
model = Args[6]
times = as.numeric(Args[7])
predict_model(Type=Type,model =model,times=times)
# predict_model("svm")
# model_Args =  commandArgs()
# predict_model(model_Args[6])




