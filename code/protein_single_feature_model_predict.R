library(e1071)
library(ROCR)
library("randomForest")
library(plyr)
library(xgboost)
library(lightgbm)
#1.每一折的模型对测试集预测一下，N折的预测分数取均值
#2.预测分数和标签做
#Rscipt protein_single_feature_model_predict.R svm 10
Args = commandArgs()
predict_model<-function(Type,model,times){
  # model = "rf"
  fileTileVecs = c("paac","qsorder","ctriad","ctdt","tpc","pse_pssm","aatp")
  # fileTileVecs = c("aac","aac_pssm","aadp_pssm","aatp","ab_pssm","ac","acc","apaac","cc","cksaagp","cksaap","ctdc","ctdt","ctriad","d_fpssm","dde","dp","dp_pssm","dpc","dpc_pssm","dr","edp","eedp","gaac","gdpc","geary","gtpc","k_separated_bigrams_pssm","ksctriad","medp","moran","nmbroto","paac","pdt","pse_pssm","pssm_ac","pssm_composition","qsorder","rpssm","rpm_pssm","tpc","tpc_pssm","s_fpssm","socnumber")
  # fileTileVecs_small = c("ab_pssm","d_fpssm","eedp","aac","dpc","rpm_pssm","k_separated_bigrams_pssm","ctdc")
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
    precision = confMat[2,2]/(confMat[2,2]+confMat[2,1])
    recall = confMat[2,2]/(confMat[2,2]+confMat[1,2])
    acc = (confMat[1,1]+confMat[2,2])/(confMat[2,2]+confMat[1,2]+confMat[1,1]+confMat[2,1])
    sp = confMat[1,1]/(confMat[1,1]+confMat[2,1])
    mcc = (confMat[1,1]*confMat[2,2]-confMat[1,2]*confMat[2,1])/sqrt((confMat[2,2]+confMat[1,2])*(confMat[2,2]+confMat[2,1])*(confMat[1,1]+confMat[1,2])*(confMat[1,1]+confMat[2,1]))
    Fvalue = 2.0*precision*recall/(precision+recall)
    #</confu>


    line = c(fileTileVecs[fi],
             sprintf("%.3f",precision),
             sprintf("%.3f",recall),
             sprintf("%.3f",sp),
             sprintf("%.3f",Fvalue),
             sprintf("%.3f",acc),
             sprintf("%.3f",mcc))
    # line = c(fileTileVecs[fi],
    #                   sprintf("%.3f",sn))
    print(line)
    mat = rbind(mat,line)
    matt = cbind(matt,unlist(precisionList))
    mattt = cbind(mattt,as.integer(unlist(precisionList)>0.5))
  
  
  }
  
  feature_name = paste(fileTileVecs,model,sep = "_")
  
  colnames(matt) <- c("Class",feature_name)
  write.table(matt,file=sprintf("result/%s_single_feature_%s_test_perc.csv",Type,model),sep=",",row.names = F)
  
  ## 存储指标数据的表???
  colnames(mat) <- c("Encoding","PRE","SN","SP","F-value","ACC","MCC")
  # colnames(mat) <- c("Encoding","SN")
  write.csv(mat,file=sprintf("result/%s_single_feature_%s_test_perf.csv",Type,model),row.names = F)
  #write.table(mat,file="ecoli_QSO.csv",sep="\t",row.names = F, col.names = T)
  
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




