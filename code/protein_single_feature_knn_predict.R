
library(e1071)
library(ROCR)
library("randomForest")
library(plyr)


#Rscript protein_single_feature_knn_predict.R
Type = "SPNG"
dp.data = read.csv(sprintf("feature_test/%s_test_paac.csv",Type),header = T)


auc.legends=vector()
matt = cbind(dp.data$Class,matt)
mattt = cbind(dp.data$Class,mattt)
fileTileVecs = c("paac","qsorder","ctriad","ctdt","tpc","pse_pssm","aatp")
times = 10
fold = 5
for(fi in 1:length(fileTileVecs))          #代表多少个csv文件
{
  precisionList = list()
 precisionList_sum=0
  testNumber <- 1
  testAUCNumber <- 1;
  testAUCNumber_new <- 1;

  test_accs=vector()
  test_precisions=vector()
  test_recalls = vector()
  test_accs = vector()
  test_mccs = vector()
  test_sns = vector()
  aucs = vector()
  test_sps = vector()
  test_Fvalues=vector()
  test_specificitys=vector()
  test_aucs=vector()
  test_pred = list()
  precisionList = list()
  labelList = list()
  foldList = list()

  for (ti in 1:times)
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
    testData = read.csv(fileVecs1,header = T)
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

    library (class)
    # myknn <- knn(data.x,data.test[,c(-1)],data.train.y,k = kValue, prob=TRUE)#change
    #<test>
    myknn_test<-knn(data.x,testData[,c(-1)],data.y,k=kValue,prob=TRUE)
    test_pred<-myknn_test
    test_pred.label<-as.array(test_pred)
    test_pred<-attr( test_pred,"prob")
    test_pred[( test_pred.label == "F")] <- c(1)- test_pred[( test_pred.label == "F")] 
    # print(test_pred)
    
    #</test>
    
    pro = test_pred
    if(ti==1){
      precisionList <- pro/times
    }
    else{
      precisionList <- pro/times+precisionList
    }
     # precisionList_sum = precisionList_sum+(precisionList[[ti]]/10)  
  }

    cat("**********************************\n");


    cat(fileTileVecs[fi])
  
    cat(":\n")
    ##使用caret包中的confusionMatrix()计算各种指标
    rf.class <- as.integer(precisionList> 0.5)
    rf.class = mapvalues(rf.class,from=c("0","1"),to=c("F","T"))
    testData$Class = mapvalues(testData$Class,from=c("-1","1"),to=c("F","T"))
    

    #precisionList[[i]] <- rfNormal.predictions[,2]
    #labelList[[i]] <- data.test$Class
    # precisionList[[testAUCNumber]] <- rfNormal.predictions
    # labelList[[testAUCNumber]] <- data.test$Class
    # foldList[[testAUCNumber]] <- data[data$folds==i,]$fold

    testAUCNumber <- testAUCNumber+1
    

    # compute AUC
    

    

  


    
    matt = cbind(matt,unlist(precisionList))
    mattt = cbind(mattt,as.integer(unlist(precisionList)>0.5))


}


feature_name = paste(fileTileVecs,"knn",sep = "_")

colnames(matt) <-c("Class",feature_name)
write.table(matt,file=sprintf("result/%s_single_feature_knn_test_perc.csv",Type),sep=",",row.names = F, col.names = T)


colnames(mattt)<-c('Class',feature_name)
write.csv(mattt,file = sprintf("result/%s_single_feature_knn_test_label.csv",Type),row.names = F)
