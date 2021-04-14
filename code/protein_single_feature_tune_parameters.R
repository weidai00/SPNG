#!/usr/bin/env Rscript
Args <- commandArgs()
library(pso)
library(e1071)
library(plyr)
library(ROCR)
library(caret)
#library(ggplot2)
library(dplyr)
#library(mlr)
library(parallelMap)
parallelStartSocket(3)
library(xgboost)
library(lightgbm)
library(randomForest)

tuneModel <- function(method, matrix, encoding, xData, yData){
  
  if(method=="svm"){
    tuned <- tune.svm(x=xData, y=yData,kernel ="radial", gamma = 2^seq(from = -10, to = 10, by =1), cost = 2^seq(from = -10, to = 10, by =1), probability=TRUE)
    tune.cost <- tuned$best.parameters$cost
    tune.gamma <- tuned$best.parameters$gamma
    argRow <- c("SVM", encoding, tune.cost, tune.gamma)
    matrix <- rbind(matrix, argRow)
  } else if(method=="rf"){
    timestart<-Sys.time()
    bestmtryMatrix <- tuneRF(x=xData, y=yData, ntreeTry=1000, stepFactor=1.5, improve=0.01, trace=TRUE, plot=FALSE, dobest=FALSE)
    bestmtry <- bestmtryMatrix[which.min(bestmtryMatrix[,2]),1]
    timeend<-Sys.time()
    caltime <-timeend-timestart
    argRow <- c("RF", encoding, bestmtry,difftime(timeend,timestart,units="mins"))
    matrix <- rbind(matrix, argRow)
  } else if(method=="knn"){
    tuned <- tune.knn(x = xData, y = yData, k = 1:100,tunecontrol=tune.control(sampling = "cross"), cross=10)
    tune.k <- tuned$best.parameters$k
    argRow <- c("KNN", encoding, tune.k)
    matrix <- rbind(matrix, argRow)
  }
  return(matrix)
}


tuneModel_one_grid <- function(xData,yData,params,grid){
  max_auc = 0.5
  best = 0
  for(i in 1:nrow(grid)){
    # 参数
    params[[names(grid)]] = grid[i,]
    print(params)
    # lgb_weight <- (yData * params$weight + 1) / sum(yData * params$weight + 1)
    lgb_train <- lgb.Dataset(
      data = data.matrix(xData),
      label = yData, 
      free_raw_data = TRUE
      # weight = lgb_weight
    )   
    # 交叉验证
    lgb_tr_mod <- lgb.cv(
      params,
      data = lgb_train,
      nrounds = 300,
      stratified = TRUE,
      nfold = 5,
      num_threads = 5,
      early_stopping_rounds = 10
      
    )   
    auc = unlist(lgb_tr_mod$record_evals$valid$auc$eval)[length(unlist(lgb_tr_mod$record_evals$valid$auc$eval))]
    if(auc > max_auc){
      max_auc = auc
      best = grid[i,]
    }    
  }
  cat("max", max_auc, "for search. \n")
  rm(lgb_train)
  rm(lgb_tr_mod)
  gc()
  params[[names(grid)]] = best
  return(params)
}

  tuneModel_pso <- function(xData,yData,learningRate, numLeaves,maxDepth, minDataInLeaf, maxBin, featureFraction, minSumHessian, lambdaL1, lambdaL2, dropRate, maxDrop)
  {
    ### lgb_weight <- (lgb_tr$Class *as.integer(weight) + 1) / sum(lgb_tr$Class * as.integer(weight) + 1)
     lgb_train <- lgb.Dataset(
         data = data.matrix(xData),
        label = yData, 
         free_raw_data = TRUE,
         ##weight = lgb_weight
    )
    
    # 参数
    params <- list(
         objective = 'binary',
         metric = 'auc',
         learning_rate = learningRate,
         num_leaves = as.integer(numLeaves),
         max_depth = as.integer(maxDepth),
         min_data_in_leaf = as.integer(minDataInLeaf),
         max_bin = as.integer(maxBin),                              
         feature_fraction = featureFraction,
         min_sum_hessian = minSumHessian,
         lambda_l1 = lambdaL1,
         lambda_l2 = lambdaL2,
         drop_rate = dropRate,
         max_drop = as.integer(maxDrop)
    )

    # cross validation
    lgb_tr_mod <- lgb.cv(
         params,
         data = lgb_train,
         nrounds = 300,
         stratified = TRUE,
         nfold = 10,
         num_threads = 5,
         early_stopping_rounds = 10
    )
       ##return AUC value for this iteration
         return(unlist(lgb_tr_mod$record_evals$valid$auc$eval)[length(unlist(lgb_tr_mod$record_evals$valid$auc$eval))])
  }       

xgboost<-function(data){
  colNum <- ncol(data)-1
  # 把原始训练集中的Class列交换到第一列（如果原始???据是从weka转来???那么class在最后一列）
  if(names(data)[length(data)] == "Class")
  {
    names<- names(data)
    newNames <- names[1:colNum]
    newNames <- c("Class",newNames)
    data <- data[,newNames]
  }
  subcolset <- names(data)
  data$Class <- mapvalues(data$Class,from=c("-1","1"),to=c(0,1))
  data$Class <- mapvalues(data$Class,from=c("F","T"),to=c(0,1))
  ## 因为randomforest的因变量y需要时factor类型
  data$Class <- as.factor(data$Class)
  
  data.x <- data[,c(-1)]
  data.y <- data$Class
  
  dtune = xgb.DMatrix(data = as.matrix(data.x), label = as.matrix(data.y))
  
  best.param = list()
  best.seed = 0
  best.auc = 0
  best.auc.index = 0
  
  ### XGBoost random grid search
  for (iter in 1:10){
    
    param = list(objective = 'binary:logistic',
                 eval_metric = 'auc',
                 max_depth = sample(4:8, 1),
                 eta = round(runif(1, 0.01, 0.03), 4),
                 gamma = round(runif(1, 0.0, 0.2), 4),
                 subsample = round(runif(1, 0.6, 0.9), 4),
                 colsample_bytree = round(runif(1, 0.5, 0.8), 4),
                 min_child_weight = sample(1:40, 1),
                 max_delta_step = sample(0:10, 1)
    )
    seed.number = sample.int(1000, 1)[[1]]
    set.seed(seed.number)
    cat("Iteration", iter, "for random grid search. \n")
    cv = xgb.cv(params = param, data = dtune, nfold = 5, nrounds = 500, verbose = F, early.stopping.round = 10, maximize = T)
    max.auc = max(cv$evaluation_log$test_auc_mean)
    max.auc.index = which.max(cv$evaluation_log$test_auc_mean)
    
    if (max.auc > best.auc){
      best.auc = max.auc
      best.auc.index = max.auc.index
      best.seed = seed.number
      best.param = param
      
    }
    cat("", sep = "\n\n")
  }
  best.param$nrounds = best.auc.index
  best.param$seed = best.seed
  best.param$auc = best.auc
  
  
  return(best.param)
  
  # write.table(best.param, file=sprintf("../table/XGBOOSTs_tunedArgs_%s.csv",), sep="\t", row.names = F, col.names = T)
}



 #fileVecs = c("aac","aac_pssm","aadp_pssm","aatp","ab_pssm","ac","acc","apaac","cc","cksaagp","cksaap","ctdc","ctdt","ctriad","d_fpssm","dde","dp","dp_pssm","dpc","dpc_pssm","dr","edp","eedp","gaac","gdpc","geary","gtpc","k_separated_bigrams_pssm","ksctriad","medp","moran","nmbroto","paac","pse_pssm","pssm_ac","pssm_composition","qsorder","rpssm","rpm_pssm","tpc","s_fpssm","socnumber")
#fileVecs = c("aac","dr","dde","qsorder","paac","ctdc","pssm_composition","pse_pssm","rpssm","d_fpssm")
#fileVecs = c("aac","dpc","dde","dp","dr","qsorder","paac","ctriad","ctdt","ctdc","geary","apaac","rpssm","dp_pssm","pse_pssm","pssm_composition")
#fileVecs = c("aac","aac_pssm","aadp_pssm","aatp","ab_pssm","ac","acc","apaac","cc","cksaagp","cksaap","ctdc","ctdt","ctriad","d_fpssm","dde","dp","dp_pssm","dpc","dpc_pssm","dr","edp","eedp","gaac","gdpc","geary","gtpc","k_separated_bigrams_pssm","ksctriad","medp","moran","nmbroto","paac","pse_pssm","pssm_ac","pssm_composition","qsorder","rpssm","rpm_pssm","s_fpssm","socnumber")
#fileVecs = c("aac_pssm","aadp_pssm","aatp","ab_pssm","d_fpssm","dp_pssm","dpc_pssm","edp","eedp","k_separated_bigrams_pssm","medp","pse_pssm","pssm_ac","pssm_composition","rpssm","rpm_pssm","s_fpssm")
#fileVecs = c("aac","dr","dde","qsorder","paac","ctdc","aac_pssm","aadp_pssm","aatp","ab_pssm","d_fpssm","dp_pssm","dpc_pssm","edp","eedp","k_separated_bigrams_pssm","medp","pse_pssm","pssm_ac","pssm_composition","rpssm","rpm_pssm","s_fpssm")
#fileVecs = c("aac_pssm","ab_pssm","d_fpssm","dp_pssm","dpc_pssm","edp","eedp","k_separated_bigrams_pssm","pse_pssm","pssm_ac","pssm_composition","rpssm","rpm_pssm","s_fpssm")
fileVecs = c("paac","qsorder","ctriad","ctdt","tpc","pse_pssm","pssm_ac","aatp")
fileTileVecs = fileVecs
SVMArgs = vector()
RFArgs = vector()
KNNArgs = vector()
LightGBMArgs = vector()
xgboostArgs = vector()
Type = "SPNG"
filenum = 10
lightgbm_method = "pso"
Args = commandArgs()

for (i in Args[6]:Args[7]){


      SVMArgs = vector()
      RFArgs = vector()
      KNNArgs = vector()
      LightGBMArgs = vector()
      xgboostArgs = vector()
    for(fi in 1:length(fileTileVecs))
    {
      
      

      print(fileTileVecs[fi])
      cat("\n")
      data = read.csv(sprintf("feature_train/%s/%s_train_%s.csv",i,Type,fileTileVecs[fi]), sep=",", header = TRUE)
      colNum <- ncol(data)-1
      # 把原始训练集中的Class列交换到第一列（如果原始???据是从weka转来???那么class在最后一列）
      if(names(data)[length(data)] == "Class")
      {
        names<- names(data)
        newNames <- names[1:colNum]
        newNames <- c("Class",newNames)
        data <- data[,newNames]
      }
      subcolset <- names(data)
      
      #data$Class <- mapvalues(data$Class,from=c("-1","1"),to=c("F","T"))
      ## 因为randomforest的因变量y需要时factor类型
      #data$Class <- as.factor(data$Class)
      
      data.x <- data[,c(-1)]
      data.y <- data$Class
      data.y_lightgbm=mapvalues(data.y,from=c(-1,1),to=c(0,1))
      data.y = as.factor(data.y)
      tune_params <- list(
        objective = 'binary',
        metric = 'auc'
       
      )
      SVMArgs <- tuneModel(method = "svm", matrix = SVMArgs, encoding = fileTileVecs[fi], xData = data.x, yData = data.y)
      RFArgs <- tuneModel(method = "rf", matrix = RFArgs, encoding = fileTileVecs[fi], xData = data.x, yData = data.y)
      KNNArgs <- tuneModel(method = "knn", matrix = KNNArgs, encoding = fileTileVecs[fi], xData = data.x, yData = data.y)



      if(lightgbm_method=="pso"){
        cat("pso\n")
        lower=c(2^(-10),20,5,2,32,0.5,0,0,0,0,1)
        upper=c(0.9,800,10,32,1024,1,0.02,0.01,0.01,1,100)

        psoobj <- psoptim(rep(NA,11),function(x)tuneModel_pso(data.x,data.y_lightgbm,x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11]),
        lower=c(2^(-10),20,5,2,32,0.5,0,0,0,0,1),upper=c(0.9,800,10,32,1024,1,0.02,0.01,0.01,1,100),control=list(fnscale =-1,trace=1,REPORT=1,trace.stats=TRUE,v.max=0.2*(upper-lower)))
        
        column =psoobj$par
        argRow <- c("LightGBM", fileTileVecs[fi], column)
      }

      
      if(lightgbm_method=="one_grid"){
        cat("one_grid\n")
        tune_params = tuneModel_one_grid(data.x,data.y_lightgbm,tune_params,expand.grid(learning_rate = seq(2^(-10),2,0.05)))
        tune_params = tuneModel_one_grid(data.x,data.y_lightgbm,tune_params,expand.grid(num_leaves = seq(20, 800, 50)))
        tune_params = tuneModel_one_grid(data.x,data.y_lightgbm,tune_params,expand.grid(max_depth = seq(3,10,1)))
        tune_params = tuneModel_one_grid(data.x,data.y_lightgbm,tune_params,expand.grid(min_data_in_leaf = 2 ^ (1:6)))
        tune_params = tuneModel_one_grid(data.x,data.y_lightgbm,tune_params,expand.grid(max_bin = 2 ^ (5:10)))
        tune_params = tuneModel_one_grid(data.x,data.y_lightgbm,tune_params,expand.grid(feature_fraction = seq(.5, 1, .02)))
        tune_params = tuneModel_one_grid(data.x,data.y_lightgbm,tune_params,expand.grid(min_sum_hessian = seq(0, .02, .001)))
        tune_params = tuneModel_one_grid(data.x,data.y_lightgbm,tune_params,expand.grid(lambda_l1 = seq(0, .01, .002)))
        tune_params = tuneModel_one_grid(data.x,data.y_lightgbm,tune_params,expand.grid(lambda_l2 = seq(0, .01, .002)))
        tune_params = tuneModel_one_grid(data.x,data.y_lightgbm,tune_params,expand.grid(drop_rate = seq(0, 1, .1)))
        tune_params = tuneModel_one_grid(data.x,data.y_lightgbm,tune_params,expand.grid(max_drop = seq(1, 100, 2)))
        argRow <- c("Lightgbm", fileTileVecs[fi], tune_params$learning_rate, tune_params$num_leaves,tune_params$max_depth,tune_params$min_data_in_leaf,tune_params$max_bin,tune_params$feature_fraction,tune_params$min_sum_hessian,tune_params$lambda_l1,tune_params$lambda_l2,tune_params$drop_rate,tune_params$max_drop)
      
      }

      
      #LightGBMArgs <- rbind(LightGBMArgs, argRow)
      
      
      
      # 
      colnames(RFArgs) <- c("Model","Encoding","mtry","time")
      write.table(RFArgs, file=sprintf("table/%s_tune_rf_%s.csv",Type,i), sep="\t", row.names = F, col.names = T)
      colnames(KNNArgs) <- c("Model","Encoding","k")
      write.table(KNNArgs, file=sprintf("table/%s_tune_knn_%s.csv",Type,i),sep="\t",row.names = F, col.names = T)
      colnames(SVMArgs) <- c("Model","Encoding","cost","gamma")
      write.table(SVMArgs,file=sprintf("table/%s_tune_svm_%s.csv",Type,i), sep="\t", row.names = F, col.names = T)
     # colnames(LightGBMArgs) <- c("Model","Encoding","learningRate", "numLeaves", "maxDepth", "minDataInLeaf", "maxBin", "featureFraction", "minSumHessian", "lambdaL1", "lambdaL2", "dropRate", "maxDrop")
     # write.table(LightGBMArgs, file=sprintf("table/%s_tune_lightgbm_%s.csv",Type,i), sep=",", row.names = F)
      
      tune_params= xgboost(data)
      argRow<-c("xgboost",fileTileVecs[fi],tune_params$nrounds,tune_params$objective,tune_params$eval_metric,tune_params$max_depth,tune_params$eta,tune_params$gamma,
               tune_params$subsample,tune_params$colsample_bytree,tune_params$min_child_weight,tune_params$max_delta_step,tune_params$seed,tune_params$auc)
      xgboostArgs = rbind(xgboostArgs,argRow)
      colnames(xgboostArgs)<-c("Model","Encoding","nrounds","objective","eval_metric","max_depth","eta","gamma","subsample","colsample_bytree","min_child_weight","max_delta_step","seed","auc")
      write.table(xgboostArgs, file=sprintf("table/%s_tune_xgboost_%s.csv",Type,i), sep="\t", row.names = F, col.names = T)
    }


}





