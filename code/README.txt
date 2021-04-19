
- feature_train
训练集特征文件

- feature_test
测试集特征文件

- model
SPNG模型

- result
模型预测结果文件

-table
模型调参结果文件


未知序列通过diffuser生成具体的特征后，需要按顺序运行以下的命令去预测

	##单一模型的预测
	Rscirpt protein_single_feature_model_predict.R svm 10 
	Rscirpt protein_single_feature_model_predict.R nb 10 
	Rscirpt protein_single_feature_model_predict.R rf 10 
	Rscirpt protein_single_feature_model_predict.R xgboost 10 
	Rscirpt protein_single_feature_model_predict.R lightgbm 10 
	Rscript protein_single_feature_knn_predict.R
	
	##组合单一模型
	Rscript protein_stacking_model_predict.R
	
	##集成模型的预测
	Rscript protein_stacking_model_predict.R svm 10

其他文件或者是调参代码，或者是交叉验证代码
