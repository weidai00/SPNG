

Type = "SPNG"

#stacking_test
test_data = vector()
data1 = vector()
data2 = vector()
data3 = vector()
data4 = vector()
data5 = vector()
data6 = vector()
xgboost = vector()
svm = vector()
rf = vector()
nb = vector()
knn = vector()
lightgbm = vector()




data1 =read.csv(sprintf("result/%s_single_feature_xgboost_test_label.csv",Type),header = T)
xgboost = data1[,-1]



data2 =read.csv(sprintf("result/%s_single_feature_svm_test_label.csv",Type),header = T)
svm = data2[,-1]



data3 =read.csv(sprintf("result/%s_single_feature_rf_test_label.csv",Type),header = T)
rf = data3[,-1]


data4 =read.csv(sprintf("result/%s_single_feature_nb_test_label.csv",Type),header = T)
Class = data4$Class
nb = data4[,-1]




data5 =read.csv(sprintf("result/%s_single_feature_knn_test_label.csv",Type),header = T)
knn = data5[,-1]


data6 =read.csv(sprintf("result/%s_single_feature_lightgbm_test_label.csv",Type),header = T)

lightgbm = data6[,-1]


test_data = cbind(Class,nb,rf,svm,xgboost,lightgbm,knn)

write.csv(test_data,sprintf("feature_test/%s_test_stacking_label.csv",Type),row.names = F)


