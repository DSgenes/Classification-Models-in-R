###Confusion Matrix###
#   Confusion Matrix is one of the Classification Matrices.It is used to evaluate performance
#   of classification algorithms.So now classification algorithms are like Naive Bayes, SBM,
#   kNN or logistic regression even random forest for that matter so how these algorithms are
#   performing can be evaluated with the help of confusion matrix.

###Confusion Matrix for Binary Classification###

library(dplyr)
library(ConfusionTableR)
library(caret)
library(tidyr)
library(mlbench)

#Load in the data
data("BreastCancer", package = "mlbench")
breast <- BreastCancer[complete.cases(BreastCancer),]
breast <- breast[,-1]
for(i in 1:9){
  breast[,i] <- as.numeric(as.character(breast[,1]))
}

#Predicting the class labels using the training dataset
#Perform train and test split on the data
train_split_idx <- caret::createDataPartition(breast$Class,
                                              p = 0.75 , list = FALSE)

train <- breast[train_split_idx,]
test <- breast[train_split_idx,]

rf_fit <- caret::train(Class~., data = train , method = "rf")

#Make predictions to expose class labels
preds <- predict(rf_fit, newdata = test, type = "raw")

predicted <- cbind(data.frame(class_preds = preds), test)
predicted

#Binary Confusion Matrix Data Frame
bin_cm <- ConfusionTableR::binary_class_cm(predicted$class_preds,
                                           predicted$class)
#Get the record level data
bin_cm$record_level_cm
glimpse(bin_cm$record_level_cm)

###Visualising the Confusion Matrix###
ConfusionTableR::binary_visualiseR(train_labels = predicted$class_preds,
                                   truth_labels = predicted$class,
                                   class_label1 = "Benign",
                                   class_label2 = "Malignant",
                                   quadrant_col1 = "red",
                                   quadrant_col2 = "green",
                                   custom_title = "Breast Cancer Confusion
                                                   Matrix",
                                   text_col = "black",
                                   cm_stat_size = 1.2)
---------------------------------------------------------------------------------------------