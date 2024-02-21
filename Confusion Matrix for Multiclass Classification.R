###Confusion Matrix###
#    A confusion matrix presents a table layout of the different outcomes of the prediction 
#    and results of a classification problem and helps visualize its outcomes. It plots
#    a table of all the predicted and actual values of a classifier.
#

###Confusion Matrix for Multiclass Classification

#Accuracy : 
#   The proportion of total number of predictions that were correct.

#Formula : Accuracy = (TP+TN)/(TP+FP+TN+FN)

#Precision :
#   The proportion of correct positive preditions.

#Formula : True positive/True Positive+False Positive

#Recall : The measure of identifying True positives

#Formula : True Positive/True Positive + False Negative

#F1-score : Harmonic mean of the precision and recall of a classifier
#Formula : F1 = 2 * precision * recall/precision + recall
--------------------------------------------------------------------------------------
install.packages("caret")
install.packages("dplyr")
install.packages("mlbench")
install.packages("tidyr")
install.packages("e1071")
install.packages("randomForest")

library(caret)
library(dplyr)
library(mlbench)
library(tidyr)
library(e1071)
library(randomForest)

data(iris)
df <- iris

#View the class distribution, as this is a multiclass problem, we can use the 
#multi-classification data table builder
#so we're going to use these observations and the variances observations to try and predict
#what species of flower it belongs to so you can see we've got

class(iris$Species)
table(iris$Species)

#now to create a data partition and we're going to use the species the classification label
#we're going to split it so we're going to use 75% of our data for training and 25% on test 

#Splitting the data into train and test splits:

train_split_idx <- caret::createDataPartition(df$Species,
                                             p=0.75, list = FALSE)
#Here we define a split index and we are now going to use a multiclass ML model to fit the data

train <- df[train_split_idx,]
test <- df[-train_split_idx,]
str(train)

#This now creates a 75% training set for training the ML model and we're going to use the
#remaining 25% as validation data to test the model.
#we're going to use caret's train function to train a random forest model and we want to expose
#the accuracy so we're going to get the most accurate model based on species.
rf_model <- caret::train(Species~.,
                         data = train,
                         method = "rf",
                         metric = "Accuracy")
rf_model
#The model is relatively accurate.We need to validate this with a confusion matrix.The random
#forest shows that it has been trained on greater than >2 classes so this moves from a binary
#model to a multi-classification model.The function contained in the package work with binary
#and multiclassification methods.

#Using the native Confusion Matrix in Caret
#Make a prediction on the fitted model with the test data

rf_class <- predict(rf_model, newdata = test, type = "raw")
rf_class

predictions <- cbind(data.frame(train_preds = rf_class, test$Species))

#Create a confusion matrix object
cm <- caret::confusionMatrix(predictions$train_preds,predictions$test.Species)

print(cm)

library(ConfusionTableR)
mc_df <- ConfusionTableR::multi_class_cm(predictions$train_preds, predictions$test.Species)

#to get the original confusion matrix 
mc_df$confusion_matrix
------------------------------------------------------------------------------------