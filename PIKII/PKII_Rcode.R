#PKII dataset
#Set working directory
setwd("E:\\Hackathon\\PKII")

#Read data
finalsplit <- read.csv("PKII_final_split.csv")

install.packages("caTools")
library(caTools)

#Partitioning into training and test sets
install.packages("caret")
library(caret)
Training_set <- subset(finalsplit, finalsplit$Status=="Training")
Test_set <- subset(finalsplit, finalsplit$Status=="Test")

#Feature scaling
Training_set[,4:630] <- scale(Training_set[,4:630])
Test_set[,4:630] <- scale(Test_set[,4:630])

#Feature selection using MARS
finalsplit_IDrmv <- read.csv("PKII_final_split_IDrmv.csv")
Training_set_IDrmv <- subset(finalsplit_IDrmv, finalsplit_IDrmv$Status=="Training")

#Encode target feature as factor
Training_set_IDrmv$Response = factor(Training_set_IDrmv$Response, levels = c(0,1))

#Important variables
install.packages("earth")
library(earth)
set.seed(1234)
marsModel <- earth(Response ~ nAcid + ., data=Training_set_IDrmv)
ev <- evimp(marsModel)
print(ev)

#-------1. Logistic Regression----
#Fitting Logistic Regression to the Training set
set.seed(1234)
classifier <- glm(formula = Response ~ TopoPSA + nAcid + IC1 + ATSC4c + FMF + C1SP2 + ATSC5i + ATSC2c + AATS0p + nRotBt,
                  family = binomial,
                  data = Training_set)
summary(classifier)

#Predict test set results
prob_pred <- predict(classifier, type = 'response', newdata = Test_set[-3])
y_pred = ifelse(prob_pred > 0.5, 1, 0)

#Making the confusion matrix
library(caret)
confusionMatrix(data = factor(y_pred), 
                reference = factor(Test_set$Response),
                positive = '1', mode = "everything")
# Calculate AUC
install.packages("cvAUC")
library("cvAUC")
AUC(y_pred,Test_set$Response)

#-------2. Naive Bayes-----

#Encode target feature as factor
Training_set$Response = factor(Training_set$Response, levels = c(0,1))

#Fitting Naive Bayes to the Training set
library(e1071)
set.seed(1234)
classifier <- naiveBayes(formula = Response ~ TopoPSA + nAcid + IC1 + ATSC4c + FMF + C1SP2 + ATSC5i + ATSC2c + AATS0p + nRotBt,
                         data = Training_set)

#Predict test set results
Test_set$Response = factor(Test_set$Response, levels = c(0,1))
y_pred = predict(classifier, newdata = Test_set[-3])

#Making the confusion matrix
library(caret)
confusionMatrix(data = factor(y_pred), 
                reference = factor(Test_set$Response),
                positive = '1', mode = "everything")

# Calculate AUC
library("cvAUC")
AUC(y_pred,Test_set$Response)

#-------3. kNN----

#Encode target feature as factor
Training_set$Response = factor(Training_set$Response, levels = c(0,1))
Test_set$Response = factor(Test_set$Response, levels = c(0,1))

#Fitting kNN to the Training set
#install.packages("class")
library(caret)
library(tidyverse)
library(class)
classifier <- train(Response ~ TopoPSA + nAcid + IC1 + ATSC4c + FMF + C1SP2 + ATSC5i + ATSC2c + AATS0p + nRotBt,
                    data = Training_set,
                    method = "knn",
                    tuneLength = 20)

#Predict test set results
Test_set$Response = factor(Test_set$Response, levels = c(0,1))
y_pred = predict(classifier, newdata = Test_set[-3])

#Making the confusion matrix
confusionMatrix(data = factor(y_pred), 
                reference = factor(Test_set$Response),
                positive = '1', mode = "everything")

# Calculate AUC
library("cvAUC")
AUC(y_pred,Test_set$Response)
