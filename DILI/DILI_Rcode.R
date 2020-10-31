#DILI dataset
#Set working directory
setwd("E:\\Hackathon\\DILI")

#Read data
DILI_finalsplit <- read.csv("DILI_final_split.csv")

install.packages("caTools")
library(caTools)

#Partitioning into training and test sets
install.packages("caret")
library(caret)
Training_set <- subset(DILI_finalsplit, DILI_finalsplit$Status=="Training")
Test_set <- subset(DILI_finalsplit, DILI_finalsplit$Status=="Prediction")

#Feature scaling
Training_set[,4:559] <- scale(Training_set[,4:559])
Test_set[,4:559] <- scale(Test_set[,4:559])

#Feature selection using MARS
DILI_finalsplit_IDrmv <- read.csv("DILI_final_split_IDrmv.csv")
Training_set_IDrmv <- subset(DILI_finalsplit_IDrmv, DILI_finalsplit_IDrmv$Status=="Training")

#Important variables
install.packages("earth")
library(earth)
set.seed(1234)
marsModel <- earth(Response ~ ., data=Training_set_IDrmv)
ev <- evimp(marsModel)
print(ev)

#-------1. Logistic Regression----
#Fitting Logistic Regression to the Training set
set.seed(1234)
classifier <- glm(formula = Response ~ ALogp2 + MIC3 + PubchemFP566 + AMR + CrippenLogP + ATSC6p + PubchemFP431 + TopoPSA + nX + minssssC + AATSC2v + ATSC2s + SC.5,
                  family = binomial,
                  data = Training_set)
summary(classifier)

#Predict test set results
prob_pred <- predict(classifier, type = 'response', newdata = Test_set[-3])
y_pred = ifelse(prob_pred > 0.5, 1, 0)

#Making the confusion matrix
confusionMatrix(data = factor(y_pred), 
                reference = factor(Test_set$Response), 
                positive = '1', mode = "everything")

# Calculate AUC
install.packages("cvAUC")
library("cvAUC")
AUC(y_pred,Test_set$Response)

#-------2. Random forest----

#Encode target feature as factor
Training_set$Response = factor(Training_set$Response, levels = c(0,1))

#Fitting Logistic Regression to the Training set
install.packages('randomForest')
library(randomForest)
set.seed(1234)
classifier <- randomForest(Response ~ ALogp2 + MIC3 + PubchemFP566 + AMR + CrippenLogP + ATSC6p + PubchemFP431 + TopoPSA + nX + minssssC + AATSC2v + ATSC2s + SC.5,
                           data = Training_set,
                           ntree = 10)

#Predict test set results
y_pred = predict(classifier, newdata = Test_set[-3])

#Making the confusion matrix
library(caret)
confusionMatrix(data = factor(y_pred),reference = factor(Test_set$Response),positive = '1', mode = "everything")

# Calculate AUC
library("cvAUC")
AUC(y_pred,Test_set$Response)

#-------3. Linear SVM----

#Encode target feature as factor
Training_set$Response = factor(Training_set$Response, levels = c(0,1))

#Fitting Linear SVM to the Training set
set.seed(1234)
library(e1071)
classifier <- svm(formula = Response ~ ALogp2 + MIC3 + PubchemFP566 + AMR + CrippenLogP + ATSC6p + PubchemFP431 + TopoPSA + nX + minssssC + AATSC2v + ATSC2s + SC.5,
                  data = Training_set,
                  type = 'C-classification',
                  kernel = 'linear')

#Predict test set results
y_pred = predict(classifier, newdata = Test_set[-3])

#Making the confusion matrix
library(caret)
confusionMatrix(data = factor(y_pred),reference = factor(Test_set$Response), positive = '1', mode = "everything")

# Calculate AUC
library("cvAUC")
AUC(y_pred,Test_set$Response)

#-------4.kernel SVM----

#Encode target feature as factor
Training_set$Response = factor(Training_set$Response, levels = c(0,1))

#Fitting kernel SVM to the Training set
set.seed(1234)
library(e1071)
classifier <- svm(formula = Response ~ ALogp2 + MIC3 + PubchemFP566 + AMR + CrippenLogP + ATSC6p + PubchemFP431 + TopoPSA + nX + minssssC + AATSC2v + ATSC2s + SC.5,
                  data = Training_set,
                  type = 'C-classification',
                  kernel = 'radial')

#Predict test set results
y_pred = predict(classifier, newdata = Test_set[-3])

#Making the confusion matrix
library(caret)
confusionMatrix(data = factor(y_pred), 
                reference = factor(Test_set$Response),
                positive = '1', mode = "everything")

# Calculate AUC
library("cvAUC")
AUC(y_pred,Test_set$Response)
