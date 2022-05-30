library(tidyverse)
library(caret)
library(readr)
library(skimr)
library(glmnet)
library(e1071)
install.packages("glmnet")
install.packages("skimr")
library(Matrix)


#   Step 0
credit_data <- read.csv(file = "loan_default_dataset.csv", header=T)

#credit_data <- read.csv(file = "CreditQuiz.csv", header=T)

summaryStats <- skim(credit_data)
summaryStats
mean(credit_data$Emp_duration)
median(credit_data$Emp_duration)

hist(credit_data$Emp_duration,
     freq = FALSE,
     main = "Credit",
     xlab = "Emp_duration",
     ylab = "Fequency",
     las = 1,
     col = c("skyblue", "chocolate2")
)

lines(density(credit_data$Emp_duration), lwd = 4, col = "red")




hist(credit_data$Amount,credit_data$Default)

boxplot(credit_data$Emp_duration~credit_data$Default)
boxplot(credit_data$Age~credit_data$Default)


#Step 1: Partition our Data

credit_predictors_dummy <- model.matrix(Default~ ., data = credit_data)#create dummy variables expect for the response
credit_predictors_dummy<- data.frame(credit_predictors_dummy[,-1]) #get rid of intercept
credit_data <- cbind(Default=credit_data$Default, credit_predictors_dummy)


#convert from Response from integer to factor
credit_data$Default<-as.factor(credit_data$Default)
levels(credit_data$Default)<-c("notdefault","Default")

credit_data$Default 
set.seed(12) #set random seed
index <- createDataPartition(credit_data$Default, p = .8,list = FALSE)
credit_train <-credit_data[index,]
credit_test <- credit_data[-index,]

credit_model <- train(Default ~ .,
                      data = credit_train,
                      method = "glmnet",
                      trControl =trainControl(method = "cv",
                                              number = 5,
                                              classProbs = TRUE,
                                              summaryFunction = twoClassSummary),
                      metric="ROC")
credit_model               


plot(varImp(credit_model))


coef(credit_model$finalModel, credit_model$bestTune$lambda)


#Step 3
#First, get the predicted probabilities of the test data.
predprob_lasso<-predict(credit_model , credit_test, type="prob")


#Step 4 
#install.packages("ROCR")
library(ROCR)
#Get AUC and ROC curve for LASSO Model.
pred_lasso <- prediction(predprob_lasso$Default, credit_test$Default,label.ordering =c("notdefault","Default") )
perf_lasso <- performance(pred_lasso, "tpr", "fpr")
plot(perf_lasso, colorize=TRUE)

#Get the AUC
auc_lasso<-unlist(slot(performance(pred_lasso, "auc"), "y.values"))

auc_lasso
