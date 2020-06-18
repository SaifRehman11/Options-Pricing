setwd("E:/Google Drive/Saif/USC MBA/Statistics/Project")


data <- read.csv("option_train.csv")
str(data)

testing <- read.csv("option_test.csv")

summary(data)
summary(testing)

stddata <- scale(data$Value)
stddata[which(abs(stddata)>3)]



library(MASS)
cor(data)

plot(data)

boxplot(data$Value, xlab = "Value", main = "Training Value Boxplot")
boxplot(data$S, xlab = "S", main = "Training S Boxplot")
boxplot(data$K, xlab = "K", main = "Training K Boxplot")
boxplot(data$tau, xlab = "Tau", main = "Training tau Boxplot")
boxplot(data$r, xlab = "r", main = "Training r Boxplot")


boxplot(testing$S, xlab = "S", main = "Testing S Boxplot")
boxplot(testing$K, xlab = "K", main = "Testing K Boxplot")
boxplot(testing$tau, xlab = "Tau", main = "Testing tau Boxplot")
boxplot(testing$r, xlab = "r", main = "Testing r Boxplot")

library(car)
sum(is.na(data))

data2 <- data
data$BS <- as.numeric(data$BS)
data$BS <- data$BS -1
data$BS <- data$BS -1
data$BS <- data$BS * -1
data$diff <- data$S-data$K


set.seed(101) # Set Seed so that same sample can be reproduced in future also


hist(data$Value)
hist(data$S)
hist(data$K)
hist(data$tau)
hist(data$r)
hist(data$BS)


# Now Selecting 50% of data as sample from total 'n' rows of the data
sample <- sample.int(n = nrow(data), size = floor(.80*nrow(data)), replace = F)
train <- data[sample, ]
test  <- data[-sample, ]

glm1 <- glm(BS ~ S + K + tau + r, family =  binomial(link="logit"),data = data)
summary(glm1)
plot(glm1)

# Assessing Outliers
outlierTest(glm1) # Bonferonni p-value for most extreme obs
qqPlot(glm1, main="QQ Plot") #qq plot for studentized resid 
leveragePlots(glm1) # leverage plots

# Influential Observations
# added variable plots 
av.Plots(glm1)
# Cook's D plot
# identify D values > 4/(n-k-1) 
cutoff <- 4/((nrow(data)-length(glm1$coefficients)-2)) 
plot(glm1, which=4, cook.levels=cutoff)
# Influence Plot 
influencePlot(glm1,	id.method="identify", main="Influence Plot", sub="Circle size is proportial to Cook's Distance" )
qqplot(glm1)
qqline(glm1)

pred = predict.glm(glm1, newdata=test, type = "response")

library(InformationValue)
optCutOff <- optimalCutoff(test$BS, pred)[1] 
optCutOff

misclass <- misClassError(test$BS, pred, threshold = 0.5)
1-misclass

fitted.results <- ifelse(pred > 0.6499,1,0)
misClasificError <- mean(fitted.results != test$BS)
print(paste('Accuracy',1-misClasificError))


accuracy <- table(pred, test[,"BS"])
sum(diag(accuracy))/sum(accuracy)

accuracy


###Using Difference


glm2 <- glm(BS ~ diff + tau + r, family =  binomial(link="logit"),data = train)
summary(glm2)
plot(glm2)

pred = predict.glm(glm2, newdata=test, type = "response")
optCutOff <- optimalCutoff(test$BS, pred)[1] 
optCutOff

misclass <- misClassError(test$BS, pred, threshold = 0.5)
1-misclass

fitted.results <- ifelse(pred > 0.5,1,0)
misClasificError <- mean(fitted.results != test$BS)
print(paste('Accuracy',1-misClasificError))

library(pROC)
roc_obj <- roc(test$BS, pred)
plot(roc_obj)
auc(roc_obj)


#

glm3 <- glm(BS ~ (diff + tau + r)^2, family =  binomial(link="logit"),data = train)
summary(glm3)

fit <- step(glm3, direction = "both")
summary(fit)

pred = predict(fit, newdata=test, type = "response")
optCutOff <- optimalCutoff(test$BS, pred)[1] 
optCutOff

misclass <- misClassError(test$BS, pred, threshold = 0.5)
1-misclass

fitted.results <- ifelse(pred > 0.5,1,0)
misClasificError <- mean(fitted.results != test$BS)
print(paste('Accuracy',1-misClasificError))



##Using Naive Bayesian Filter

library(e1071)
# Naive_Bayes_Model=naiveBayes(BS ~ S + K + tau + r, data=train)
# #What does the model say? Print the model summary
# Naive_Bayes_Model
# summary(Naive_Bayes_Model)
# 
# #Prediction on the dataset
# NB_Predictions=predict(Naive_Bayes_Model,train)
# #Confusion matrix to check accuracy
# table(NB_Predictions,train$BS)


#Decision Trees

library(rpart)
library(rpart.plot)

fit <- rpart(BS~K+S+tau+r, data = train, method = 'class')
fit
plot(fit)
text(fit)
rpart.plot(fit)

pred = predict(fit, train, type = 'class')
pred = as.numeric(pred)-1
optCutOff <- optimalCutoff(train$BS, pred)[1] 
optCutOff

misClassError(train$BS, pred, threshold = 0.5)

fitted.results <- ifelse(pred > 0.5,1,0)
misClasificError <- mean(fitted.results != test$BS)
print(paste('Accuracy',1-misClasificError))


fit <- rpart(BS~diff+tau+r, data = train, method = 'class')
fit
plot(fit)
text(fit)
rpart.plot(fit)



pred = predict(fit, train, type = 'class')
pred = as.numeric(pred)-1
optCutOff <- optimalCutoff(train$BS, pred)[1] 
optCutOff

misClassError(train$BS, pred, threshold = 0.5)

fitted.results <- ifelse(pred > 0.5,1,0)
misClasificError <- mean(fitted.results != train$BS)
print(paste('Accuracy',1-misClasificError))



#SVM

svm1 <- svm(BS ~ diff + tau + r, data = train)
summary(svm1)
# pred = predict(svm1, train)
# optCutOff <- optimalCutoff(train$BS, pred)[1] 
# optCutOff
# misClassError(train$BS, pred, threshold = 0.5)
# fitted.results <- ifelse(pred > 0.5,1,0)
# misClasificError <- mean(fitted.results != train$BS)
# print(paste('Accuracy',1-misClasificError))

pred = predict(svm1, test)
optCutOff <- optimalCutoff(test$BS, pred)[1] 
optCutOff
misClassError(test$BS, pred, threshold = 0.5)
fitted.results <- ifelse(pred > 0.5,1,0)
misClasifiedError <- mean(fitted.results != test$BS)
print(paste('Accuracy',1-misClasifiedError))





svm1 <- svm(BS ~ K + S + tau + r, data = train)
summary(svm1)
# pred = predict(svm1, train)
# optCutOff <- optimalCutoff(train$BS, pred)[1] 
# optCutOff
# misClassError(train$BS, pred, threshold = 0.5)
# fitted.results <- ifelse(pred > 0.5,1,0)
# misClasificError <- mean(fitted.results != train$BS)
# print(paste('Accuracy',1-misClasificError))

pred = predict(svm1, test)
optCutOff <- optimalCutoff(test$BS, pred)[1] 
optCutOff
misClassError(test$BS, pred, threshold = 0.5)
fitted.results <- ifelse(pred > 0.5,1,0)
misClasifiedError <- mean(fitted.results != test$BS)
print(paste('Accuracy',1-misClasifiedError))



predfinal <- predict(fit, validation)
fitted.results <- ifelse(predfinal > 0.5,1,0)
write.csv(x = fitted.results, "bspredictions4.csv")




svm1 <- svm(BS ~ (K + S + tau + r)^2, data = train)
summary(svm1)

pred = predict(svm1, test)
optCutOff <- optimalCutoff(test$BS, pred)[1] 
optCutOff
misClassError(test$BS, pred, threshold = 0.5)
fitted.results <- ifelse(pred > 0.5,1,0)
misClasifiedError <- mean(fitted.results != test$BS)
print(paste('Accuracy',1-misClasifiedError))


#
library(MASS)

fit <- lda(BS ~ diff + tau + r, data = train)
summary(fit)

pred = predict(fit, test)
pred <- as.numeric(pred$class)-1
optCutOff <- optimalCutoff(test$BS, pred)[1] 
optCutOff
misClassError(test$BS, pred, threshold = 0.5)
fitted.results <- ifelse(pred > 0.5,1,0)
misClasifiedError <- mean(fitted.results != test$BS)
print(paste('Accuracy',1-misClasifiedError))



#kNN 
library(class)
fit <- knn(train = train, test = test, cl=train$BS)
summary(fit)
table(test$BS, fit)
misClasificError <- mean(fit != test$BS)
print(paste('Accuracy',1-misClasificError))

predfinal <- predict(fit, validation)


#RandomForest
library(randomForest)
fit <- randomForest(BS ~ K + S + tau + r, train, ntree=500)
summary(fit)
pred= predict(fit,train)
optCutOff <- optimalCutoff(train$BS, pred)[1] 
optCutOff
misClassError(train$BS, pred, threshold = 0.5)
fitted.results <- ifelse(pred > 0.5,1,0)
misClasificError <- mean(fitted.results != train$BS)
print(paste('Accuracy',1-misClasificError))

pred= predict(fit,test)
optCutOff <- optimalCutoff(test$BS, pred)[1] 
optCutOff
misClassError(test$BS, pred, threshold = 0.5)
fitted.results <- ifelse(pred > 0.5,1,0)
misClasificError <- mean(fitted.results != test$BS)
print(paste('Accuracy',1-misClasificError))

##predicting on final data
validation <- read.csv("option_test.csv")
validation$diff <- validation$S - validation$K


fit <- randomForest(BS ~ diff+ tau + r, train, ntree=8)
summary(fit)
plot(fit)

pred= predict(fit,test)
optCutOff <- optimalCutoff(test$BS, pred)[1] 
optCutOff
misClassError(test$BS, pred, threshold = 0.5)
fitted.results <- ifelse(pred > 0.5,1,0)
misClasificError <- mean(fitted.results != test$BS)
print(paste('Accuracy',1-misClasificError))

predfinal <- predict(fit, validation)
fitted.results <- ifelse(predfinal > 0.5,1,0)
write.csv(x = fitted.results, "bspredictions3.csv")




fit <- randomForest(BS ~ diff+ tau + r, train, ntree=6)
summary(fit)
plot(fit)

pred= predict(fit,test)
optCutOff <- optimalCutoff(test$BS, pred)[1] 
optCutOff
misClassError(test$BS, pred, threshold = 0.5)
fitted.results <- ifelse(pred > 0.5,1,0)
misClasificError <- mean(fitted.results != test$BS)
print(paste('Accuracy',1-misClasificError))



##nnet

library(nnet)
fit <- nnet(BS ~ diff + tau + r, train, size = 20)

pred= predict(fit,test)
optCutOff <- optimalCutoff(test$BS, pred)[1] 
optCutOff
misClassError(test$BS, pred, threshold = 0.5)
fitted.results <- ifelse(pred > 0.5,1,0)
misClasificError <- mean(fitted.results != test$BS)
print(paste('Accuracy',1-misClasificError))

##xgboost 

library(xgboost)
traintest <- train[,c(4,5,7)]
trainlabel <- train$BS
testtest <- test[,c(4,5,7)]
testlabel <- test$BS
xgb_train <- xgb.DMatrix(data=as.matrix(traintest), label=trainlabel)
xgb_test <- xgb.DMatrix(data=as.matrix(testtest), label=testlabel)

#fit <- xgboost()
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)
xgbcv <- xgb.cv( params = params, data = xgb_train, nrounds = 100, nfold = 5, showsd = T, stratified = T,  maximize = F)
min(xgbcv$test.error.mean)

xgb1 <- xgb.train (params = params, data = xgb_train, nrounds = 79, watchlist = list(val=xgb_test,train=xgb_train),  maximize = F , eval_metric = "error")
pred <- predict (xgb1,xgb_test)
optCutOff <- optimalCutoff(test$BS, pred)[1] 
optCutOff
misClassError(test$BS, pred, threshold = 0.469)
fitted.results <- ifelse(pred > 0.5,1,0)
misClasificError <- mean(fitted.results != test$BS)
print(paste('Accuracy',1-misClasificError))

summary(xgb1)

#xgb with caret
library(caret)
TrainControl <- trainControl( method = "repeatedcv", number = 10, repeats = 4)

fit<- train(as.factor(BS) ~ diff + tau + r, data = train, method = "xgbTree", trControl = TrainControl)
plot(fit)
summary(fit)
pred <- predict (fit,test)
pred = as.numeric(pred)-1

misClassError(test$BS, pred)
misClasificError <- mean(pred != test$BS)
print(paste('Accuracy',1-misClasificError))






#######Value Prediction

#GLM
glm1 <- glm(Value ~ S+K + tau + r,data = train)
summary(glm1)
plot(glm1)

# Assessing Outliers
outlierTest(glm1) # Bonferonni p-value for most extreme obs
qqPlot(glm1$residuals, main="QQ Plot") #qq plot for studentized resid 
leveragePlots(glm1) # leverage plots

# Influential Observations
# added variable plots 
av.Plots(glm1)
# Cook's D plot
# identify D values > 4/(n-k-1) 
cutoff <- 4/((nrow(data)-length(glm1$coefficients)-2)) 
plot(glm1, which=4, cook.levels=cutoff)
# Influence Plot 
influencePlot(glm1,	id.method="identify", main="Influence Plot", sub="Circle size is proportial to Cook's Distance" )

pred = predict.glm(glm1, newdata=train)
summary(glm1)
library(forecast)
accuracy(pred, train$Value)

pred = predict.glm(glm1, newdata=test)
accuracy(pred, test$Value)
plot(pred)
plot(pred,test$Value)

#
glm1 <- glm(Value ~ diff + tau + r,data = train)
summary(glm1)

pred = predict.glm(glm1, newdata=train)
accuracy(pred, train$Value)

pred = predict.glm(glm1, newdata=test)
accuracy(pred, test$Value)

#
glm1 <- glm(Value ~ (diff + tau + r)^2,data = train)
summary(glm1)
fit <- step(glm1, direction = "both")
summary(fit)
pred = predict.glm(fit, newdata=train)
accuracy(pred, train$Value)

pred = predict.glm(fit, newdata=test)
accuracy(pred, test$Value)

plot(fit)


#SVM
svm1 <- svm(Value ~ diff + tau + r, data = train)
summary(svm1)
pred = predict(svm1, train)
accuracy(pred, train$Value)


pred = predict(svm1, test)
accuracy(pred, test$Value)

plot(svm1$residuals)


#randomForest

fit <- randomForest(Value ~ diff + tau + r, train, ntree=200)
summary(fit)
pred= predict(fit,train)
accuracy(pred, train$Value)


pred = predict(fit, test)
accuracy(pred, test$Value)
plot(fit)


#xgb with caret
TrainControl <- trainControl( method = "repeatedcv", number = 10, repeats = 4)

fit<- train(Value ~ diff + tau + r, data = train, method = "xgbLinear", trControl = TrainControl,verbose = FALSE)
plot(fit)
summary(fit)
pred <- predict (fit,train)
accuracy(pred, train$Value)


pred <- predict (fit,test)
accuracy(pred, test$Value)

##predicting on final data
validation <- read.csv("option_test.csv")
validation$diff <- validation$S - validation$K

predfinal <- predict (fit,validation)
write.csv(x = predfinal, "valuepredictions.csv")
