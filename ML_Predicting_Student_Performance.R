rm(list = ls())
#Team 2-5

'''
The dataset being analyzed contains real-world data about student performance in Math and Portugese classes from two Portuguese secondary schools. 
Mark reports and questionaries were used to compile the data. Our model analyzes different factors such as 
parent occupation, study time, number of absences, ect. to predict student performance.
The objective of the model is to identify failing students early and provide additional assistance, tutoring, ext.
'''
##Link to the final presentation: https://docs.google.com/presentation/d/1-HuE26hCWLJqLN5ttMeTKMWeu-TDMIiIRahWLt-K5-c/edit?usp=sharing

#######################
#Prepare data
#######################
##Commented link: https://archive.ics.uci.edu/ml/datasets/student+performance

d1=read.table("student-mat.csv",sep=",",header=TRUE)
d2=read.table("student-por.csv",sep=",",header=TRUE
d3=rbind(d1,d2,by=c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))

#d3 = d3[,-c(31,32)] # delete G1, G2
old_G3 = d3[,31]
newG3 = rep("Pass",length(old_G3))
newG3[old_G3 <= 11] <- 'Failed'

mydata = cbind(d3[,-32:-33],newG3)


#mydata = cbind(d3[,-32:-33],newG3)

mydata = na.omit(mydata)
mydata$age = as.numeric(mydata$age)
mydata$absences = as.numeric(mydata$absences)
mydata$G1 = as.numeric(mydata$G1)

######Converting characters to factors
#####Probably could have made this more concise, but only some columns needed to be converted to factors.
mydata$Medu = as.factor(mydata$Medu)
mydata$Fedu = as.factor(mydata$Fedu)
mydata$traveltime = as.factor(mydata$traveltime)
mydata$studytime = as.factor(mydata$studytime)
mydata$failures = as.factor(mydata$failures)
mydata$famrel = as.factor(mydata$famrel)
mydata$freetime = as.factor(mydata$freetime)
mydata$goout = as.factor(mydata$goout)
mydata$Dalc = as.factor(mydata$Dalc)
mydata$Walc = as.factor(mydata$Walc)
mydata$health = as.factor(mydata$health)



require(caret)
#splitting data
set.seed(5072)
trainIndex <- createDataPartition(mydata$newG3, p = .8, list = FALSE,times = 1)
trainset <- mydata[ trainIndex,]
testset  <- mydata[-trainIndex,]
str(mydata)



#########################
### treebag model########
#########################

fitControl <- trainControl(method = "oob",number = 10)
gbmFit1 <- train(newG3 ~ ., data = trainset , method = "treebag",
                 trControl = fitControl, verbose = FALSE,keepX=TRUE)
treebag.pred <- predict(gbmFit1,testset)
table.b <- table(testset$newG3, treebag.pred,dnn=c("Actual", "Predicted"))
confusionMatrix(testset$newG3, treebag.pred)
table.b
#
# #error rate
mean(testset$newG3 != treebag.pred)
# # type I error
table.b[1, 2] / sum(table.b[1, ])
# # type II error
table.b[2, 1] / sum(table.b[2, ])
# confusionMatrix(testset$newG3,treebag.pred)



################################
####SVM code####################
################################
trctrl <- trainControl(method = 'repeatedcv', number=10, repeats=3)
svm_Linear <- train(newG3 ~., data = trainset, na.action = na.exclude, method = "svmLinear", trControl = trctrl,
                    tuneLength = 10)
svm_Linear
test_pred <- predict(svm_Linear, newdata = testset)

confusionMatrix(testset$newG3,test_pred)
grid <- expand.grid(C=c(0, 0.1, 0.5, 1, 2, 4, 8, 16))

svm_Linear_Grid <- train(newG3 ~., data = trainset, na.action = na.exclude, method = "svmLinear", trControl = trctrl,
                         tuneGrid = grid, tuneLength = 10)
svm_Linear_Grid
test_pred <- predict(svm_Linear_Grid, newdata = testset)
confusionMatrix(testset$newG3, test_pred)

svm_Radial_Grid <- train(newG3 ~., data = trainset, na.action = na.exclude, method = "svmRadial", trControl = trctrl,
                         tuneLength = 10)
test_pred <- predict(svm_Radial_Grid, newdata = testset)
svm_Radial_Grid
confusionMatrix(testset$newG3, test_pred)
table.b = table(testset$newG3, test_pred)
table.b
#error rate
mean(testset$newG3 != test_pred)
# type I error
table.b[1, 2] / sum(table.b[1, ])
# type II error
table.b[2, 1] / sum(table.b[2, ])


################################
####Boosting Model##############
################################
fitControl <- trainControl(method = "repeatedcv",number = 10,repeats = 10)

gbmFit1 <- train(newG3 ~ ., data = trainset , method = "gbm", 
                 trControl = fitControl, verbose = FALSE)
boost.pred <- predict(gbmFit1,testset)
table.b <- table(testset$newG3, boost.pred,dnn=c("Actual", "Predicted"))
table.b
confusionMatrix(testset$newG3, boost.pred)

#error rate
mean(testset$newG3 != boost.pred)
# type I error
table.b[1, 2] / sum(table.b[1, ])
# type II error
table.b[2, 1] / sum(table.b[2, ])



################################
####Random Forrest Model########
################################

accuracy_fun = function(actual,predicted){
      cm = confusionMatrix(actual,predicted)
      return(cm$overall[1])
}


installIfAbsentAndLoad <- function(neededVector) {
      for(thispackage in neededVector) {
            if( ! require(thispackage, character.only = TRUE) )
            { install.packages(thispackage)}
            require(thispackage, character.only = TRUE)
      }
}
needed = c("randomForest", "pROC", "verification", "rpart")
installIfAbsentAndLoad(needed)

require(randomForest)

##Grow an initial tree
rf <- randomForest(formula=trainset$newG3 ~ .,data=trainset,ntree=500, mtry=4,
                   importance=TRUE,localImp=TRUE,replace=FALSE,na.action=na.exclude)
head(rf$predicted,25)
###Examine Error Rates for the number of trees
min.err <- min(rf$err.rate[,"OOB"])
min.err.idx <- which(rf$err.rate[,"OOB"]== min.err)
min.err.idx
rf$err.rate[min.err.idx[1],]


##Grow a better tree
#Goal: eliminate Type 1 error

cutoffs = seq(.2,.8,.05)

mtry = seq(4)

cutoffs_vector = c()
results = c()
mtry_vector = c()

for (j in mtry){
      for (i in cutoffs){
            RfLowerError <- randomForest(formula=trainset$newG3 ~ .,data=trainset,ntree=min.err.idx[1], mtry=4,
                                         importance=TRUE,localImp=TRUE,replace=FALSE,cutoff=c(i,1-i),na.action=na.exclude)
            
            ###Evaluate by scoring the test set
            prtestBest <- predict(RfLowerError, newdata=(testset))
            
            results = c(results,accuracy_fun(testset$newG3, prtestBest) )
            cutoffs_vector = c(cutoffs_vector,cutoffs[i])
            mtry_vector = c(mtry_vector,mtry[j])
      }
}

(d= data.frame(cutoffs,results,mtry))
(bestcutoff = d[which.max(d[,2]),1])
(bestresult = d[which.max(d[,2]),2])
(bestmtry = d[which.max(d[,2]),3])



#Overall best model:
rfBestModel <- randomForest(formula=trainset$newG3 ~ .,data=trainset,ntree=min.err.idx[1], mtry=bestmtry,
                            importance=TRUE,localImp=TRUE,replace=FALSE,cutoff=c(bestcutoff,1-bestcutoff),na.action=na.exclude)

prtestBest <- predict(rfBestModel, newdata=(testset))
confusionMatrix(testset$newG3, prtestBest)
saccuracy_fun(testset$newG3, prtestBest)


table.b <- table(testset$newG3, prtestBest,dnn=c("Actual", "Predicted"))
table.b
#
# #error rate
mean(testset$newG3 != prtestBest)
# # type I error
table.b[1, 2] / sum(table.b[1, ])
# # type II error
table.b[2, 1] / sum(table.b[2, ])
              



##########
####ROC###
##########
fitControl <- trainControl(method = "repeatedcv",number = 10)
gbmFit1 <- train(newG3 ~ ., data = trainset , method = "rocc",
                 trControl = fitControl,keepX=TRUE)
treebag.pred <- predict(gbmFit1,testset)
table.b <- table(testset$newG3, treebag.pred,dnn=c("Actual", "Predicted"))
confusionMatrix(testset$newG3, treebag.pred)
table.b
#
