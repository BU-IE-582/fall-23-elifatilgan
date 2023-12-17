library(fastDummies)
library(data.table)
library(kknn)
library(dplyr)
library(rpart)
library(caret)
library(mlr)
library(rattle)
library(ranger)
library(gbm)
setwd("~/Desktop/HW2")
set.seed(456)

#Student Data
student=read.table("student-por.csv",sep=";",header=TRUE)
# Convert binary variables to 0 and 1
student$school <- ifelse(student$school == "GP", 0, 1)
student$sex <- ifelse(student$sex == "F", 0, 1)
student$address <- ifelse(student$address == "U", 0, 1)
student$famsize <- ifelse(student$famsize == "LE3", 0, 1)
student$Pstatus <- ifelse(student$Pstatus == "T", 0, 1)
student$schoolsup <- ifelse(student$schoolsup == "no", 0, 1)
student$famsup <- ifelse(student$famsup == "no", 0, 1)
student$paid <- ifelse(student$paid == "no", 0, 1)
student$activities <- ifelse(student$activities == "no", 0, 1)
student$nursery <- ifelse(student$nursery == "no", 0, 1)
student$higher <- ifelse(student$higher == "no", 0, 1)
student$internet <- ifelse(student$internet == "no", 0, 1)
student$romantic <- ifelse(student$romantic == "no", 0, 1)


student$target <- student$G3
student$G3 <- NULL  # Remove the old column.

# Convert some columns to factors
student[, sapply(student, is.character)] <- lapply(student[, sapply(student, is.character)], as.factor)
student_train <- student[1:449,]
student_test <- student[450:649,]
rownames(student_test) <- NULL

head(student,5)



#Specific dataset for Nearest Neighbor since it can't handle categorical data.
student_nn <- copy(student)
# Specify the columns to be transformed to be used in k-NN
categorical_columns <- c("Mjob", "Fjob", "reason", "guardian")
# Create dummy variables using one-hot encoding
student_nn <- dummy_cols(student_nn, select_columns = categorical_columns)
# Remove the original categorical columns
student_nn <- student_nn[, !(names(student_nn) %in% categorical_columns)]
# Standardize the features excluding the target variable.
student_nn[, !names(student_nn) %in% c("target")] <- scale(student_nn[, !names(student_nn) %in% c("target")])
#Turn the regression data into classification data:
student_nn$target<- as.factor(ifelse(student_nn$target >= mean(student_nn$target),"high","low"))
#Split data into train and test subsets so that we have 200 test data. (More train data would not hurt so it is okay that it is not exactly 200.)
student_nn_train <- student_nn[1:449,]
student_nn_test <- student_nn[450:649,]
rownames(student_nn_test) <- NULL
head(student_nn,5)

card=read.csv("creditcard.csv",header=TRUE)
card$target <- card$Class
card$Class <- NULL
head(card,5)
barplot(table(card$target),col="pink")
set.seed(123)
splitIndex <- createDataPartition(card$target, p = 0.8, list = FALSE, times = 1)
card_train <- card[splitIndex, ]
card_test <- card[-splitIndex, ]
rownames(card_train) <- NULL
rownames(card_test) <- NULL

# Initialize empty data frames to store sampled data
card_train_sampled <- data.frame()
card_test_sampled <- data.frame()

# Sample within each class for training data
for (class_label in unique(card_train$target)) {
  class_data <- subset(card_train, target == class_label)
  sampled_rows <- class_data[sample(nrow(class_data), 5000 * (sum(card$target==class_label)/nrow(card)),replace=TRUE), ]
  card_train_sampled <- rbind(card_train_sampled, sampled_rows)
}

# Sample within each class for testing data
for (class_label in unique(card_test$target)) {
  class_data <- subset(card_test, target == class_label)
  sampled_rows <- class_data[sample(nrow(class_data), 2000 * (sum(card$target==class_label)/nrow(card)),replace=TRUE), ]
  card_test_sampled <- rbind(card_test_sampled, sampled_rows)
}

rownames(card_train_sampled) <- NULL
rownames(card_test_sampled) <- NULL


train_class_percentages <- prop.table(table(card_train_sampled$target == 1)) * 100
test_class_percentages <- prop.table(table(card_test_sampled$target == 1)) * 100
result_table <- data.frame(
  Dataset = c("Train Data","Test Data"),
  `Class=1 Ratio` = c(train_class_percentages[2], test_class_percentages[2])
)
result_table
card_nn_train <- copy(card_train_sampled)
card_nn_train[, !names(card_nn_train) %in% c("target")] <- scale(card_nn_train[, !names(card_nn_train) %in% c("target")])
card_nn_test <- copy(card_test_sampled)
card_nn_test[, !names(card_nn_test) %in% c("target")] <- scale(card_nn_test[, !names(card_nn_test) %in% c("target")])
rownames(card_nn_test) <- NULL
rownames(card_nn_train) <- NULL

card_train_sampled$target <- as.factor(card_train_sampled$target+1)
card_test_sampled$target <- as.factor(card_test_sampled$target+1)
head(card_nn_train,5)
mobile <- read.csv("train.csv",header=TRUE)
mobile$target <- as.factor(mobile$price_range+1)
mobile$price_range <- NULL
mobile_train <- mobile[1:1800,]
mobile_test <- mobile[1801:2000,]
rownames(mobile_test) <- NULL
head(mobile,5)
mobile_nn_train <- copy(mobile_train)
mobile_nn_train[, !names(mobile_nn_train) %in% c("target")] <- scale(mobile_nn_train[, !names(mobile_nn_train) %in% c("target")])
mobile_nn_test <- copy(mobile_test)
mobile_nn_test[, !names(mobile_nn_test) %in% c("target")] <- scale(mobile_nn_test[, !names(mobile_nn_test) %in% c("target")])
head(mobile_nn_train,5)

musk <- read.table("clean1.data",sep=",")
clean1_names <- read.table("clean1.names", header = FALSE,sep=":")
colnames(musk) <- clean1_names$V1
musk$molecule_name <- as.factor(musk$molecule_name)
musk$conformation_name <- as.factor(musk$conformation_name)
musk$target <- as.factor(musk$Class+1)
musk$Class <- NULL
musk <- musk[sample(nrow(musk)),]
musk_train <- musk[1:276,]
musk_test <- musk[277:476,]
rownames(musk_test) <- NULL
head(musk,5)

#Specific dataset for Nearest Neighbor since it can't handle categorical data.
musk_nn <- copy(musk)
# Specify the columns to be transformed to be used in k-NN
categorical_columns <- c("molecule_name", "conformation_name")
# Create dummy variables using one-hot encoding
musk_nn <- dummy_cols(musk_nn, select_columns = categorical_columns)
# Remove the original categorical columns
musk_nn <- musk_nn[, !(names(musk_nn) %in% categorical_columns)]
# Standardize the features excluding the target variable.
musk_nn[, !names(musk_nn) %in% c("target")] <- scale(musk_nn[, !names(musk_nn) %in% c("target")])
#Split data into train and test subsets so that we have 200 test data. (More train data would not hurt so it is okay that it is not exactly 200.)
musk_nn_train <- musk_nn[1:276,]
musk_nn_test <- musk_nn[277:476,]
rownames(musk_nn_test) <- NULL
head(musk_nn_train,5)

activity_train=read.csv("train_activity.csv",header=TRUE)
activity_test = read.csv("test_activity.csv",header=TRUE)
activity_train$target <- as.factor(activity_train$Activity)
activity_train$Activity <- NULL
activity_test$target <- as.factor(activity_test$Activity)
activity_test$Activity <- NULL
activity_train <- activity_train[sample(nrow(activity_train),3600),]
activity_test <- activity_test[sample(nrow(activity_test),1800),]

head(activity_train,5)
k_NN <- function(train,test,k_list) {
  #Perform 10-fold cross validation using the train data.
  d_list <- c(1,2)
  cv_values <- data.frame()
  for(k in seq_along(k_list)) {
    for(d in seq_along(d_list)) {
      fit <- cv.kknn(target~ .,train,kcv=10,k=k_list[k],distance=d_list[d])
      y <- fit[[1]][,1]
      y_hat <- fit[[1]][,2]
      misclassification_error <- mean(y_hat != y)
      cv_values[k,d] <- misclassification_error
    }
  }
  #Find missclassification errors on the train set to choose the best model:
  min_value <- min(cv_values)
  min_indices <- which(cv_values == min_value, arr.ind = TRUE)[1,]
  best_k <- k_list[min_indices[["row"]]]
  best_d <- d_list[min_indices[["col"]]]
  if(best_d ==1) {
    cat("The chosen parameters are: Distance method= Manhattan, k=",best_k,"\n")
  }else{
    cat("The chosen parameters are: Distance method= Euclidean, k=",best_k,"\n")
  }
  #Train the final model and use test data to create confusion matrix. 
  final_model <- kknn(target~ .,train,test,k=best_k,distance=best_d)
  confusion_matrix <- table(test$target, final_model$fitted.values)
  acc <- sum(diag(confusion_matrix))/sum(confusion_matrix)
  cat("Accuracy on test data: ",acc,"\n")
  return(cv_values)
}
CART <- function(train,test,type) {
  if(type=="r") {
    task <- makeRegrTask(data = train, target = "target")
    learner <- makeLearner("regr.rpart", cp = 0)
  } else {
    if(is.factor(train$target)== FALSE) {
      mean <- mean(train$target)
      train$target<- as.factor(ifelse(train$target >= mean,"high","low"))
      test$target<- as.factor(ifelse(test$target >= mean,"high","low"))
    }
    task <- makeClassifTask(data = train, target = "target")
    learner <- makeLearner("classif.rpart", cp = 0)
  }
  # Define the search space for tuning minbucket
  param_set <- makeParamSet(makeDiscreteParam("minbucket", values=c(5,8,11,14,17)))
  #Set up the cross-validation
  inner <- makeResampleDesc("CV", iters = 10)
  # Custom learner training function to set minsplit
  configureMinSplit <- function(learner, params) {
    minsplit_val <- 2 * params$minbucket
    learner$par.vals <- c(learner$par.vals, list(minsplit = minsplit_val))
    return(learner)
  }
  res <- tuneParams(learner, task, resampling = inner, par.set = param_set, control = makeTuneControlGrid())
  # Print the best parameter configuration
  print(res)
  best_minbucket<- res[["x"]][["minbucket"]]
  
  final_model <- rpart(target ~ ., data = train, cp = 0, minbucket=best_minbucket, minsplit = 2 * best_minbucket)
  new_predictions <- predict(final_model, newdata = test)
  if(type== "r") {
    rmse <- sqrt(mean((test$target - new_predictions)^2))
    cat("Final model's RMSE value on the Test Data is: ",rmse)
  } else {
    new_predictions <- apply(new_predictions, 1, function(row) {
      ifelse(row[1] > row[2], "high", "low")
    })
    confusion_matrix <- table(test$target, new_predictions)
    acc <- sum(diag(confusion_matrix))/sum(confusion_matrix)
    cat("Accuracy on test data: ",acc,"\n")
  }
  final_model[["acc"]] <- acc
  return(final_model)
}
RF <- function(train,test,type) {
  #Creating a regression or classification task depending on the 'target' variable:
  if(type=="r") {
    task <- makeRegrTask(data = train, target = "target")
    learner <- makeLearner(
      "regr.ranger",  
      num.trees = 500,
      min.node.size = 5
    )
    
  } else {
    if(is.factor(train$target)== FALSE) {
      mean <- mean(train$target)
      train$target<- as.factor(ifelse(train$target >= mean,"high","low"))
      test$target<- as.factor(ifelse(test$target >= mean,"high","low"))
    }
    task <- makeClassifTask(data = train, target = "target")
    learner <- makeLearner(
      "classif.ranger",  
      num.trees = 500,
      min.node.size = 5
    )
  }
  #Defining the search space for tuning mtry
  param_set <- makeParamSet(
    makeIntegerParam("mtry", lower = 1, upper = min(100,ncol(train) - 1))
  )
  #Setting up the cross-validation
  inner <- makeResampleDesc("CV", iters = 10)
  res <- tuneParams(learner, task, resampling = inner, par.set = param_set, control = makeTuneControlGrid())
  print(res)
  best_mtry <- res[["x"]][["mtry"]]
  final_model <- ranger(target ~ ., data = train, mtry= best_mtry, min.bucket=5,num.trees = 500)
  new_predictions <- predict(final_model,data = test)
  if(type=="r") {
    rmse <- sqrt(mean((test$target - new_predictions[["predictions"]])^2))
    cat("Final model's RMSE value on the Test Data is: ",rmse,"\n")
  } else {
    confusion_matrix <- table(test$target, new_predictions[["predictions"]])
    acc <- sum(diag(confusion_matrix))/sum(confusion_matrix)
    cat("Accuracy on test data: ",acc,"\n")
  }
  final_model[["acc"]] <- acc
  return(final_model)
}
GradBM <- function(train,test,type) {
  n_folds=10
  fitControl=trainControl(method = "cv",
                          number = n_folds,
  )
  gbmGrid=expand.grid(n.minobsinnode = 10,
                      n.trees = c(50,100,150,200,250), 
                      shrinkage = c(0.01,0.05,0.1,0.15,0.2),
                      interaction.depth = c(1,2,3,4,5))
  if(type !="r") {
    if(is.factor(train$target)== FALSE) {
      mean <- mean(train$target)
      train$target<- as.factor(ifelse(train$target >= mean,"high","low"))
      test$target<- as.factor(ifelse(test$target >= mean,"high","low"))
    }
  }
  
  gbm_fit=caret::train(target ~ ., data = train, 
                       method = "gbm", 
                       trControl = fitControl, metric=ifelse(is.factor(train$target),'Accuracy',"RMSE"),
                       tuneGrid = gbmGrid,
                       verbose=F) 
  best_n_trees <- gbm_fit[["bestTune"]][["n.trees"]]
  best_depth <- gbm_fit[["bestTune"]][["interaction.depth"]]
  best_shrinkage <- gbm_fit[["bestTune"]][["shrinkage"]]
  new_predictions <- predict(gbm_fit,test)
  if(type=="r") {
    rmse <- sqrt(mean((test$target - new_predictions)^2))
    cat("Final model's RMSE value on the Test Data is: ",rmse,"\n")
  } else {
    confusion_matrix <- table(test$target, new_predictions)
    acc <- (sum(diag(confusion_matrix))/sum(confusion_matrix))[1]
    cat("Accuracy on test data: ",acc,"\n")
  }
  return(gbm_fit)
}

k_values <- c(1,3,5,7,9)
result_nn <- k_NN(activity_train,activity_test,k_values)
plot(k_values, result_nn$V2, type = 'l', col = 'blue', xlab = '# Neighbor', ylab = 'misclassification_error',xlim=c(1,10),ylim=c(min(result_nn),max(result_nn)))
lines(k_values, result_nn$V1, col = 'red')
legend('bottomright', legend = c('Manhattan', 'Euclidean'), col = c('red', 'blue'), lty = 1)

result_cart <- CART(activity_train,activity_test,"c")
cat("Chosen minbucket: ",result_cart[["control"]][["minbucket"]],"\n")
cat("Accuracy on test data: ",result_cart[["acc"]],"\n")
fancyRpartPlot(result_cart)

result_rf <- RF(activity_train,activity_test,"c")
cat("Chosen mtry: ",result_rf[["mtry"]],"\n")
cat("Accuracy on test data: ",result_rf[["acc"]],"\n")
result_rf[["confusion.matrix"]]

result_gbm <- GradBM(activity_train,activity_test,"c")
plot(result_gbm)
































































































