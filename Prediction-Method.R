### Several different functions for prediction methods including
### random forest, xgboost and sparse additive model.


library(ranger)
library(xgboost)
library(SAM)


### random.forest
### Function: Random Forest Model fitting with a data splitting option,
###           use out-of-bag error for hyper-parameter tuning
### Input: Y: n by 1 outcome vector
###        X: n by p_x covariate matrix
###        data.split: logic, to do data splitting or not, default by FALSE
###        split.prop: Proportion of data to be used for inference(size of subsample A1), default by 0.5
###        num.trees: Number of trees in random forest,default by 500
###        mtry: Number of covariates to consider at each split, default by p/3
###        max.depth: Maximal depth of each tree, with default value 0 referring to unlimited depth
###        min.node.size: Minimal size of each leaf node
###        MSE.thol: A large value used for the start of hyper-parameter selection, default by 1e12
### Output: predicted.values: Predictions of the outcome Y
###         forest.model: Random forest model object
###         params: Best hyper-parameters selected by minimizing out-of-bag error
###         A1.ind: Indices of data in subsample A1, available when data.split=TRUE
###         A2.ind: Indices of data in subsample A2, available when data.split=TRUE
###         MSE.oob: Minimal out-of-bag error using the best hyper-parameters
random.forest <- function(Y,X,data.split=FALSE,split.prop=0.5,num.trees=500,mtry=NULL,max.depth=0,min.node.size=5,MSE.thol=1e12) {
  X <- as.matrix(X); Y <- as.matrix(Y)
  n <- NROW(X); p <- NCOL(X)
  if (is.null(mtry)) mtry <- round(p/3)
  
  Data <- data.frame(cbind(Y, X))
  names(Data) <- c("Y", paste("X", 1:p, sep = ""))
  # search grid
  params.grid <- expand.grid(
    num.trees = num.trees,
    mtry = mtry,
    max.depth = max.depth,
    min.node.size = min.node.size
  )
  
  
  # data splitting or not
  if (data.split) {
    # split the data into two parts A1 and A2, A2 for building model, A1 for inference
    n.A1 <- round(split.prop*n)
    A1.ind <- 1:n.A1
    A2.ind <- setdiff(1:n,A1.ind)
    Data.A1 <- Data[A1.ind, ]
    Data.A2 <- Data[A2.ind, ]
    Data.train <- Data.A2
    Data.test <- Data.A1
  } else {
    Data.train <- Data.test <- Data
  }


  forest <- NULL;
  MSE.oob <- MSE.thol
  params <- NULL
  # use oob error to do hyper-parameter tuning
  for (i in 1:nrow(params.grid)) {
    temp.forest <- ranger(Y~., data = Data.train,
                          num.trees=params.grid$num.trees[i],
                          mtry=params.grid$mtry[i],
                          max.depth = params.grid$max.depth[i],
                          min.node.size = params.grid$min.node.size[i]
    )
    if (temp.forest$prediction.error <= MSE.oob) {
      forest <- temp.forest
      params <- params.grid[i,]
      MSE.oob <- temp.forest$prediction.error
    }
  }
  # predictions
  predicted.values <- predict(forest, data = Data.test, type = "response")$predictions
  if (data.split) {
    returnList <- list(predicted.values = predicted.values,
                       forest.model = forest,
                       params = params,
                       A1.ind = A1.ind,
                       A2.ind = A2.ind,
                       MSE.oob = MSE.oob)
  } else {
    returnList <- list(predicted.values = predicted.values,
                       forest.model = forest,
                       params = params,
                       MSE.oob = MSE.oob)
  }
  returnList
}


### XGBoost
### Xgboost model fitting with a data splitting option,
### use cross validation for hyper-parameter tuning
### Input: Y: n by 1 outcome vector
###        X: n by p_x covariate matrix
###        data.split: logic, to do data splitting or not, default by FALSE
###        split.prop: Proportion of data to be used for inference(size of subsample A1), default by 0.5
###        kfold: Number of folds for cross validation
###        nrounds: Number of rounds/trees, default by 1000
###        eta: Learning rate, default by 0.3
###        max_depth: Maximal depth of the tree, default by 6
###        min_child_weight: Minimum sum of sample weight (hessian) needed in a child, default by 1
###        subsample: Subsample ratio of the training samples, default by 1
###        colsample_bytree: Ratio of variables when constructing each tree, default by 1
### Output: predicted.values: Predictions of the outcome Y
###         xgb.model: XGBoost model object
###         params: Best hyper-parameters selected by cross validation
###         A1.ind: Indices of data in subsample A1, available when data.split=TRUE
###         A2.ind: Indices of data in subsample A2, available when data.split=TRUE
###         MSE.cv: Minimal cross-validated error using the best hyper-parameters
XGBoost <- function(Y,X,data.split=FALSE,split.prop=0.5,kfold=5,
                    nrounds=1000,eta=0.3,max_depth=6,min_child_weight=1,subsample=1,colsample_bytree=1) { # hyper-parameters
  X <- as.matrix(X); Y <- as.matrix(Y)
  n <- NROW(X); p <- NCOL(X)
  # search grid
  params.grid <- expand.grid(
    eta = eta,
    max_depth = max_depth,
    min_child_weight = min_child_weight,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    optimal_rounds = 0, # a place to save the optimal number of rounds
    min_MSE = 0 # a place to save test MSE
  )
  
  if (data.split) {
    # split the data into two parts A1 and A2, A2 for building model, A1 for inference
    n.A1 <- round(split.prop*n)
    A1.ind <- 1:n.A1
    A2.ind <- setdiff(1:n,A1.ind)
    Y.A1 <- Y[A1.ind,]; X.A1 <- X[A1.ind,]
    Y.A2 <- Y[A2.ind,]; X.A2 <- X[A2.ind,]
    Data.train <- xgb.DMatrix(X.A2,label=Y.A2)
    Data.test <- xgb.DMatrix(X.A1,label=Y.A1)
  } else {
    Data.train <- Data.test <- xgb.DMatrix(X,label=Y)
  }
  
  # model fittting
  for (i in 1:nrow(params.grid)) {
    # create parameter list
    params <- list(
      eta = params.grid$eta[i],
      max_depth = params.grid$max_depth[i],
      min_child_weight = params.grid$min_child_weight[i],
      subsample = params.grid$subsample[i],
      colsample_bytree = params.grid$colsample_bytree[i]
    )
    xgb.tune <- xgb.cv(
      params = params,
      data = Data.train,
      nrounds = nrounds,
      nfold = kfold,
      objective = "reg:squarederror",  # for regression models
      verbose = 0,               # silent
      early_stopping_rounds = 20 # stop if no improvement for 20 consecutive trees
    )
    # add min test RMSE and trees to grid
    params.grid$optimal_rounds[i] <- which.min(xgb.tune$evaluation_log$test_rmse_mean)
    params.grid$min_MSE[i] <- min(xgb.tune$evaluation_log$test_rmse_mean)^2
  }
  # find the best hyper-parameters
  params.grid <- params.grid[order(params.grid$min_MSE),]
  best.round <- params.grid[1,"optimal_rounds"]
  best.params <- as.list(params.grid[1,1:5])
  MSE.cv <- min(params.grid$min_MSE)
  
  
  # final model
  xgb.final <- xgb.train(
    params = best.params,
    data = Data.train,
    nrounds = best.round,
    objective = "reg:squarederror",
    verbose = 0
  )
  
  
  # make predictions
  predicted.values <- predict(xgb.final,newdata = Data.test)
  if (data.split) {
    returnList <- list(predicted.values = predicted.values,
                       xgb.model = xgb.final,
                       A1.ind = A1.ind,
                       A2.ind = A2.ind,
                       MSE.cv = MSE.cv)
  } else {
    returnList <- list(predicted.values = predicted.values,
                       xgb.model = xgb.final,
                       MSE.cv = MSE.cv)
  }
  returnList
}


### SAM
### Function: Sparse Additive Model with built-in cross validation
### Input: Y: n by 1 outcome vector
###        X: n by p_x covariate matrix
###        high_dim: logic, fit high-dimensional model, a penalty is used if TRUE, default by TRUE
###        lam.seq: A sequence of penalty parameters
###        kfold: Number of folds for cross validation
### Output: predicted.values: Predictions of the outcome Y
###         sam.model: Sparse additive model object
###         best.lam: Best lambda that minimize the cross-validated test MSE, available if high_dim=TRUE
###         MSE.cv: Minimal MSE by cross validation
SAM <- function(Y,X,high_dim = TRUE,lam.seq=NULL,kfold=5) {
  X <- as.matrix(X); Y <- as.matrix(Y)
  n <- NROW(X); p <- NCOL(X)
  # search grid for lambda
  if (!high_dim) {
    lam.seq <- 0 # no regularization if fitting the low-dimensional model
  } else {
    if (is.null(lam.seq)) lam.seq <- 0.5^seq(0, 10, l=100) # approximately from 0.001 to 1
  }
  # cross validation for the best lambda
  MSE.cv.seq <- rep(0,length(lam.seq))
  folds <- cut(seq(1,n),breaks=kfold,labels=FALSE)
  for (k in 1:kfold) {
    test.ind <- which(folds == k, arr.ind = TRUE)
    # create train and test data
    X.test <- X[test.ind,]
    Y.test <- Y[test.ind]
    X.train <- X[-test.ind,]
    Y.train <- Y[-test.ind]
    sam.k <- samQL(X.train,Y.train,lambda = lam.seq)
    Y.fit <- predict(sam.k, newdata = X.test)$values
    MSE.cv.seq <- MSE.cv.seq + apply(Y.fit-Y.test,2,function(x){mean(x^2)})/kfold
  }
  best.lam <- lam.seq[which.min(MSE.cv.seq)]
  MSE.cv <- min(MSE.cv.seq)
  
  # final model
  sam.fit <- samQL(X,Y,lambda = best.lam)
  predicted.values <- predict(sam.fit, newdata=X)$values
  
  returnList <- list(predicted.values = predicted.values,
                     sam.model = sam.fit,
                     MSE.cv = MSE.cv)
  if (high_dim) {
    returnList$best.lam <- best.lam
  }
  returnList
}



