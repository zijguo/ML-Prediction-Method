### Several different functions for prediction methods including
### random forest, xgboost and sparse additive model.


library(Matrix)
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
### Output: forest.model: Random forest model object
###         params: Best hyper-parameters selected by minimizing out-of-bag error
###         predicted.values: Or predicted.A1 for data splitting, refers to predictions for the outcome
###         nodes: n(n.A1) by num.trees matrix, rows refer to different samples, columns refer to
###                different trees, the entrees are leaf node indices of each sample in each tree
###         A1.ind: Indices of data in subsample A1, available when data.split=TRUE
###         MSE.oob: Minimal out-of-bag error using the best hyper-parameters
random.forest <- function(Y,X,data.split=FALSE,split.prop=0.5,num.trees=NULL,mtry=NULL,max.depth=NULL,min.node.size=NULL,MSE.thol=1e12) {
  X <- as.matrix(X); Y <- as.matrix(Y)
  n <- NROW(X); p <- NCOL(X)
  
  # default hyper-parameters
  if (is.null(num.trees)) num.trees <- 200
  if (is.null(mtry)) mtry <- seq(round(p/3),round(2*p/3),by=1)
  if (is.null(max.depth)) max.depth <- c(0,2,4,6)
  if(is.null(min.node.size)) min.node.size <- c(5,10,20)
  
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
  # nodes
  nodes <- predict(forest, data = Data.test, type = "terminalNodes")$predictions
  
  if (data.split) {
    returnList <- list(forest.model = forest,
                       params = params,
                       predicted.A1 = predicted.values,
                       nodes = nodes,
                       A1.ind = A1.ind,
                       MSE.oob = MSE.oob)
  } else {
    returnList <- list(forest.model = forest,
                       params = params,
                       predicted.values = predicted.values,
                       nodes = nodes,
                       MSE.oob = MSE.oob,
                       Data = Data)
  }
  returnList
}


### RF.weight
### Function: Compute the weight matrix of Random Forest
### Input: nodes: A n by num.trees matrix, rows refer to different samples, columns refer to
###               different trees, the entrees are leaf node indices of each sample in each tree
### Output: out.weight: A n by n symmetric sparse weight matrix(class dgCMatrix), the ith row represents
###                the weights of outcome of each sample on the prediction of the ith outcome
RF.weight <- function(nodes) {
  n.A1 <- NROW(nodes); num.trees <- NCOL(nodes)
  out.weight <- matrix(0,n.A1,n.A1)
  for (j in 1:num.trees) {
    weight.mat <- matrix(0,n.A1,n.A1) # weight matrix for single tree
    unique.nodes <- unique(nodes[,j])
    for (i in 1:length(unique.nodes)) {
      ind <- nodes[,j]==unique.nodes[i] # indices of samples in the node
      num.samples <- sum(ind) # number of samples in the node
      w <- 1/(num.samples-1)  # weight, to remove self-prediction
      weight.vec <- ifelse(ind,yes=w,no=0)
      weight.mat[ind,] <- matrix(rep(weight.vec,num.samples),num.samples,byrow=T)/num.trees
    }
    diag(weight.mat) <- 0 # remove self prediction
    out.weight <- out.weight + weight.mat
  }
  out.weight <- Matrix(out.weight, sparse = T) # sparse matrix to save memory
  return(out.weight)
}


### XGBoost
### Xgboost model fitting with a data splitting option,
### use cross validation for hyper-parameter tuning
### Input: Y: n by 1 outcome vector
###        X: n by p_x covariate matrix
###        data.split: logic, to do data splitting or not, default by FALSE
###        split.prop: Proportion of data to be used for inference(size of subsample A1)
###        kfold: Number of folds for cross validation
###        nrounds: Number of rounds/trees
###        eta: Learning rate
###        max_depth: Maximal depth of the tree
###        min_child_weight: Minimum sum of sample weight (hessian) needed in a child
###        subsample: Subsample ratio of the training samples
###        colsample_bytree: Ratio of variables when constructing each tree
### Output: xgb.model: XGBoost model object
###         params: Best hyper-parameters selected by cross validation
###         predicted.values(predicted.A1): Predictions of the outcome Y
###         A1.ind: Indices of data in subsample A1, available when data.split=TRUE
###         MSE.cv: Minimal cross-validated error using the best hyper-parameters
XGBoost <- function(Y,X,data.split=FALSE,split.prop=0.5,kfold=5,
                    nrounds=NULL,eta=NULL,max_depth=NULL,min_child_weight=NULL,subsample=NULL,colsample_bytree=NULL) { # hyper-parameters
  X <- as.matrix(X); Y <- as.matrix(Y)
  n <- NROW(X); p <- NCOL(X)
  
  # default value for hyper-parameters
  if (is.null(nrounds)) nrounds <- 1000
  if (is.null(eta)) eta <- c(0.05, 0.1, 0.3)
  if (is.null(max_depth)) max_depth <- c(3, 6)
  if (is.null(min_child_weight)) min_child_weight <- c(1 ,3, 5)
  if (is.null(subsample)) subsample <- c(0.8, 1)
  if (is.null(colsample_bytree)) colsample_bytree <- c(0.8, 1)
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
    returnList <- list(xgb.model = xgb.final,
                       params = best.params,
                       predicted.A1 = predicted.values,
                       A1.ind = A1.ind,
                       MSE.cv = MSE.cv)
  } else {
    returnList <- list(xgb.model = xgb.final,
                       params = best.params,
                       predicted.values = predicted.values,
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
###        degree: Number of basis functions
###        kfold: Number of folds for cross validation
### Output: predicted.values: Predictions of the outcome Y
###         sam.model: Sparse additive model object
###         best.lam: Best lambda that minimize the cross-validated test MSE, available if high_dim=TRUE
###         MSE.cv: Minimal MSE by cross validation
SAM <- function(Y,X,high_dim = TRUE,lam.seq=NULL,degree=NULL,kfold=5) {
  X <- as.matrix(X); Y <- as.matrix(Y)
  n <- NROW(X); p <- NCOL(X)
  # search grid for lambda
  if (!high_dim) {
    lam.seq <- 0 # no regularization if fitting the low-dimensional model
  } else {
    if (is.null(lam.seq)) lam.seq <- 0.5^seq(0, 10, l=100) # approximately from 0.001 to 1
  }
  if (is.null(degree)) degree = 3:6
  # cross validation for the best lambda and degree
  len_d <- length(degree)
  len_lam <- length(lam.seq)
  MSE <- matrix(0, len_d, 3)
  colnames(MSE) <- c("degree", "lambda", "MSE")
  
  # cross validation
  folds <- cut(seq(1,n),breaks=kfold,labels=FALSE)
  for (i in 1:len_d) {
    mse.lam <- rep(0, len_lam)
    for (fold in 1:kfold) {
      testInd <- which(folds == fold, arr.ind = TRUE)
      # create train and test data
      X.test <- X[testInd, ]
      Y.test <- Y[testInd]
      X.train <- X[-testInd, ]
      Y.train <- Y[-testInd]
      
      sam.fit <- samQL(X.train, Y.train, p = degree[i], lambda = lam.seq)
      Y.fit <- predict(sam.fit, newdata = X.test)$values
      mse.lam <- mse.lam + apply(Y.test-Y.fit, 2, function(x) mean(x^2))/kfold
    }
    
    temp.lam <- lam.seq[which.min(mse.lam)]
    MSE[i, "degree"] <- degree[i]
    MSE[i, "lambda"] <- temp.lam
    MSE[i, "MSE"] <- min(mse.lam)
  }
  best.ind <- which.min(MSE[, "MSE"])
  best.degree <- MSE[best.ind, "degree"]
  best.lam <- MSE[best.ind, "lambda"]
  
  # final model
  sam.final <- samQL(X,Y,p = best.degree,lambda = best.lam)
  predicted.values <- predict(sam.fit, newdata=X)$values
  
  returnList <- list(sam.model = sam.final,
                     predicted.values = predicted.values,
                     MSE.cv = min(MSE[,"MSE"]))
  if (high_dim) {
    returnList$best.lam <- best.lam
  }
  returnList
}



