source("Prediction-Method.R", encoding = "UTF-8")

# generating data
n = 1500
p= 1500
X = 0.5*matrix(runif(n*p),n,p) + matrix(rep(0.5*runif(n),p),n,p)
colnames(X) <- paste("X", 1:p, sep = "")
X4 <- X[,1:4] # for low dim

# generating response
Y = -2*sin(X[,1]) + X[,2]^2-1/3 + X[,3]-1/2 + exp(-X[,4])+exp(-1)-1 + rnorm(n)

# train and test data
test.ind <- sample(1:n,500)
Y.test <- Y[test.ind]
Y.train <- Y[-test.ind]
X.test <- X[test.ind,]
X.train <- X[-test.ind,]
X4.test <- X4[test.ind,]
X4.train <- X4[-test.ind,]


# random forest
rf.fit <- random.forest(Y.train,X4.train)
# MSE with full data, use Y[rf.fit$A1.ind] if data.split=TRUE
mean((rf.fit$predicted.values-Y.train)^2)
# MSE using CV
rf.fit$MSE.oob
# test MSE
Y.pred.rf <- predict(rf.fit$forest.model,data=X4.test,type="response")$predictions
mean((Y.pred.rf-Y.test)^2)
### Example to compute weight
# weight <- RF.weight(rf.fit$nodes)


# xgboost
xgb.fit <- XGBoost(Y,X4)
# MSE with full data, use Y[rf.fit$A1.ind] if data.split=TRUE
mean((xgb.fit$predicted.values-Y)^2)
# MSE using CV
xgb.fit$MSE.cv
# test MSE
Y.pred.xgb <- predict(xgb.fit$xgb.model,newdata=X4.test)
mean((Y.pred.xgb-Y.test)^2)


# Sparse Additive Model
sam.fit <- SAM(Y,X4,high_dim = FALSE)
# MSE with full data, less overfitting
mean((sam.fit$predicted.values-Y)^2)
# MSE using CV
sam.fit$MSE.cv
# test MSE
Y.pred.sam <- predict(sam.fit$sam.model,newdata=X4.test)$values
mean((Y.pred.sam-Y.test)^2)


# # high-dimensional example for SAM
# sam.fit.hi <- SAM(Y,X)
# # MSE with full data, less overfitting
# mean((sam.fit.hi$predicted.values-Y)^2)
# # MSE using CV
# sam.fit.hi$MSE.cv



