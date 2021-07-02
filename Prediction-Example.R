source("Prediction-Method.R", encoding = "UTF-8")

# generating data
n = 1000
d = 1500
X = 0.5*matrix(runif(n*d),n,d) + matrix(rep(0.5*runif(n),d),n,d)
X4 <- X[,1:4] # for low dim

# generating response
Y = -2*sin(X[,1]) + X[,2]^2-1/3 + X[,3]-1/2 + exp(-X[,4])+exp(-1)-1 + rnorm(n)


# random forest
# parameters
num.trees = 500
mtry = 1:NCOL(X4)
max.depth = c(0,2,4,6)
min.node.size = c(5,10,20)
# model
rf.fit <- random.forest(Y,X4,
                        num.trees = num.trees,
                        mtry = mtry,
                        max.depth = max.depth,
                        min.node.size = min.node.size
                        )
# MSE with full data
mean((rf.fit$predicted.values-Y)^2)
# MSE using CV
rf.fit$MSE.oob


# xgboost
# parameters
nrounds = 1000
eta = c(0.05, 0.1, 0.3)
max_depth = c(3, 6)
min_child_weight = c(1, 3, 5)
subsample = c(0.8, 1)
colsample_bytree = c(0.8, 1)
#model
xgb.fit <- XGBoost(Y,X4,
                   nrounds = nrounds,
                   eta = eta,
                   max_depth = max_depth,
                   min_child_weight = min_child_weight,
                   subsample = subsample,
                   colsample_bytree = colsample_bytree
                   )
# MSE with full data
mean((xgb.fit$predicted.values-Y)^2)
# MSE using CV
xgb.fit$MSE.cv


# Sparse Additive Model
sam.fit <- SAM(Y,X4,high_dim = FALSE)
# MSE with full data, less overfitting
mean((sam.fit$predicted.values-Y)^2)
# MSE using CV
sam.fit$MSE.cv


# high-dimensional example for SAM
sam.fit.hi <- SAM(Y,X)
# MSE with full data, less overfitting
mean((sam.fit.hi$predicted.values-Y)^2)
# MSE using CV
sam.fit.hi$MSE.cv



