library(XBART)
library(BART)
library(xgboost)
seed = 10
set.seed(seed)

#####################
# simulation parameters
n <- 5000 # training size
nt <- 2000 # testing size
p <- 30 # number of X variables
p_cat <- 0 # number of categorical X variables
k <- 6 # number of classes
lam <- matrix(0, n, k)
lamt <- matrix(0, nt, k)


set.seed(100)
#####################
# simulate data
K <- matrix(rnorm(3 * p), p, 3)
X_train <- t(K %*% matrix(rnorm(3 * n), 3, n))
X_test <- t(K %*% matrix(rnorm(3 * nt), 3, nt))
X_train <- pnorm(X_train)
X_test <- pnorm(X_test)
X_train <- cbind(X_train, matrix(rbinom(n * p_cat, 1, 0.5), nrow = n))
X_test <- cbind(X_test, matrix(rbinom(nt * p_cat, 1, 0.5), nrow = nt))
lam[, 1] <- abs(3 * X_train[, 1] - X_train[, 2])
lam[, 2] <- 2
lam[, 3] <- 3 * X_train[, 3]^2
lam[, 4] <- 4 * (X_train[, 4] * X_train[, 5])
lam[, 5] <- 2 * (X_train[, 5] + X_train[, 6])
lam[, 6] <- 2 * (X_train[, 1] + X_train[, 3] - X_train[, 5])
lamt[, 1] <- abs(3 * X_test[, 1] - X_test[, 2])
lamt[, 2] <- 2
lamt[, 3] <- 3 * X_test[, 3]^2
lamt[, 4] <- 4 * (X_test[, 4] * X_test[, 5])
lamt[, 5] <- 2 * (X_test[, 5] + X_test[, 6])
lamt[, 6] <- 2 * (X_test[, 1] + X_test[, 3] - X_test[, 5])

#####################
# vary s to make the problem harder s < 1 or easier s > 2
s <- 10
pr <- exp(s * lam)
pr <- t(scale(t(pr), center = FALSE, scale = rowSums(pr)))
y_train <- sapply(1:n, function(j) sample(0:(k - 1), 1, prob = pr[j, ]))

pr <- exp(s * lamt)
pr <- t(scale(t(pr), center = FALSE, scale = rowSums(pr)))
y_test <- sapply(1:nt, function(j) sample(0:(k - 1), 1, prob = pr[j, ]))

#########################  parallel ####################
# num_sweeps = ceiling(200/log(n)) 
# num_sweeps = 20
# burnin = 5
# num_trees = 20
# max_depth = NULL
# mtry = NULL 
separate_tree = FALSE
tm = proc.time()
# fit <- XBART.multinomial(y = matrix(y_train), num_class = k, X = X_train, 
#     num_trees = num_trees, num_sweeps = num_sweeps, burnin = burnin,
#     p_categorical = p_cat, tau_a = 3.5, tau_b = 3,
#     verbose = T, parallel = T,
#     separate_tree = separate_tree, 
#     update_tau = F, update_weight = T, a = 0.1, update_phi = F)

num_sweeps <- 30
burnin <- 2
num_trees <- 50
max_depth <- 25
mtry <- p
num_class <- k
fit <- XBART.multinomial(
  y = matrix(y_train), num_class = num_class, X = X_train,
  num_trees = num_trees, num_sweeps = num_sweeps, max_depth = max_depth, update_weight = TRUE,
  num_cutpoints = 100, burnin = burnin, mtry = mtry, p_categorical = p_cat, 
  tau_a = (num_trees * 2 / 2.5^2 + 0.5), tau_b = (num_trees * 2 / 2.5^2), 
  verbose = T, parallel = T, separate_tree = FALSE, update_tau = FALSE, update_phi = FALSE, 
  a = 1 / num_class, no_split_penalty = 0.5, alpha = 0.95, beta = 2, Nmin = 15 * num_class, 
  weight = 2.5, MH_step = 0.05
)
tm = proc.time()-tm
cat(paste("\n", "parallel xbart runtime: ", round(tm["elapsed"],3)," seconds"),"\n")

pred <- predict(fit, X_test, burnin = burnin)
yhat <- pred$label # prediction of classes
phat <- pred$prob # prediction of probability in each class
cat(paste("xbart classification accuracy: ",round(mean(y_test == yhat),3)),"\n")

if(separate_tree){
    type = "separate"
}else{
    type = "shared"
}

# extreme case, only draw one posterior sample
# warm start should be better than the root initialization
n_posterior = 100
thinning = 5

tm2 = proc.time()
# (burnin+1):
fit.bart.warmstart <- mlbart_ini(fit$treedraws[(burnin+1):num_sweeps], x.train = X_train, y.train = y_train, num_class=k, x.test=X_test, 
    type=type, power=2, base=0.95, ntree = num_trees, ndpost = n_posterior, keepevery=thinning, nskip=burnin, 
    update_phi = F, update_weight = T, weight = fit$weight[1, num_sweeps],
    c = (num_trees * 2 / 2.5^2 + 0.5), d = (num_trees * 2 / 2.5^2)
    ) #nskipp = burnin
tm2 = proc.time()-tm2 + tm
cat(paste("warmstart runtime: ", round(tm2["elapsed"],3)," seconds"),"\n")
phat.bart.warmstart <- t(apply(fit.bart.warmstart$yhat.test, c(2, 3), mean))
yhat.bart.warmstart <- apply(phat.bart.warmstart, 1, which.max) - 1

# Trace plot acc / individual prediction
yhat_trace <- apply(fit.bart.warmstart$yhat.test, 1, function(phat) apply(phat, 2, which.max) - 1)
acc_trace <- apply(yhat_trace, 2, function(yhat) mean(yhat == y_test))
plot(1:n_posterior, acc_trace, type = "l")
# 

# n_posterior = 200
# thinning = 10

tm3 = proc.time()
fit.bart <- mlbart(x.train = X_train, y.train = y_train, num_class=k, x.test=X_test, 
                       type='shared', power=1.25, base=0.95, 
                       ntree = num_trees, ndpost = n_posterior, keepevery=thinning, nskip=0, update_phi = F, update_weight = T) #nskip = burnin
tm3 = proc.time()-tm3
cat(paste("bart runtime: ", round(tm3["elapsed"],3)," seconds"),"\n")
phat.bart <- t(apply(fit.bart$yhat.test, c(2, 3), mean))
yhat.bart <- apply(phat.bart, 1, which.max) - 1

# Separate version
# tm5 = proc.time()
# fit.bart.sep <- mlbart(x.train = X_train, y.train = y_train, num_class=k, x.test=X_test, 
#                    type='separate', power=1.25, base=0.95, 
#                    ntree = num_trees, ndpost = n_posterior, keepevery=thinning, nskip=0) #nskip = burnin
# tm5 = proc.time()-tm5
# cat(paste("bart runtime: ", round(tm5["elapsed"],3)," seconds"),"\n")
# phat.bart.sep <- t(apply(fit.bart.sep$yhat.test, c(2, 3), mean))
# yhat.bart.sep <- apply(phat.bart.sep, 1, which.max) - 1

tm4 <- proc.time()
# fit.xgb <- xgboost(data = X_train, label = y_train, num_class = k, verbose = 0, max_depth = 4, subsample = 0.80, nrounds = 500, early_stopping_rounds = 2, eta = 0.1, params = list(objective = "multi:softprob"))
fit.xgb <- xgboost(data = as.matrix(X_train), label = matrix(y_train),
                   num_class=k, verbose = 0,
                   nrounds=200,
                   early_stopping_rounds = 50,
                   params=list(objective="multi:softprob"))
tm4 <- proc.time() - tm4
cat(paste("XGBoost runtime: ", round(tm2["elapsed"], 3), " seconds"), "\n")
phat.xgb <- predict(fit.xgb, X_test)
phat.xgb <- matrix(phat.xgb, ncol = k, byrow = TRUE)
yhat.xgb <- max.col(phat.xgb) - 1

spr <- split(phat, row(phat))
logloss <- sum(mapply(function(x,y) -log(x[y]), spr, y_test+1, SIMPLIFY =TRUE)) / nt

spr.bart <- split(phat.bart, row(phat.bart))
logloss.bart <- sum(mapply(function(x,y) -log(x[y]), spr.bart, y_test+1, SIMPLIFY =TRUE)) / nt

spr.bart.warmstart <- split(phat.bart.warmstart, row(phat.bart.warmstart))
logloss.bart.warmstart <- sum(mapply(function(x,y) -log(x[y]), spr.bart.warmstart, y_test+1, SIMPLIFY =TRUE))  / nt

spr <- split(phat.xgb, row(phat.xgb))
logloss.xgb <- sum(mapply(function(x, y) -log(x[y]), spr, y_test + 1, SIMPLIFY = TRUE)) / nt


# # 
par(mfrow = c(2, 2))
ind <- 1
ind_trace <- fit.bart.warmstart$yhat.test[,y_test[ind] + 1,ind]
plot(1:n_posterior, ind_trace, type = "l", main = "warmstart")

ind_trace <- fit.bart$yhat.test[,y_test[ind] + 1,ind]
plot(1:n_posterior, ind_trace, type = "l", main = "BART")

ind_trace <- pred$yhats[,ind,y_test[ind] + 1]
plot(1:length(ind_trace), ind_trace, type = "l", main = "XBART")

print(paste("Xgboost prob for ind", ind, "=", phat.xgb[ind, y_test[ind]+1]))
print(paste("True prob for ind", ind, "=", pr[ind, y_test[ind]+1]))

results = matrix(0, 3, 4)
results[1,] = c(round(logloss,3),  round(logloss.bart,3), round(logloss.bart.warmstart, 3), round(logloss.xgb, 3))
results[2,] = c(round(tm["elapsed"],3), round(tm3["elapsed"],3), round(tm2["elapsed"],3), round(tm4["elapsed"], 3))
results[3,] = c(round(mean(y_test == yhat),3), round(mean(yhat.bart == y_test),3), round(mean(yhat.bart.warmstart == y_test),3), round(mean(yhat.xgb == y_test),3))

rownames(results) = c("logloss", "runtime", "accuracy")
colnames(results) = c("XBART", "BART", "Warmstart", "Xgb")

print(results)


