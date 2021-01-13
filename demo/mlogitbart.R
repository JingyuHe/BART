library(XBART)
library(BART)
#seed = 10
#set.seed(seed)

# 
# n = 200
# nt = 50
n = 10000
nt = 5000
p = 6
p_cat = 0
k = 6
lam = matrix(0,n,k)
lamt = matrix(0,nt,k)


K = matrix(rnorm(3*p),p,3)
X_train = t(K%*%matrix(rnorm(3*n),3,n))
X_test = t(K%*%matrix(rnorm(3*nt),3,nt))

X_train = pnorm(X_train)
X_test = pnorm(X_test)

#X_train = matrix(runif(n*p,-1,1), nrow=n)
#X_test = matrix(runif(nt*p,-1,1), nrow=nt)

X_train = cbind(X_train, matrix(rbinom(n*p_cat, 1, 0.5), nrow = n))
X_test = cbind(X_test, matrix(rbinom(nt*p_cat, 1, 0.5), nrow = nt))

# X_train = cbind(X_train, matrix(rpois(n*p_cat, 20), nrow=n))
# X_test = cbind(X_test, matrix(rpois(nt*p_cat, 20), nrow=nt))


lam[,1] = abs(3*X_train[,1] - X_train[,2])
lam[,2] = 2
lam[,3] = 3*X_train[,3]^2
lam[,4] = 4*(X_train[, 4] * X_train[,5])
lam[,5] = 2*(X_train[,5] + X_train[,6])
lam[,6] = 2*(X_train[,1] + X_train[,3] - X_train[,5])
lamt[,1] = abs(3*X_test[,1] - X_test[,2])
lamt[,2] = 2
lamt[,3] = 3*X_test[,3]^2
lamt[,4] = 4*(X_test[,4]*X_test[,5])
lamt[,5] = 2*(X_test[,5] + X_test[,6])
lamt[,6] = 2*(X_test[,1] + X_test[,3] - X_test[,5])

# lam[,1] = 3*abs(2*X_train[,1] - X_train[,2])
# lam[,2] = 2
# lam[,3] = 3*X_train[,3]^2
# lam[,4] = 5*(X_train[, 4] * X_train[,5])
# lam[,5] = (X_train[,5] + 2*X_train[,6])
# lam[,6] = 2*(X_train[,1] + X_train[,3] - X_train[,5])
# lamt[,1] = 3*abs(2*X_test[,1] - X_test[,2])
# lamt[,2] = 2
# lamt[,3] = 3*X_test[,3]^2
# lamt[,4] = 5*(X_test[,4]*X_test[,5])
# lamt[,5] = (X_test[,5] + 2*X_test[,6])
# lamt[,6] = 2*(X_test[,1] + X_test[,3] - X_test[,5])


# vary s to make the problem harder s < 1 or easier s > 2
s = 1
pr = exp(s*lam)
pr = t(scale(t(pr),center=FALSE, scale = rowSums(pr)))
y_train = sapply(1:n,function(j) sample(0:(k-1),1,prob=pr[j,]))

pr = exp(s*lamt)
pr = t(scale(t(pr),center=FALSE, scale = rowSums(pr)))
y_test = sapply(1:nt,function(j) sample(0:(k-1),1,prob=pr[j,]))



# num_sweeps = ceiling(200/log(n)) 
num_sweeps = 20
burnin = 3
num_trees = 100
max_depth = 20
mtry = NULL # round((p + p_cat)/3)
#########################  parallel ####################3
tm = proc.time()
fit = XBART.multinomial(y=matrix(y_train), num_class=k, X=X_train, Xtest=X_test,
                        num_trees=num_trees, num_sweeps=num_sweeps, max_depth=max_depth,
                        num_cutpoints=NULL, alpha=0.95, beta=1.25, tau_a = 1, tau_b = 1,
                        no_split_penality = 1,  burnin = burnin, mtry = mtry, p_categorical = p_cat,
                        kap = 1, s = 1, verbose = FALSE, set_random_seed = FALSE,
                        random_seed = NULL, sample_weights_flag = TRUE, separate_tree = FALSE, stop_threshold = 0,
                        weight = 1, hmult = 1, heps = 0)


tm = proc.time()-tm
cat(paste("\n", "parallel xbart runtime: ", round(tm["elapsed"],3)," seconds"),"\n")
phat = apply(fit$yhats_test[burnin:num_sweeps,,], c(2,3), mean) 
yhat = apply(phat,1,which.max)-1
cat(paste("xbart classification accuracy: ",round(mean(y_test == yhat),3)),"\n")


tm2 = proc.time()
fit.bart <- mlbart(x.train = X_train, y.train = y_train, num_class=k, x.test=X_test, 
                   type='separate', power=2, base=0.95, ntree = 100, numcut=100L, ndpost = 1000, keepevery=10)
tm2 = proc.time()-tm2
cat(paste("bart runtime: ", round(tm2["elapsed"],3)," seconds"),"\n")
phat.bart <- t(apply(fit.bart$yhat.test, c(2, 3), mean))
yhat.bart <- apply(phat.bart, 1, which.max) - 1

spr <- split(phat, row(phat))
logloss <- sum(mapply(function(x,y) -log(x[y]), spr, y_test+1, SIMPLIFY =TRUE))

spr.bart <- split(phat.bart, row(phat.bart))
logloss.bart <- sum(mapply(function(x,y) -log(x[y]), spr.bart, y_test+1, SIMPLIFY =TRUE))

cat(paste("xbart logloss : ",round(logloss,3)),"\n")
cat(paste("bart logloss : ", round(logloss.bart,3)),"\n")

cat(paste("\n", "xbart runtime: ", round(tm["elapsed"],3)," seconds"),"\n")
cat(paste("xgboost runtime: ", round(tm2["elapsed"],3)," seconds"),"\n")

cat(paste("xbart classification accuracy: ",round(mean(y_test == yhat),3)),"\n")
cat(paste("bart classification accuracy: ", round(mean(yhat.bart == y_test),3)),"\n")

