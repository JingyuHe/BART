library(XBART)
library(BART)
seed = 10
set.seed(seed)

# 
# n = 200
# nt = 50
n = 1000
nt = 500
p = 6
p_cat = 0
k = 3
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
# lam[,4] = 4*(X_train[, 4] * X_train[,5])
# lam[,5] = 2*(X_train[,5] + X_train[,6])
# lam[,6] = 2*(X_train[,1] + X_train[,3] - X_train[,5])
lamt[,1] = abs(3*X_test[,1] - X_test[,2])
lamt[,2] = 2
lamt[,3] = 3*X_test[,3]^2
# lamt[,4] = 4*(X_test[,4]*X_test[,5])
# lamt[,5] = 2*(X_test[,5] + X_test[,6])
# lamt[,6] = 2*(X_test[,1] + X_test[,3] - X_test[,5])

# vary s to make the problem harder s < 1 or easier s > 2
s = 15
pr = exp(s*lam)
pr = t(scale(t(pr),center=FALSE, scale = rowSums(pr)))
y_train = sapply(1:n,function(j) sample(0:(k-1),1,prob=pr[j,]))

pr = exp(s*lamt)
pr = t(scale(t(pr),center=FALSE, scale = rowSums(pr)))
y_test = sapply(1:nt,function(j) sample(0:(k-1),1,prob=pr[j,]))



# num_sweeps = ceiling(200/log(n)) 
num_sweeps = 20
burnin = 0
num_trees = 10
max_depth = 2
mtry = NULL 
separate_tree = FALSE


# extreme case, only draw one posterior sample
# warm start should be better than the root initialization
n_posterior = 1
thinning = 1


#########################  parallel ####################3
tm = proc.time()
fit = XBART.multinomial(y=matrix(y_train), num_class=k, X=X_train, Xtest=X_test,
                        num_trees=num_trees, num_sweeps=num_sweeps, max_depth=max_depth,
                        num_cutpoints=NULL, alpha=0.95, beta=1.25, 
                        no_split_penality = 1,  burnin = burnin, mtry = mtry, p_categorical = p_cat,
                        update_tau = FALSE, separate_tree = separate_tree, stop_threshold = 0, hmult = 1, heps = 0)
tm = proc.time()-tm
cat(paste("\n", "parallel xbart runtime: ", round(tm["elapsed"],3)," seconds"),"\n")
phat = apply(fit$yhats_test[burnin:num_sweeps,,], c(2,3), mean)
yhat = apply(phat,1,which.max)-1
cat(paste("xbart classification accuracy: ",round(mean(y_test == yhat),3)),"\n")

if(separate_tree){
    type = "separate"
}else{
    type = "shared"
}


tm4 = proc.time()
fit.bart.warmstart <- mlbart_ini(fit$treedraws[20], x.train = X_train, y.train = y_train, num_class=k, x.test=X_test, type=type, power=2, base=0.95, ntree = num_trees, ndpost = n_posterior, keepevery=thinning, nskip=burnin)
tm4 = proc.time()-tm4
cat(paste("warmstart runtime: ", round(tm4["elapsed"],3)," seconds"),"\n")
phat.bart.warmstart <- t(apply(fit.bart.warmstart$yhat.test, c(2, 3), mean))
yhat.bart.warmstart <- apply(phat.bart.warmstart, 1, which.max) - 1


tm2 = proc.time()
fit.bart.sep <- mlbart(x.train = X_train, y.train = y_train, num_class=k, x.test=X_test, 
                   type='separate', power=2, base=0.95, ntree = num_trees, ndpost = n_posterior, keepevery=thinning, nskip=burnin)
tm2 = proc.time()-tm2
cat(paste("bart runtime: ", round(tm2["elapsed"],3)," seconds"),"\n")
phat.bart.sep <- t(apply(fit.bart.sep$yhat.test, c(2, 3), mean))
yhat.bart.sep <- apply(phat.bart.sep, 1, which.max) - 1


tm3 = proc.time()
fit.bart.shrd <- mlbart(x.train = X_train, y.train = y_train, num_class=k, x.test=X_test, 
                       type='shared', power=2, base=0.95, 
                       ntree = num_trees, ndpost = n_posterior, keepevery=thinning, nskip=burnin)
tm3 = proc.time()-tm3
cat(paste("bart runtime: ", round(tm3["elapsed"],3)," seconds"),"\n")
phat.bart.shrd <- t(apply(fit.bart.shrd$yhat.test, c(2, 3), mean))
yhat.bart.shrd <- apply(phat.bart.shrd, 1, which.max) - 1

spr <- split(phat, row(phat))
logloss <- sum(mapply(function(x,y) -log(x[y]), spr, y_test+1, SIMPLIFY =TRUE))

spr.bart.sep <- split(phat.bart.sep, row(phat.bart.sep))
logloss.bart.sep <- sum(mapply(function(x,y) -log(x[y]), spr.bart.sep, y_test+1, SIMPLIFY =TRUE))

spr.bart.shrd <- split(phat.bart.shrd, row(phat.bart.shrd))
logloss.bart.shrd <- sum(mapply(function(x,y) -log(x[y]), spr.bart.shrd, y_test+1, SIMPLIFY =TRUE))

spr.bart.warmstart <- split(phat.bart.warmstart, row(phat.bart.warmstart))
logloss.bart.warmstart <- sum(mapply(function(x,y) -log(x[y]), spr.bart.warmstart, y_test+1, SIMPLIFY =TRUE))


results = matrix(0, 3, 4)
results[1,] = c(round(logloss,3), round(logloss.bart.sep,3), round(logloss.bart.shrd,3), round(logloss.bart.warmstart, 3))
results[2,] = c(round(tm["elapsed"],3), round(tm2["elapsed"],3), round(tm3["elapsed"],3), round(tm4["elapsed"],3))
results[3,] = c(round(mean(y_test == yhat),3), round(mean(yhat.bart.sep == y_test),3), round(mean(yhat.bart.shrd == y_test),3), round(mean(yhat.bart.warmstart == y_test),3))

rownames(results) = c("logloss", "runtime", "accuracy")
colnames(results) = c("XBART", "BART separate", "BART shared", "Warmstart")

print(results)

