
# Warm start BART with initialization at XBART trees
This package customized the [original BART package](https://cran.r-project.org/web/packages/BART/index.html) for warm start. It can load fitted trees from XBART and run the regular BART MCMC sampler. 

Besides, this package provides multinomial classification BART, see the function ``mlbart``. 

The main function fitting warm-start BART is ``wbart_ini`` for regression, and ``mlbart_ini`` for classification trees. See the demo script in the [XBART package](https://github.com/JingyuHe/XBART) for details, ``tests/test_regression_warmstart.R`` and ``tests/test_classification_warmstart.R``.
