#! /bin/bash
echo Building R
cd ../../
R CMD REMOVE BART
R CMD INSTALL BART
cd BART/demo/
echo Testing R
Rscript test.mlogitbart.warmstart.R
