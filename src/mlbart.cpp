/*
 *  BART: Bayesian Additive Regression Trees
 *  Copyright (C) 2018 Robert McCulloch and Rodney Sparapani
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  https://www.R-project.org/Licenses/GPL-2
 */

#include <ctime>
#include "common.h"
#include "tree.h"
#include "treefuns.h"
#include "info.h"
#include "mlbartfuns.h"
#include "bd.h"
#include "mlogitbart.h"
#include "rtnorm.h"
#include "lambda.h"

#ifndef NoRcpp

#define TRDRAW(a, b) trdraw(a, b)
#define TEDRAW(a, b) tedraw(a, b)

RcppExport SEXP mlogitbart(
   SEXP _type,          //1:mlbart, 2:mlbart (shared trees)
   SEXP _in,            //number of observations in training data
   SEXP _ip,            //dimension of x
   SEXP _inp,           //number of observations in test data
   SEXP _ik,            //k, number of categories in y
   SEXP _ix,            //x, train,  pxn (transposed so rows are contiguous in memory)
   SEXP _iy,            //y, train,  nx1
   // SEXP _idelta,        //censoring indicator
   SEXP _ixp,           //x, test, pxnp (transposed so rows are contiguous in memory)
   SEXP _im,            //number of trees
   SEXP _inc,           //number of cut points
   SEXP _ind,           //number of kept draws (except for thinnning ..)
   SEXP _iburn,         //number of burn-in draws skipped
   SEXP _ithin,         //thinning == keepevery
   SEXP _ipower,        // beta
   SEXP _ibase,         // alpha
   // SEXP _Offset,
   // SEXP _itau,
   // SEXP _inu,
   // SEXP _ilambda,
   // SEXP _isigest,
   // SEXP _iw,
   // SEXP _idart,         //dart prior: true(1)=yes, false(0)=no
   // SEXP _itheta,
   // SEXP _iomega,
   // SEXP _igrp,
   // SEXP _ia,            //param a for sparsity prior
   // SEXP _ib,            //param b for sparsity prior
   // SEXP _irho,          //param rho for sparsity prior (default to p)
   // SEXP _iaug,          //categorical strategy: true(1)=data augment false(0)=degenerate trees
   SEXP _ia0,
   SEXP _inprintevery,
   SEXP _Xinfo,
   SEXP _update_phi
)
{
   //process args
   int type = Rcpp::as<int>(_type);
   bool separate = type == 1 ? true : false;
   size_t n = Rcpp::as<int>(_in);
   size_t p = Rcpp::as<int>(_ip);
   size_t np = Rcpp::as<int>(_inp);
   size_t k = Rcpp::as<int>(_ik);
   Rcpp::NumericVector  xv(_ix);
   double *ix = &xv[0];
   Rcpp::NumericVector  yv(_iy); 
   double *iy = &yv[0];
   // Rcpp::IntegerVector  deltav(_idelta); 
   // int *delta = &deltav[0];
   Rcpp::NumericVector  xpv(_ixp);
   double *ixp = &xpv[0];
   size_t m = Rcpp::as<int>(_im);
   Rcpp::IntegerVector _nc(_inc);
   int *numcut = &_nc[0];
   size_t nd = Rcpp::as<int>(_ind);
   size_t burn = Rcpp::as<int>(_iburn);
   size_t thin = Rcpp::as<int>(_ithin);
   double mybeta = Rcpp::as<double>(_ipower);
   double alpha = Rcpp::as<double>(_ibase);
   double a0 = Rcpp::as<double>(_ia0);
   bool update_phi = Rcpp::as<bool>(_update_phi);
   size_t nkeeptrain = nd/thin;  // = ndpost    //Rcpp::as<int>(_inkeeptrain);
   size_t nkeeptest = nd/thin;   // = ndpost    //Rcpp::as<int>(_inkeeptest);
   size_t nkeeptreedraws = nd/thin; //Rcpp::as<int>(_inkeeptreedraws);
   size_t printevery = Rcpp::as<int>(_inprintevery);
   Rcpp::NumericMatrix varprb(nkeeptreedraws,p);
   Rcpp::IntegerMatrix varcnt(nkeeptreedraws,p);
   Rcpp::NumericMatrix Xinfo(_Xinfo);
   Rcpp::NumericVector sdraw(nd+burn);
   Rcpp::NumericMatrix trdraw(nkeeptrain, n*k);
   Rcpp::NumericMatrix tedraw(nkeeptest, np*k);


   //random number generation
   arn gen;

   mlbart bm(k, m);

   if(Xinfo.size()>0) {
     xinfo _xi;
     _xi.resize(p);
     for(size_t i=0;i<p;i++) {
       _xi[i].resize(numcut[i]);
       for(size_t j=0;j<numcut[i];j++) _xi[i][j]=Xinfo(i, j);
     }
     bm.setxinfo(_xi);
   }
#else

#define TRDRAW(a, b) trdraw[a][b]
#define TEDRAW(a, b) tedraw[a][b]

void mlogitbart(
   int type,            //1:wbart, 2:pbart, 3:lbart
   size_t n,            //number of observations in training data
   size_t p,		//dimension of x
   size_t np,		//number of observations in test data
   size_t k,
   double* ix,		//x, train,  pxn (transposed so rows are contiguous in memory)
   double* iy,		//y, train,  nx1
   // int* delta,          //censoring indicator
   double* ixp,		//x, test, pxnp (transposed so rows are contiguous in memory)
   size_t m,		//number of trees
   int *numcut,		//number of cut points
   size_t nd,		//number of kept draws (except for thinnning ..)
   size_t burn,		//number of burn-in draws skipped
   size_t thin,		//thinning
   double mybeta,
   double alpha,
   // double Offset,
   // double tau,
   // double nu,
   // double lambda,
   // double sigma,
   // double* iw,
   // bool dart,           //dart prior: true(1)=yes, false(0)=no   
   // double theta,
   // double omega, 
   // int* grp,
   // double a,		//param a for sparsity prior                                          
   // double b,		//param b for sparsity prior                                          
   // double rho,		//param rho for sparsity prior (default to p)                         
   // bool aug,		//categorical strategy: true(1)=data augment false(0)=degenerate trees
//   size_t nkeeptrain, //   size_t nkeeptest, //   size_t nkeeptreedraws,
   double a0,
   size_t printevery,
   unsigned int n1, // additional parameters needed to call from C++
   unsigned int n2,
   double* sdraw,
   double* _trdraw,
   double* _tedraw
)
{
   //return data structures (using C++)
   size_t nkeeptrain=nd/thin, nkeeptest=nd/thin, nkeeptreedraws=nd/thin;
   std::vector<double*> trdraw(nkeeptrain);
   std::vector<double*> tedraw(nkeeptest);

   for(size_t i=0; i<nkeeptrain; ++i) trdraw[i]=&_trdraw[i*n];
   for(size_t i=0; i<nkeeptest; ++i) tedraw[i]=&_tedraw[i*np];

   //matrix to return dart posteriors (counts and probs)
   std::vector< std::vector<size_t> > varcnt;
   std::vector< std::vector<double> > varprb;

   //random number generation
   arn gen(n1, n2);

   mlbart bm(k, m);
#endif

   std::stringstream treess;  //string stream to write trees to
   treess.precision(10);
   treess << nkeeptreedraws << " " << m*k << " " << p << endl;

   printf("*****Calling mlbart: type=%d\n", type);

   size_t skiptr=thin, skipte=thin, skiptreedraws=thin;
/*
   size_t skiptr,skipte,skipteme,skiptreedraws;
   if(nkeeptrain) {skiptr=nd/nkeeptrain;}
   else skiptr = nd+1;
   if(nkeeptest) {skipte=nd/nkeeptest;}
   else skipte=nd+1;
   if(nkeeptreedraws) {skiptreedraws = nd/nkeeptreedraws;}
   else skiptreedraws=nd+1;
*/

   //--------------------------------------------------
   //print args
   printf("*****Data:\n");
   printf("data:n,p,np,k: %zu, %zu, %zu, %zu\n",n,p,np,k);
   printf("y1,yn: %lf, %lf\n",iy[0],iy[n-1]);
   printf("x1,x[n*p]: %lf, %lf\n",ix[0],ix[n*p-1]);
   if(np) printf("xp1,xp[np*p]: %lf, %lf\n",ixp[0],ixp[np*p-1]);
   printf("*****Number of Trees: %zu\n",m);
   printf("*****Number of Cut Points: %d ... %d\n", numcut[0], numcut[p-1]);
   printf("*****burn,nd,thin: %zu,%zu,%zu\n",burn,nd,thin);
// printf("Prior:\nbeta,alpha,tau,nu,lambda,offset: %lf,%lf,%lf,%lf,%lf,%lf\n",
//                    mybeta,alpha,tau,nu,lambda,Offset);
   cout << "*****Prior:beta,alpha,a0: " 
	<< mybeta << ',' << alpha << ',' << a0 << endl;
// if(type==1) {
//    // printf("*****sigma: %lf\n",sigma);
//    printf("*****w (weights): %lf ... %lf\n",iw[0],iw[n-1]);
// }
   // cout << "*****Dirichlet:sparse,theta,omega,a,b,rho,augment: " 
	// << dart << ',' << theta << ',' << omega << ',' << a << ',' 
	// << b << ',' << rho << ',' << aug << endl;
   //printf("*****nkeeptrain,nkeeptest: %zu, %zu\n",nkeeptrain,nkeeptest);
   printf("*****printevery: %zu\n",printevery);

   // --------------------------------------------------
   // create temporaries
   // double df=n+nu;
   double *z = new double[n]; 

   for(size_t i=0; i<n; i++) z[i] = iy[i];
   // --------------------------------------------------
   //set up BART model
   bm.setprior(m, a0, alpha, mybeta, update_phi);
   bm.setdata(p,n,ix,z,numcut, separate);
   // bm.setdart(a,b,rho,aug,dart);
   // dart iterations
   std::vector<double> ivarprb (p,0.);
   std::vector<size_t> ivarcnt (p,0);
   //--------------------------------------------------
   //temporary storage
   //out of sample fit
   double* fhattrain = 0;
   double* fhattest=0; 
   if(n)  { fhattrain = new double[n*k];}
   if(np) { fhattest = new double[np*k]; }

   //--------------------------------------------------
   //mcmc
   printf("\nMCMC\n");
   //size_t index;
   size_t trcnt=0; //count kept train draws
   size_t tecnt=0; //count kept test draws
   bool keeptest,/*keeptestme*/keeptreedraw;

   time_t tp;
   int time1 = time(&tp), total=nd+burn;
   xinfo& xi = bm.getxinfo();

   for(size_t i=0;i<total;i++) {
      if(i%printevery==0) printf("done %zu (out of %lu)\n",i,nd+burn);
      //if(i%printevery==0) printf("%22zu/%zu\r",i,total);
      // if(i==(burn/2)&&dart) bm.startdart();
      //draw bart
      bm.draw(gen);

   //    //draw sigma
   //    if(type==1) {
	// double rss=0.;
	// for(size_t k=0;k<n;k++) rss += pow((iy[k]-bm.f(k))/(iw[k]), 2.); 
	// sigma = sqrt((nu*lambda + rss)/gen.chi_square(df));
	// sdraw[i]=sigma;
   //    }
   //    for(size_t k=0; k<n; k++) {
	// if(type==1) {
	//   svec[k]=iw[k]*sigma;
	//   if(delta[k]==0) z[k]= rtnorm(bm.f(k), iy[k], svec[k], gen);
	// }
	// else {
	//   z[k]= sign[k]*rtnorm(sign[k]*bm.f(k), -sign[k]*Offset, svec[k], gen);
	//   if(type==3) 
	//     svec[k]=sqrt(draw_lambda_i(pow(svec[k], 2.), sign[k]*bm.f(k), 1000, 1, gen));
	//   }
   //    }
   
      if(i>=burn) {
         if(nkeeptrain && (((i-burn+1) % skiptr) ==0)) {
            bm.predict(p,n, ix, fhattrain, true);
            for(size_t j=0;j<n*k;j++) TRDRAW(trcnt,j)=fhattrain[j];
            trcnt+=1;
         }
         keeptest = nkeeptest && (((i-burn+1) % skipte) ==0) && np;
         if(keeptest) {
	         bm.predict(p,np,ixp,fhattest, true);
            for(size_t j=0;j<np*k;j++) TEDRAW(tecnt,j)=fhattest[j];
            tecnt+=1;
         }
         keeptreedraw = nkeeptreedraws && (((i-burn+1) % skiptreedraws) ==0);
         if(keeptreedraw) {
            for(size_t j=0;j<m*k;j++) {
	            treess << bm.gettree(j);

               #ifndef NoRcpp
               ivarcnt=bm.getnv();
               ivarprb=bm.getpv();
               size_t k=(i-burn)/skiptreedraws;
               for(size_t j=0;j<p;j++){
                  varcnt(k,j)=ivarcnt[j];
                  varprb(k,j)=ivarprb[j];
               }
               #else
               varcnt.push_back(bm.getnv());
               varprb.push_back(bm.getpv());
               #endif
	         }
         }
      }
   }

   int time2 = time(&tp);
   printf("time: %ds\n",time2-time1);
   printf("trcnt,tecnt: %zu,%zu\n",trcnt,tecnt);

   if(fhattest) delete[] fhattest;
   delete[] z;
   // delete[] svec;
   // if(type!=1) delete[] sign;

#ifndef NoRcpp
   //return list
   Rcpp::List ret;
   // if(type==1) ret["sigma"]=sdraw;
   ret["yhat.train"]=trdraw; // dim of trdraw = ndpost , n*k; the prob of i obs in class j is trdraw[:, i*k + j]
   ret["yhat.test"]=tedraw;
   ret["varcount"]=varcnt;
   ret["varprob"]=varprb;

   Rcpp::List xiret(xi.size());
   for(size_t i=0;i<xi.size();i++) {
      Rcpp::NumericVector vtemp(xi[i].size());
      std::copy(xi[i].begin(),xi[i].end(),vtemp.begin());
      xiret[i] = Rcpp::NumericVector(vtemp);
   }

   Rcpp::List treesL;
   treesL["cutpoints"] = xiret;
   treesL["trees"]=Rcpp::CharacterVector(treess.str());
   ret["treedraws"] = treesL;
   
   return ret;
#else

#endif

}
