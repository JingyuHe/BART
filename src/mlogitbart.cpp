/*
 *  BART: Bayesian Additive Regression Trees
 *  Copyright (C) 2017 Robert McCulloch and Rodney Sparapani
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

#include "mlogitbart.h"

using namespace std;

// constructor
mlbart::mlbart(size_t ik):bart(100 * ik), k(ik),  phi(0), separate(true), mpi(), mdi() {}
mlbart::mlbart(size_t ik, size_t im):bart(im * ik), k(ik),  phi(0), separate(true), mpi(), mdi() {} // construct m by k trees, im-th tree for class ik is t[im*k + ik]


void mlbart::setdata(size_t p, size_t n, double *x, double *y, int *nc, bool separate, double weight)
{
   this->p=p; this->n=n; this->x=x; this->y=y; this->separate = separate;
   if(xi.size()==0) makexinfo(p,n,&x[0],xi,nc);

   // initialize theta value for all trees to be 1;
   for (size_t j = 0; j < t.size(); j++) t[j].settheta(1.);

   if(allfit) delete[] allfit;
   allfit = new double[n*k]; // n by k with 1 ... n for class 0, n+1, ..., n+n for class 1, etc.
   predict(p,n,x,allfit,false); // keep allfit in logscale

   if(r) delete[] r;
   r = new double[n*k]; 

   if(ftemp) delete[] ftemp;
   ftemp = new double[n*k]; // n by k with 1 ... n for class 0, n+1, ..., n+n for class 1, etc.

   if(phi) delete[] phi;
   phi = new double[n];
   for (size_t i = 0; i < n; i++) phi[i] = 1;

   mdi.n=n; mdi.p=p; mdi.x = &x[0]; mdi.y=y;
   mdi.k = k; mdi.phi = phi; mdi.f = allfit; mdi.ik = 0; 
   mdi.weight = weight; mdi.weight_latent = weight; mdi.logloss_last_sweep = log(1 / k);
   for(size_t j=0;j<p;j++){
     nv.push_back(0);
     pv.push_back(1/(double)p);
   }
}
// --------------------------------------------------
void mlbart::predict(size_t p, size_t n, double *x, double *fp, bool normalize)
//uses: m,t,xi
{
   double *fptemp = new double[n];

   for(size_t j=0;j<n*k;j++) fp[j]=0.0;
   for(size_t j=0;j< (size_t) m/k ;j++) {
      for(size_t ik = 0; ik < k; ik++)
      {
         fit(t[j*k + ik], xi, p, n, x, fptemp);
         for (size_t i = 0; i < n; i++) fp[i*k + ik] += log(fptemp[i]); // fp for i-th obs and j-th class is fp[i*k + j].
      }
   }

   if (normalize){
      // normalization
      double denom = 0.0;
      double max_log_prob = -INFINITY;
      for(size_t i = 0; i < n; i++)
      {
         max_log_prob = -INFINITY;
         for (size_t ik = 0; ik < k; ik++)
         {
         if (fp[i*k + ik] > max_log_prob) max_log_prob = fp[i*k + ik];  
         }
         denom = 0.0;
         for (size_t ik = 0; ik < k; ik++)
         {
            fp[i*k + ik] = exp(fp[i*k + ik] - max_log_prob);
            denom += fp[i*k + ik];
         } 
         for (size_t ik = 0; ik < k; ik ++) fp[i*k + ik] = fp[i*k + ik] / denom;
      }
   }
   // else{
   //    // put log(fp) back to orginal scale
   //    for(size_t j=0;j<n*k;j++) fp[j]=exp(fp[j]);
   // }
   
   delete[] fptemp;
}
// --------------------------------------------------

void mlbart::draw(rn& gen)
{
   for(size_t j=0; j< (size_t) m/k;j++) {
      // cout << "ok 1" << endl;
      if (mpi.update_phi) {
         drphi(phi, allfit, n, k, gen);
      }
      if (separate){
         // cout << "ok 2" << endl;
         for(size_t ik = 0; ik < k; ik++){ // loop through categories
            // update allfit for class ik
            mdi.ik = ik;
            fit(t[j*k + ik],xi,p,n,x,&ftemp[ik * n]); 
            for(size_t i=0;i<n;i++) allfit[ik * n + i] += - log(ftemp[ik * n + i]);
            //bd function 
            // cout << "ok 3" << endl;
            mlbd(t[j*k + ik],xi,mdi,mpi,phi,nv,pv,aug,gen);
            // cout << "ok 4" << endl;
            // update allfit with new lambdas
            drlamb(t[j*k + ik],xi,mdi,mpi,gen);
            // cout << "ok 5" << endl;
            fit(t[j*k + ik],xi,p,n,x,&ftemp[ik * n]); // update ftemp, ftemp[i, k] is *(k*n + i)
            for(size_t i=0;i<n;i++) allfit[ik * n + i] += log(ftemp[ik * n + i]);
         }
      } else{ // shared tree
         // cout << "ok 6" << endl;
         // update allfit for class ik
         for(size_t ik = 0; ik < k; ik++){ // loop through categories
            mdi.ik = ik;
            fit(t[j*k + ik],xi,p,n,x,&ftemp[ik * n]);
            for(size_t i=0;i<n;i++) allfit[ik * n + i] += - log(ftemp[ik * n + i]);
         }
         // bd function, shared tree version
         mlbdShrTr(t,j,xi,mdi,mpi,phi,nv,pv,aug,gen);
         // update allfit with new lambdas
         for (size_t ik = 0; ik < k; ik ++) {
            // cout << "ik = " << ik << endl;
            mdi.ik = ik;
            drlamb(t[j*k + ik],xi,mdi,mpi,gen);
            fit(t[j*k + ik],xi,p,n,x,&ftemp[ik * n]); // update ftemp, ftemp[i, k] is *(k*n + i)
            for(size_t i=0;i<n;i++)  allfit[ik * n + i] += log(ftemp[ik * n + i]);
         }
      }
   }
}


void mlbart::drweight(rn& gen, double &save_weight)
{
   // Calculate logloss
   size_t y_i;
   double sum_fits;
   double logloss = 0; // reset logloss
   double prob = 0.0;

   double temp;
   double temp_sum;

   for (size_t i = 0; i < n; i++)
   {
      sum_fits = 0;
      y_i = mdi.y[i];
      for (size_t j = 0; j < k; ++j)
      {
         sum_fits += exp(mdi.f[j * n + i]);
      }
      // calculate logloss
      prob = exp(mdi.f[y_i * n + i]) / sum_fits; // logloss =  - log(p_j)
      logloss += -log(prob);
   }

   logloss = logloss / n;

   double exp_logloss = exp(-1.0 * logloss);
   double exp_logloss_last_sweep = exp(-1.0 * mdi.logloss_last_sweep);
   // double mu = -0.251 + 4.125 * exp_logloss - 15.09 * pow(exp_logloss, 2) + 14.90 * pow(exp_logloss, 3);
   // double mu_last_sweep = -0.251 + 4.125 * exp_logloss_last_sweep - 15.09 * pow(exp_logloss_last_sweep, 2) + 14.90 * pow(exp_logloss_last_sweep, 3);

   double weight_latent_proposal;
   double weight_proposal;
   double value1;
   double value2;
   double value3;
   double value4;
   double mu;
   double mu_last_sweep;

   if (exp_logloss <= 0.6)
   {
      mu = 1.5;
   }
   else
   {
      mu = 29.55 - 119.02 * exp_logloss + 152.95 * pow(exp_logloss, 2) - 60.98 * pow(exp_logloss, 3);
   }

   // sampling weight by random walk
   // MH_step is standard deviation
   // std::normal_distribution<>
   //    dd(0, MH_step);
   // gen.normal(0, MH_step)

   weight_latent_proposal = exp(mu + gen.normal() * mpi.MH_step) + 1.0;

   // double weight_proposal = fabs(weight_latent_proposal - 1.0) + 1.0;

   weight_proposal = weight_latent_proposal < 10.0 ? weight_latent_proposal : 10.0;

   value3 = normal_density(log(mdi.weight_latent - 1.0), mu, mpi.MH_step, true);

   value4 = normal_density(log(weight_latent_proposal - 1.0), mu, mpi.MH_step, true);

   // notice that logloss is negative, multiply by -1 to convert to likelihood
   // double weight, double logloss, double a, size_t n, size_t k)
   value1 = w_likelihood(mdi.weight, logloss, mpi.a, n, k);

   value2 = w_likelihood(weight_proposal, logloss, mpi.a, n, k);

   // this is in log scale
   double ratio = value2 - value1 + value3 - value4 + normal_density(log(weight_latent_proposal - 1.0), log(2.0), 0.5, true) - normal_density(log(mdi.weight_latent - 1.0), log(2.0), 0.5, true);

   ratio = std::min(1.0, exp(ratio));

   // std::uniform_real_distribution<> dis(0.0, 1.0);

   if (gen.uniform() < ratio)
   {
      // accept
      mdi.weight = weight_proposal;
      mdi.weight_latent = weight_latent_proposal;
      mdi.logloss_last_sweep = logloss;
   }
   save_weight = mdi.weight;
   return;
}

