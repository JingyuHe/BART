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
mlbart::mlbart(size_t ik):bart(100 * ik), k(ik), mpi(), mdi(), phi(0) {}
mlbart::mlbart(size_t ik, size_t im):bart(im * ik), k(ik), mpi(), mdi(), phi(0) {} // construct m by k trees, im-th tree for class ik is t[im*k + ik]


void mlbart::setdata(size_t p, size_t n, double *x, double *y, int *nc)
{
   this->p=p; this->n=n; this->x=x; this->y=y;
   if(xi.size()==0) makexinfo(p,n,&x[0],xi,nc);

   // initialize theta value for all trees to be 1;
   for (size_t j = 0; j < t.size(); j++) t[j].settheta(1.);

   if(allfit) delete[] allfit;
   allfit = new double[n*k]; // n by k with 1 ... n for class 0, n+1, ..., n+n for class 1, etc.
   predict(p,n,x,allfit);

   if(r) delete[] r;
   r = new double[n*k]; 

   if(ftemp) delete[] ftemp;
   ftemp = new double[n*k]; // n by k with 1 ... n for class 0, n+1, ..., n+n for class 1, etc.

   if(phi) delete[] phi;
   phi = new double[n];

   mdi.n=n; mdi.p=p; mdi.x = &x[0]; mdi.y=y;
   mdi.k = k; mdi.phi = phi; mdi.f = allfit; mdi.ik = 0;
   for(size_t j=0;j<p;j++){
     nv.push_back(0);
     pv.push_back(1/(double)p);
   }
}
// --------------------------------------------------
void mlbart::predict(size_t p, size_t n, double *x, double *fp)
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
   cout << "finish fit" << endl;
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

   delete[] fptemp;
}
// --------------------------------------------------

void mlbart::draw(rn& gen)
{
   drphi(phi, allfit, n, k, gen);
   for(size_t j=0; j< (size_t) m/k;j++) { 
      for(size_t ik = 0; ik < k; ik++){ // loop through categories
        mdi.ik = ik;
        cout << "tree " << j << ", class " << ik << endl;
        fit(t[j*k + ik],xi,p,n,x,&ftemp[ik*n]);
        for(size_t i=0;i<n;i++) {
            allfit[ik*n + i] = allfit[ik*n + i]/ftemp[ik*n + i];
            // r[ik*n + i] = y[ik*n + i]-allfit[ik*n + i];
            if (isnan(allfit[ik*n + i])) {cout << "allfit " << ik << "*" << n << " + " << i << " is nan, ftemp = " << ftemp[ik*n+i] << endl; exit(1);}
        }
        cout << "mlbd" << endl;
        mlbd(t[j*k + ik],xi,mdi,mpi,phi,nv,pv,aug,gen);
        cout << "drlamb" << endl;
        drlamb(t[j*k + ik],xi,mdi,mpi,gen);
        cout << "fit" << endl;
        fit(t[j*k + ik],xi,p,n,x,&ftemp[ik*n]); // update ftemp, ftemp[i, k] is *(k*n + i)
        for(size_t i=0;i<n;i++) 
        {
           allfit[ik*n + i] *= ftemp[ik*n + i];
           if (isnan(allfit[ik*n + i])) {cout << "allfit " << ik << "*" << n << " + " << i << " is nan, ftemp = " << ftemp[ik*n+i] << endl; exit(1);}
        }
        
      }
   }
//    What is dartOn?
//    if(dartOn) {
//      draw_s(nv,lpv,theta,gen);
//      draw_theta0(const_theta,theta,lpv,a,b,rho,gen);
//      for(size_t j=0;j<p;j++) pv[j]=::exp(lpv[j]);
//    }
}