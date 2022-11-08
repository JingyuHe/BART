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

#ifndef GUARD_heterbart_h
#define GUARD_heterbart_h
#include "bart.h"
#include "mlbartfuns.h"
#include "mlbd.h"


class mlbart : public bart
{
public:
   mlbart(size_t ik);
   mlbart(size_t ik, size_t im);

   //friends
   friend bool mlbd(tree& x, xinfo& xi, mlogitdinfo& mdi, mlogitpinfo& pi, double *phi, 
	     std::vector<size_t>& nv, std::vector<double>& pv, bool aug, rn& gen);

   friend bool mlbdShrTr(std::vector<tree>& t, size_t tree_iter, xinfo& xi, mlogitdinfo& mdi, mlogitpinfo& pi, double *phi, 
	     std::vector<size_t>& nv, std::vector<double>& pv, bool aug, rn& gen);

   void setprior(double m, double a0, double alpha, double beta, bool update_phi, bool update_weight, double a, double MH_step) {
      mpi.a0=a0; mpi.c = m / pow(a0, 2) + 0.5; mpi.d=m / pow(a0, 2); mpi.logz3 = lgamma(mpi.c) - mpi.c * log(mpi.d); 
      mpi.update_phi = update_phi; mpi.update_weight = update_weight; 
      mpi.a = a; mpi.MH_step = MH_step;
      cout << "prior a0 = " << a0 << ", m = " << m << ", c=" <<  mpi.c << ", d=" << mpi.d << ", z3 = "<< exp(mpi.logz3) << ", update_phi " << update_phi  << ", update_weight " << update_weight << endl;
      pi.alpha = alpha; pi.mybeta = beta;
   }

   void setprior(double m, double a0, double c, double d, double alpha, double beta, bool update_phi, bool update_weight, double a, double MH_step) {
      mpi.a0=a0; mpi.c = c; mpi.d=d; mpi.logz3 = lgamma(mpi.c) - mpi.c * log(mpi.d); 
      mpi.update_phi = update_phi; mpi.update_weight = update_weight; 
      mpi.a = a; mpi.MH_step = MH_step;
      cout << "prior a0 = " << a0 << ", m = " << m << ", c=" <<  mpi.c << ", d=" << mpi.d << ", z3 = "<< exp(mpi.logz3) << ", update_phi " << update_phi  << ", update_weight " << update_weight << endl;
      pi.alpha = alpha; pi.mybeta = beta;
   }

   void setdata(size_t p, size_t n, double *x, double *y, int *nc, bool separate, double weight);
   void predict(size_t p, size_t n, double *x, double *fp, bool normalize);
   void draw(rn& gen);
   void drweight(rn& gen, double &save_weight);

protected:
   bool separate;
   size_t k; // number of categories
   double *phi;
   mlogitdinfo mdi;
   mlogitpinfo mpi;
};



#endif
