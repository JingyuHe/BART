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

#include "mlbd.h"

bool mlbd(tree& x, xinfo& xi, mlogitdinfo& mdi, mlogitpinfo& mpi, double *phi,
	std::vector<size_t>& nv, std::vector<double>& pv, bool aug, rn& gen)
{
   tree::npv goodbots;  //nodes we could birth at (split on)
   double PBx = getpb(x,xi,mpi,goodbots); //prob of a birth at x

   if(gen.uniform() < PBx) { //do birth or death
      //--------------------------------------------------
      //draw proposal
      tree::tree_p nx; //bottom node
      size_t v,c; //variable and cutpoint
      double pr; //part of metropolis ratio from proposal and prior
      bprop(x,xi,mpi,goodbots,PBx,nx,v,c,pr,nv,pv,aug,gen);

      //--------------------------------------------------
      //compute sufficient statistics
      size_t nr,nl; //counts in proposed bots
      double syl, syr; //sum of y in proposed bots
      mlgetsuff(x,nx,v,c,xi,mdi,nl,syl,nr,syr);
      // cout << "suff stats " << "nl=" << nl << ", syl="<<syl << ", nr=" << nr << ", syr="<<syr<<endl;

      //--------------------------------------------------
      //compute alpha
      double alpha=0.0, lalpha=0.0;
      double lhl, lhr, lht;
      if((nl>=5) && (nr>=5)) { //cludge?
         lhl = mllh(nl,syl, mpi.c, mpi.d, mpi.z3);
         lhr = mllh(nr,syr, mpi.c, mpi.d, mpi.z3);
         lht = mllh(nl+nr,syl+syr, mpi.c, mpi.d, mpi.z3);
   
         alpha=1.0;
         lalpha = log(pr) + (lhl+lhr-lht); // + log(sigma);
         lalpha = std::min(0.0,lalpha);
      }
      //--------------------------------------------------
      //try metrop
      double mul,mur; //means for new bottom nodes, left and right
      double uu = gen.uniform();
      bool dostep = (alpha > 0) && (log(uu) < lalpha);
      if(dostep) {
         mul = drawnodelambda(nl,syl, mpi.c, mpi.d,gen);
         mur = drawnodelambda(nr,syr, mpi.c, mpi.d,gen);
         x.birthp(nx,v,c,mul,mur);
	 nv[v]++;
         return true;
      } else {
         return false;
      }
   } else {
      //--------------------------------------------------
      //draw proposal
      double pr;  //part of metropolis ratio from proposal and prior
      tree::tree_p nx; //nog node to death at
      dprop(x,xi,mpi,goodbots,PBx,nx,pr,gen);

      //--------------------------------------------------
      //compute sufficient statistics
      size_t nr,nl; //counts at bots of nx
      double syl, syr; //sum at bots of nx
      mlgetsuff(x, nx->getl(), nx->getr(), xi, mdi, nl, syl, nr, syr);
      // cout << "suff stats " << "nl=" << nl << ", syl="<<syl << ", nr=" << nr << ", syr="<<syr<<endl;

      //--------------------------------------------------
      //compute alpha
      double lhl, lhr, lht;
      lhl = mllh(nl,syl, mpi.c, mpi.d, mpi.z3);
      lhr = mllh(nr,syr, mpi.c, mpi.d, mpi.z3);
      lht = mllh(nl+nr,syl+syr, mpi.c, mpi.d, mpi.z3);

      double lalpha = log(pr) + (lht - lhl - lhr); // - log(sigma);
      lalpha = std::min(0.0,lalpha);

      //--------------------------------------------------
      //try metrop
      double mu;
      if(log(gen.uniform()) < lalpha) {
         mu = drawnodelambda(nl+nr,syl+syr, mpi.c, mpi.d,gen);
	 nv[nx->getv()]--;
         x.deathp(nx,mu);
         return true;
      } else {
         return false;
      }
   }
}
