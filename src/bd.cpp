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

#include "bd.h"

// core function of birth and death of a new tree
// This package does not have SWAP and CHANGE proposal
bool bd(tree& x, xinfo& xi, dinfo& di, pinfo& pi, double sigma, 
	std::vector<size_t>& nv, std::vector<double>& pv, bool aug, rn& gen)
{
   // here x is the tree to update
   // xi is covariate data
   // pi prior info

   tree::npv goodbots;  //nodes we could birth at (split on)

   // this function list all bottom nodes candidates for split, return PBx prob of each node 
   double PBx = getpb(x,xi,pi,goodbots); //prob of a birth at x

   if(gen.uniform() < PBx) { 

      //--------------------------------------------------
      //draw proposal
      tree::tree_p nx; //bottom node
      size_t v,c; //variable and cutpoint
      double pr; //part of metropolis ratio from proposal and prior
      bprop(x,xi,pi,goodbots,PBx,nx,v,c,pr,nv,pv,aug,gen);

      //--------------------------------------------------
      //compute sufficient statistics
      size_t nr,nl; //counts in proposed bots
      double syl, syr; //sum of y in proposed bots, sufficient statistics being used later
      getsuff(x,nx,v,c,xi,di,nl,syl,nr,syr);

      //--------------------------------------------------
      //compute alpha
      double alpha=0.0, lalpha=0.0;
      double lhl, lhr, lht;

      // here alpha is an indicator, if alpha = 0, means that the following loop is not executed
      // this proposal generates at least one too small leaf, and will not be accepted
      if((nl>=5) && (nr>=5)) { //cludge?

         // calculate likelihood !!!
         // split versus no split
         lhl = lh(nl,syl,sigma,pi.tau);
         lhr = lh(nr,syr,sigma,pi.tau);
         lht = lh(nl+nr,syl+syr,sigma,pi.tau);

         // the Metropolis Hastings ratio
         alpha=1.0;
         lalpha = log(pr) + (lhl+lhr-lht) + log(sigma);
         lalpha = std::min(0.0,lalpha);
      }

      //--------------------------------------------------
      //try metrop
      double mul,mur; //means for new bottom nodes, left and right
      double uu = gen.uniform();
      // if the proposal is accepted
      bool dostep = (alpha > 0) && (log(uu) < lalpha);
      if(dostep) {
         mul = drawnodemu(nl,syl,pi.tau,sigma,gen);
         mur = drawnodemu(nr,syr,pi.tau,sigma,gen);

         // attach the new nodes to the child, with the decision rules and leaf parameters
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
      // death proposal
      dprop(x,xi,pi,goodbots,PBx,nx,pr,gen);

      //--------------------------------------------------
      //compute sufficient statistics
      size_t nr,nl; //counts at bots of nx
      double syl, syr; //sum at bots of nx
      getsuff(x, nx->getl(), nx->getr(), xi, di, nl, syl, nr, syr);

      //--------------------------------------------------
      //compute alpha
      double lhl, lhr, lht;
      lhl = lh(nl,syl,sigma,pi.tau);
      lhr = lh(nr,syr,sigma,pi.tau);
      lht = lh(nl+nr,syl+syr,sigma,pi.tau);

      double lalpha = log(pr) + (lht - lhl - lhr) - log(sigma);
      lalpha = std::min(0.0,lalpha);

      //--------------------------------------------------
      //try metrop
      double mu;
      if(log(gen.uniform()) < lalpha) {
         mu = drawnodemu(nl+nr,syl+syr,pi.tau,sigma,gen);
	 nv[nx->getv()]--;
         x.deathp(nx,mu);
         return true;
      } else {
         return false;
      }
   }
}
