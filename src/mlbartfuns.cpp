#define BOOST_MATH_OVERFLOW_ERROR_POLICY errno_on_error

#include "mlbartfuns.h"
#include <boost/math/special_functions/bessel.hpp>
#include <gsl/gsl_sf_bessel.h>
#include <Rcpp.h>


using namespace boost::math;
//--------------------------------------------------
// load classification tree from output of XBART for warm start
void load_classification_tree(std::istream& is, tree &t, size_t itree, size_t iclass, xinfo& xi)
{  
   size_t tid,pid; //tid: id of current node, pid: parent's id
   std::map<size_t,tree::tree_p> pts;  //pointers to nodes indexed by node id
   size_t nn; //number of nodes
   double temp = 0.0;
   size_t temp_index = 0;
   t.tonull();
   size_t theta_size;
   is >> theta_size;

   double temp_c = 0.0;
// cout << "size of xi " << xi.size() << " " << xi[0].size() << endl;
   //read number of nodes----------
   is >> nn;
   // cout << "size of loaded tree " << itree << " " << iclass << " " << nn << endl;
   // cout << "size of theta " << theta_size << endl;
   if(!is) {
      // cout << ">> error: unable to read number of nodes" << endl; 
      return;
   }

   //read in vector of node information----------
   std::vector<node_info> nv(nn);
   for(size_t i=0;i!=nn;i++) 
   {
      // is >> nv[i].id >> nv[i].v >> nv[i].c >> nv[i].theta;

      // the raw output of XBART is raw value of cutpoints
      // BART define cutpoint by its index in the xi matrix
      is >> nv[i].id >> nv[i].v >> temp_c; // >> nv[i].c;

      // search index in xi for the cutpoint
      temp_index = 0;

      while(xi[nv[i].v][temp_index] <= temp_c){
         temp_index ++ ;
      }

      if(temp_index >= xi[0].size())
      {
         // avoid overflow
         temp_index = xi[0].size() - 1;
      }

      nv[i].c = temp_index;
      
      if(nv[i].v == 0 && temp_c == 0)
      {
         // this is a leaf node
         nv[i].c = 0;
      }


      // cout << "cutpoint of XBART: " << temp_c << " cutpoint loaded: " << xi[nv[i].v][nv[i].c] << endl;

      for(size_t kk = 0; kk < theta_size; kk++)
      {  
         // XBART returns a entire vector of theta, take the one we need
         if(kk == iclass){
            is >> nv[i].theta;
         }else{
            is >> temp;
         }
      }
      // cout << "loaded theta " << nv[i].theta << endl;

      // cout << "values read in " << nv[i].id << " " << nv[i].v << " " << nv[i].c << " " << nv[i].theta << endl;
      if(!is) {
         // cout << ">> error: unable to read node info, on node  " << i+1 << endl;
         return;
      }
   }
   //first node has to be the top one
   pts[1] = &(t); //careful! this is not the first pts, it is pointer of id 1.
   t.setv(nv[0].v); 
   t.setc(nv[0].c); 
   t.settheta(nv[0].theta);
   t.setp(0);

   //now loop through the rest of the nodes knowing parent is already there.
   for(size_t i=1;i!=nv.size();i++) 
   {
      tree::tree_p np = new tree;
      np->setv(nv[i].v); np->setc(nv[i].c); np->settheta(nv[i].theta);
      tid = nv[i].id;
      pts[tid] = np;
      pid = tid/2;
      // set pointers
      if(tid % 2 == 0) { //left child has even id
         pts[pid]->setl(np);
      } else {
         pts[pid]->setr(np);
      }
      np->setp(pts[pid]);
   }
   return;
}



//--------------------------------------------------
//compute r = \sum y_ij and s = \sum phi_i f_(t)(x) for left and right give bot and v,c
void mlgetsuff(tree& x, tree::tree_p nx, size_t v, size_t c, xinfo& xi, mlogitdinfo& mdi, double& nl, double& syl, double& nr, double& syr)
{
   double *xx;//current x
   nl=0; syl=0.0;
   nr=0; syr=0.0;

   for(size_t i=0;i<mdi.n;i++) {
      xx = mdi.x + i*mdi.p;
      if(nx==x.bn(xx,xi)) { //does the bottom node = xx's bottom node
         if(xx[v] < xi[v][c]) {
            if (mdi.y[i] == mdi.ik) {nl+= mdi.weight;} // does xx belong to category ik
            syl += mdi.weight * mdi.phi[i] * exp(mdi.f[mdi.ik * mdi.n + i]); // mdi.f = allfit = f_(h)
          } else {
            if (mdi.y[i] == mdi.ik) {nr+= mdi.weight;}
            syr += mdi.weight * mdi.phi[i] * exp(mdi.f[mdi.ik * mdi.n + i]);
          }
      }
   }

}

//--------------------------------------------------
//compute n and \sum y_i for left and right bots
void mlgetsuff(tree& x, tree::tree_p l, tree::tree_p r, xinfo& xi, mlogitdinfo& mdi, double& nl, double& syl, double& nr, double& syr)
{
   double *xx;//current x
   nl=0; syl=0.0;
   nr=0; syr=0.0;

   for(size_t i=0;i<mdi.n;i++) {
      xx = mdi.x + i*mdi.p;
      tree::tree_cp bn = x.bn(xx,xi);
      if(bn==l) {
        if (mdi.y[i] == mdi.ik) {nl+= mdi.weight;}
        syl += mdi.weight * mdi.phi[i] * exp(mdi.f[mdi.ik * mdi.n + i]);
      }
      if(bn==r) {
        if (mdi.y[i] == mdi.ik) {nr+= mdi.weight;}
        syr += mdi.weight * mdi.phi[i] * exp(mdi.f[mdi.ik * mdi.n + i]);
      }
   }
}

//--------------------------------------------------
//get sufficients stats for all bottom nodes, this way just loop through all the data once.
void mlallsuff(tree& x, xinfo& xi, mlogitdinfo& mdi, tree::npv& bnv, std::vector<double>& nv, std::vector<double>& syv)
{
   tree::tree_cp tbn; //the pointer to the bottom node for the current observations
   size_t ni;         //the  index into vector of the current bottom node
   double *xx;        //current x

   bnv.clear();
   x.getbots(bnv);

   typedef tree::npv::size_type bvsz;
   bvsz nb = bnv.size();
   nv.resize(nb);
   syv.resize(nb);
   // cout << "bnv size " << nb << endl;
   std::map<tree::tree_cp,size_t> bnmap;
   for(bvsz i=0;i!=bnv.size();i++) {bnmap[bnv[i]]=i;nv[i]=0;syv[i]=0.0;}
   for(size_t i=0;i<mdi.n;i++) {
      xx = mdi.x + i*mdi.p;
      tbn = x.bn(xx,xi);
      ni = bnmap[tbn];
      if (mdi.y[i] == mdi.ik) {nv[ni]+=mdi.weight;}
      syv[ni] += mdi.weight * mdi.phi[i] * exp(mdi.f[mdi.ik * mdi.n + i]);
   }
   // cout << "phi = " << mdi.phi[0] << "; f = " << mdi.f[mdi.ik*mdi.n + 0] << endl;
}
//--------------------------------------------------
// draw all the bottom node lambda's
void drlamb(tree& t, xinfo& xi, mlogitdinfo& mdi, mlogitpinfo& mpi, rn& gen)
{
   tree::npv bnv;
   std::vector<double> nv;
   std::vector<double> syv;
   mlallsuff(t,xi,mdi,bnv,nv,syv);

   for(tree::npv::size_type i=0;i!=bnv.size();i++) 
      bnv[i]->settheta(drawnodelambda(nv[i],syv[i],mpi.c,mpi.d,gen));
}

//lh, replacement for lil that only depends on sum y.
double mllh(size_t n, double sy, double c, double d, double logz3)
{
   //  double z1 = gignorm(-c + n, 2*d, 2*sy);
   //  double z2 = gignorm(c + n, 0, 2*(d + sy));
   //  // double z3 = gignorm(c, 0, 2*d); // should be predefined
   // return log((z1 + z2) / 2 / z3);

   double logz1 = loggignorm(-c + n, 2*d, 2*sy);
   double logz2 = loggignorm(c + n, 0, 2*(d + sy));
   double logminz = logz1 < logz2 ? logz1 : logz2;
   double numrt;

   if (logz1 - logminz > 100) {
         numrt = logz1; // approximate log(exp(x) + 1) = x
   } else if (logz2 - logminz > 100)
   { 
         numrt = logz2;
   } else {
         numrt = log(exp(logz1 - logminz) + exp(logz2 - logminz)) + logminz;
   }
   return numrt - log(2) - logz3;
}
//--------------------------------------------------
//draw one lambda from post 
double drawnodelambda(double n, double sy, double c, double d, rn& gen)
{
   /////////////////////////// generalize inversed Gaussian distribution
   double logz1 = loggignorm(-c+n, 2*d, 2*sy);
   double logz2 = loggignorm(c+n, 0, 2*(d+sy));
   // cout << "z1 = " << z1 << " z2 = " << z2 << endl;
   // double _pi =  z1 / (z1+z2) = 1 / (1 + z2 / z1) = 1 / (1 + exp(log(z2 / z1))) = 1 / (1 + exp(log(z2) - log(z1)))
   double _pi = 1 / (1 + exp(logz2 - logz1));
   double u = gen.uniform();
   double ret;

   double eta, chi, psi;
   Rcpp::Function f("rgig");

    if (u < _pi){ 
      // draw from gig(-c+r, 2*d, 2*s)
      eta = -c + n; 
      chi = 2*d;
      psi = 2*sy;
      Rcpp::NumericVector ret_r = f(1, eta, chi, psi);
      ret = ret_r(0);
    } else { 
      //   ret = gen.gamma(c+n, 1) / (d+sy); 
      eta = c + n; 
      chi = 0;
      psi = 2 * (d+sy);
      Rcpp::NumericVector ret_r = f(1, eta, chi, psi);
      ret = ret_r(0);
    }
   //  cout << "n = " << n << " sy = " << sy  << " ret = " << ret << endl;
    return ret;
}

void drphi(double *phi, double *allfit, size_t n, size_t k, rn& gen)
{
    double sum_fit; 
    for (size_t i = 0; i < n; i++){
        sum_fit = 0.0;
        for (size_t j = 0; j < k; j++){
            sum_fit += exp(allfit[j*n + i]);
        }
        phi[i] = gen.gamma(1, sum_fit); 
    }
}

double w_likelihood(double weight, double logloss, double a, size_t n, size_t k)
{
   double output = n * (std::lgamma(weight + (k - 1) * a + 1) - std::lgamma(weight + 1) - weight * logloss);
   return output;
}

// return the nomalization term for generazlied inverse gaussian (gig) distribution, see mlnomial BART, Jared Murray
double gignorm(double eta, double chi, double psi) 
{ 
    double ret;
    if ((eta > 0)&&(chi==0)&&(psi>0)){
        ret = exp(lgamma(eta) + eta * log(2 / psi));
    }else if ((eta < 0)&&(chi>0)&&(psi==0)){
        ret = exp(lgamma(-eta) - eta * log(2 / chi));
    }else if ((chi>0)&&(psi>0)){
        double bessel_k = cyl_bessel_k(eta, sqrt(chi*psi));
        ret = exp(log(2*bessel_k) - (eta / 2) * log(psi / chi));
    }
    return ret;
}

double loggignorm(double eta, double chi, double psi) 
{ 
    // cout << "eta = " << eta << " chi = " << chi << " psi = " << psi << endl;
    double ret;
    if ((eta > 0)&&(chi==0)&&(psi>0)){
        ret = lgamma(eta) + eta * log(2 / psi);
    }else if ((eta < 0)&&(chi>0)&&(psi==0)){
        ret = (lgamma(-eta) - eta * log(2 / chi));
    }else if ((chi>0)&&(psi>0)){
        // cout << "eta = " << eta << " sqrt(chi*psi) = " << sqrt(chi*psi) << " bessel_k = " << bessel_k;
        double sq = sqrt(chi*psi);
        double lbessel_k = eta > 0 ? gsl_sf_bessel_lnKnu(eta, sq) : log(boost::math::cyl_bessel_k(eta, sq));
        // ret = exp(log(2*bessel_k) - (eta / 2) * log(psi / chi));
        ret = (log(2) + lbessel_k - (eta / 2) * log(psi / chi));
        // cout << " lnKnu = " << lbessel_k <<  " exp(lKn) = " << exp(lbessel_k) << " log(ret) = " << log(2) + lbessel_k - (eta / 2) * log(psi / chi) << " ret = " << ret << endl;
    }
    return ret;
}


double lgigkernal(double x, double eta, double chi, double psi)
{
    // return pow(x, eta-1)*exp(-(chi/x + psi*x)/2);
    return (eta-1)*log(x) - (chi/x + psi*x)/2;
}


double getpbShrTr(std::vector<tree>& trees, size_t tree_iter, size_t k, xinfo& xi, pinfo& pi, std::vector<tree::npv>& goodbots)
{
   double pb;  //prob of birth to be returned
   std::vector<tree::npv> bnv(k); //all the bottom nodes
//    t.getbots(bnv);   
    for (size_t j = 0; j < k; j++) trees[tree_iter * k + j].getbots(bnv[j]);
    for(size_t i=0;i!=bnv[0].size();i++)
    {
        if(cansplit(bnv[0][i],xi))  
        {
            for (size_t j = 0; j < k; j++) goodbots[j].push_back(bnv[j][i]);
        }
    }
        
   if(goodbots[0].size()==0) { //are there any bottom nodes you can split on?
      pb=0.0;
   } else {
      if(trees[tree_iter * k].treesize()==1) pb=1.0; //is there just one node?
      else pb=pi.pb;
   }
   return pb;
}

size_t bpropShrTr(tree& x, xinfo& xi, pinfo& pi, tree::npv& goodbots, double& PBx, tree::tree_p& nx, size_t& v, size_t& c, double& pr, std::vector<size_t>& nv, std::vector<double>& pv, bool aug, rn& gen)
{
    /////////// Same as dprorp from bartfuns, but return ni /////////////////

      //draw bottom node, choose node index ni from list in goodbots
      size_t ni = floor(gen.uniform()*goodbots.size());
      nx = goodbots[ni]; //the bottom node we might birth at

      //draw v,  the variable
      std::vector<size_t> goodvars; //variables nx can split on
      int L,U; //for cutpoint draw
      // Degenerate Trees Strategy (Assumption 2.2)
      if(!aug){
      getgoodvars(nx,xi,goodvars);
	gen.set_wts(pv);
	v = gen.discrete();
	L=0; U=xi[v].size()-1;
	if(!std::binary_search(goodvars.begin(),goodvars.end(),v)){ // if variable is bad
	  c=nx->getbadcut(v); // set cutpoint of node to be same as next highest interior node with same variable
	}
	else{ // if variable is good
	  nx->rg(v,&L,&U);
	  c = L + floor(gen.uniform()*(U-L+1)); // draw cutpoint usual way
	}
      }
      // Modified Data Augmentation Strategy (Mod. Assumption 2.1)
      // Set c_j = s_j*E[G] = s_j/P{picking a good var}
      // where  G ~ Geom( P{picking a good var} )
      else{
	std::vector<size_t> allvars; //all variables
	std::vector<size_t> badvars; //variables nx can NOT split on
	std::vector<double> pgoodvars; //vector of goodvars probabilities (from S, our Dirichlet vector draw)
	std::vector<double> pbadvars; //vector of badvars probabilities (from S,...)
	getgoodvars(nx,xi,goodvars);
	//size_t ngoodvars=goodvars.size();
	size_t nbadvars=0; //number of bad vars
	double smpgoodvars=0.; //P(picking a good var)
	double smpbadvars=0.; //P(picking a bad var)
//	size_t nbaddraws=0; //number of draws at a particular node
	//this loop fills out badvars, pgoodvars, pbadvars, 
	//there may be a better way to do this...
	for(size_t j=0;j<pv.size();j++){
	  allvars.push_back(j);
	  if(goodvars[j-nbadvars]!=j) {
	    badvars.push_back(j);
	    pbadvars.push_back(pv[j]);
	    smpbadvars+=pv[j];
	    nbadvars++;
	  }
	  else {
	    pgoodvars.push_back(pv[j]);
	    smpgoodvars+=pv[j];
	  }
	}
	//set the weights for variable draw and draw a good variable
	gen.set_wts(pgoodvars);
	v = goodvars[gen.discrete()];
	if(nbadvars!=0){ // if we have bad vars then we need to augment, otherwise we skip
	  //gen.set_p(smpgoodvars); // set parameter for G
	  //nbaddraws=gen.geometric(); // draw G = g ~ Geom
	  // for each bad variable, set its c_j equal to its expected count
	  /*
	    gen.set_wts(pbadvars); 
	  for(size_t k=0;k!=nbaddraws;k++) {
	    nv[badvars[gen.discrete()]]++;
	    }
	  */
	  for(size_t j=0;j<nbadvars;j++)
	    nv[badvars[j]]=nv[badvars[j]]+(1/smpgoodvars)*(pv[badvars[j]]/smpbadvars); 	  
	}
/*
      size_t vi = floor(gen.uniform()*goodvars.size()); //index of chosen split variable
      v = goodvars[vi];
*/

      //draw c, the cutpoint
      //int L,U;
      L=0; U = xi[v].size()-1;
      nx->rg(v,&L,&U);
      c = L + floor(gen.uniform()*(U-L+1)); //U-L+1 is number of available split points
      }
      //--------------------------------------------------
      //compute things needed for metropolis ratio

      double Pbotx = 1.0/goodbots.size(); //proposal prob of choosing nx
      size_t dnx = nx->depth();
      double PGnx = pi.alpha/pow(1.0 + dnx,pi.mybeta); //prior prob of growing at nx

      double PGly, PGry; //prior probs of growing at new children (l and r) of proposal
      if(goodvars.size()>1) { //know there are variables we could split l and r on
         PGly = pi.alpha/pow(1.0 + dnx+1.0,pi.mybeta); //depth of new nodes would be one more
         PGry = PGly;
      } else { //only had v to work with, if it is exhausted at either child need PG=0
         if((int)(c-1)<L) { //v exhausted in new left child l, new upper limit would be c-1
            PGly = 0.0;
         } else {
            PGly = pi.alpha/pow(1.0 + dnx+1.0,pi.mybeta);
         }
         if(U < (int)(c+1)) { //v exhausted in new right child r, new lower limit would be c+1
            PGry = 0.0;
         } else {
            PGry = pi.alpha/pow(1.0 + dnx+1.0,pi.mybeta);
         }
      }

      double PDy; //prob of proposing death at y
      if(goodbots.size()>1) { //can birth at y because splittable nodes left
         PDy = 1.0 - pi.pb;
      } else { //nx was the only node you could split on
         if((PGry==0) && (PGly==0)) { //cannot birth at y
            PDy=1.0;
         } else { //y can birth at either l or r
            PDy = 1.0 - pi.pb;
         }
      }

      double Pnogy; //death prob of choosing the nog node at y
      size_t nnogs = x.nnogs();
      tree::tree_p nxp = nx->getp();
      if(nxp==0) { //no parent, nx is the top and only node
         Pnogy=1.0;
      } else {
         if(nxp->ntype() == 'n') { //if parent is a nog, number of nogs same at x and y
            Pnogy = 1.0/nnogs;
         } else { //if parent is not a nog, y has one more nog.
           Pnogy = 1.0/(nnogs+1.0);
         }
      }

      pr = (PGnx*(1.0-PGly)*(1.0-PGry)*PDy*Pnogy)/((1.0-PGnx)*Pbotx*PBx);

      return ni;
}


size_t dpropShrTr(tree& x, xinfo& xi, pinfo& pi,tree::npv& goodbots, double& PBx, tree::tree_p& nx, double& pr, rn& gen)
{
    /////////// Same as dprorp from bartfuns, but return ni /////////////////
    
      //draw nog node, any nog node is a possibility
      tree::npv nognds; //nog nodes
      x.getnogs(nognds);
      size_t ni = floor(gen.uniform()*nognds.size());
      nx = nognds[ni]; //the nog node we might kill children at

      //--------------------------------------------------
      //compute things needed for metropolis ratio

      double PGny; //prob the nog node grows
      size_t dny = nx->depth();
      PGny = pi.alpha/pow(1.0+dny,pi.mybeta);

      //better way to code these two?
      double PGlx = pgrow(nx->getl(),xi,pi);
      double PGrx = pgrow(nx->getr(),xi,pi);

      double PBy;  //prob of birth move at y
      if(nx->ntype()=='t') { //is the nog node nx the top node
         PBy = 1.0;
      } else {
         PBy = pi.pb;
      }

      double Pboty;  //prob of choosing the nog as bot to split on when y
      int ngood = goodbots.size();
      if(cansplit(nx->getl(),xi)) --ngood; //if can split at left child, lose this one
      if(cansplit(nx->getr(),xi)) --ngood; //if can split at right child, lose this one
      ++ngood;  //know you can split at nx
      Pboty=1.0/ngood;

      double PDx = 1.0-PBx; //prob of a death step at x
      double Pnogx = 1.0/nognds.size();

      pr =  ((1.0-PGny)*PBy*Pboty)/(PGny*(1.0-PGlx)*(1.0-PGrx)*PDx*Pnogx);
      return ni;
}

double normal_density(double y, double mean, double var, bool take_log)
{
    // density of normal distribution
    double output = 0.0;

    output = -0.5 * log(2.0 * M_PI * var) - pow(y - mean, 2) / 2.0 / var;
    if (!take_log)
    {
        output = exp(output);
    }
    return output;
}
//
