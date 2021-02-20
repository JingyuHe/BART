#define BOOST_MATH_OVERFLOW_ERROR_POLICY errno_on_error

#include "mlbartfuns.h"
#include <boost/math/special_functions/bessel.hpp>



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
void mlgetsuff(tree& x, tree::tree_p nx, size_t v, size_t c, xinfo& xi, mlogitdinfo& mdi, size_t& nl, double& syl, size_t& nr, double& syr)
{
   double *xx;//current x
   nl=0; syl=0.0;
   nr=0; syr=0.0;

   for(size_t i=0;i<mdi.n;i++) {
      xx = mdi.x + i*mdi.p;
      if(nx==x.bn(xx,xi)) { //does the bottom node = xx's bottom node
         if(xx[v] < xi[v][c]) {
            if (mdi.y[i] == mdi.ik) {nl++;} // does xx belong to category ik
            syl += mdi.phi[i] * exp(mdi.f[mdi.ik * mdi.n + i]); // mdi.f = allfit = f_(h)
          } else {
            if (mdi.y[i] == mdi.ik) {nr++;}
            syr += mdi.phi[i] * exp(mdi.f[mdi.ik * mdi.n + i]);
          }
      }
   }

}

//--------------------------------------------------
//compute n and \sum y_i for left and right bots
void mlgetsuff(tree& x, tree::tree_p l, tree::tree_p r, xinfo& xi, mlogitdinfo& mdi, size_t& nl, double& syl, size_t& nr, double& syr)
{
   double *xx;//current x
   nl=0; syl=0.0;
   nr=0; syr=0.0;

   for(size_t i=0;i<mdi.n;i++) {
      xx = mdi.x + i*mdi.p;
      tree::tree_cp bn = x.bn(xx,xi);
      if(bn==l) {
        if (mdi.y[i] == mdi.ik) {nl++;}
        syl += mdi.phi[i] * exp(mdi.f[mdi.ik * mdi.n + i]);
      }
      if(bn==r) {
        if (mdi.y[i] == mdi.ik) {nr++;}
        syr += mdi.phi[i] * exp(mdi.f[mdi.ik * mdi.n + i]);
      }
   }
}

//--------------------------------------------------
//get sufficients stats for all bottom nodes, this way just loop through all the data once.
void mlallsuff(tree& x, xinfo& xi, mlogitdinfo& mdi, tree::npv& bnv, std::vector<size_t>& nv, std::vector<double>& syv)
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

   std::map<tree::tree_cp,size_t> bnmap;
   for(bvsz i=0;i!=bnv.size();i++) {bnmap[bnv[i]]=i;nv[i]=0;syv[i]=0.0;}

   for(size_t i=0;i<mdi.n;i++) {
      xx = mdi.x + i*mdi.p;
      tbn = x.bn(xx,xi);
      ni = bnmap[tbn];
      if (mdi.y[i] == mdi.ik) {++(nv[ni]);}
      syv[ni] += mdi.phi[i] * exp(mdi.f[mdi.ik * mdi.n + i]);
    //   cout << "phi = " << mdi.phi[i] << "; f = " << mdi.f[mdi.ik*mdi.n + i] << endl;
   }
}
//--------------------------------------------------
// draw all the bottom node lambda's
void drlamb(tree& t, xinfo& xi, mlogitdinfo& mdi, mlogitpinfo& mpi, rn& gen)
{
   tree::npv bnv;
   std::vector<size_t> nv;
   std::vector<double> syv;
   mlallsuff(t,xi,mdi,bnv,nv,syv);

   for(tree::npv::size_type i=0;i!=bnv.size();i++) 
      bnv[i]->settheta(drawnodelambda(nv[i],syv[i],mpi.c,mpi.d,gen));
}

//lh, replacement for lil that only depends on sum y.
double mllh(size_t n, double sy, double c, double d, double z3)
{
    double z1 = gignorm(-c + n, 2*d, 2*sy);
    double z2 = gignorm(c + n, 0, 2*(d + sy));
    // double z3 = gignorm(c, 0, 2*d); // should be predefined
   return log((z1 + z2) / 2 / z3);
}
//--------------------------------------------------
//draw one lambda from post 
double drawnodelambda(size_t n, double sy, double c, double d, rn& gen)
{
    /////////////////////////// generalize inversed Gaussian distribution

    // lambda ~ pi*GIG(-c+r, 2d, 2s) + (1-pi)*Gamma(c+r, d+s)
    // pi = Z(-c+r, 2*d, 2*s) / (Z(-c+r, 2d, 2s) + Z(c+r, 0, 2*(d+s)))
    // r = n, s = sy
    double z1 = gignorm(-c+n, 2*d, 2*sy);
    double z2 = gignorm(c+n, 0, 2*(d+sy));
    double _pi =  z1 / (z1+z2);
    if (gen.uniform() < _pi){ // draw from gig(-c+r, 2*d, 2*s)
        double eta = -c + n; 
        double chi = 2*d;
        double psi = 2*sy;
        size_t num_try = 0;
        double u, v, x;

        if ((psi == 0)&&(eta < 0)&&(chi > 0)) return 1/gen.gamma(-eta, chi/2); // if psi == 0, its a inverse gamma distribution invGamma(-eta, chi/2)

        if ((chi == 0)&&(eta > 0)&&(psi > 0)) return gen.gamma(eta, psi/2); // if chi == 0, it's Gamma(eta, psi/2)

        double beta = sqrt(chi*psi);
        if ((eta < 1)&&(eta >= 0)&&(beta <= sqrt(1-eta)*2/3)) {
            /////////////// Rejection method for non-T-concave part ///////////////////////
            // source: https://core.ac.uk/download/pdf/11008021.pdf
            double k1, k2, k3, A1, A2, A3, A, h;
            double m = beta / ((1-eta) + sqrt(pow(1-eta, 2) + pow(beta, 2)));
            double x0 = beta / (1-eta);
            double xs = x0 > 2/beta ? x0 : 2/beta;
            k1 = exp( (eta-1)*log(m) - beta * (m + 1/m) / 2 ); // g(m) = x^(eta-1)*exp(-beta * (m+1/m) / 2)
            A1 = k1 * x0;

            if (x0 < 2/beta) {
                k2 = exp(-beta);
                if (eta == 0) { A2 = k2 * log(2 / pow(beta, 2)); }
                else { A2 = k2 * (pow(2/beta, eta) - pow(x0, eta)) / eta; }
            }else{  k2 = 0; A2 = 0; }
            
            k3 = pow(xs, eta - 1); A3 = 2 * k3 * exp(-xs * beta / 2) / beta;
            A = A1 + A2 + A3;

            while (num_try < 1000){
                u = gen.uniform(); v = gen.uniform() * A;
                if (v <= A1) { x = x0 * v / A1; h = k1; }
                else if (v <= (A1 + A2)) {
                    v = v - A1;
                    if (eta == 0) { x = beta * exp(v * exp(beta)); }
                    else { x = pow( pow(x0, eta) + v * eta / k2, 1/eta ); h = k2 * pow(x, eta - 1);  }
                } else {
                    v = v - A1 - A2;
                    x = - 2 / beta * log( exp( -xs * beta / 2) - v * beta / 2 / k3 ); 
                    h = k3 * exp(-x * beta / 2);
                }
                if (u * h <= exp( (eta - 1)*log(x) - beta * (x + 1/x) / 2 ) ) { return x;} // uh <= g(x, eta, chi , psi)
                else { num_try += 1; }
            }
            // cout << "Warning: Sampling lambda exceeds 1000 iterations in rejection methhod for non-T-concave part" << endl;
            // cout << "eta = " << eta << "; chi = " << chi << "; psi = " << psi << endl; 
            // cout << "c = " << c << "; d = " << d << "; n = " << n << "; sy = " << sy << endl;
            // cout << "k1 = " << k1 << "; k2 = " << k2 << "; k3 = " << k3 << endl;
            // cout << "A1 = " << A1 << "; A2 = " << A2 << "; A3 = " << A3 << "; x = " << x << endl;
            return x;
        }
        
        if ((eta <= 1)&&(eta >= 0)&&(beta <= 1)&&((beta >= 1/2) | (beta >= sqrt(1-eta)*2/3))){
            /////////////// Ratio-of-Uniforms without node shift ///////////////////////
            // source: https://core.ac.uk/download/pdf/11008021.pdf
            double m = beta / ((1-eta) + sqrt((pow(1-eta, 2) + pow(beta, 2))));
            double xp = ((1+eta) + sqrt(pow(1+eta, 2) + pow(beta, 2))) / beta;
            double vp = sqrt( exp( (eta-1)*log(m) - beta * (m + 1/m) / 2 ) ); // sqrt(g(m))
            double up = xp * sqrt( exp( (eta - 1)*log(xp) - beta * (xp + 1/xp) /2 ));

            while (num_try < 1000){
                u = gen.uniform() * up; v = gen.uniform() * vp;
                x = u/v;
                if (pow(v, 2) <= exp( (eta-1)*log(x) - beta * (x + 1/x) /2 )) { return x;}
                else { num_try += 1; } 
            }
            // cout << "Warning: Sampling lambda exceeds 1000 iterations in ratio-of-uniforms without mode shift" << endl;
            // cout << "eta = " << eta << "; chi = " << chi << "; psi = " << psi << endl; 
            // cout << "c = " << c << "; d = " << d << "; n = " << n << "; sy = " << sy << endl;
            // cout << "m = " << m << "; xp = " << xp << "; vp = " << vp << "; up = " << up << "; x = " << x << endl;
            return x;
        }

        /////////////// Ratio-of-Uniforms method ///////////////////////        
        double bx, dx, ib, id, u1, u2;
        bx = sqrt(pow(eta, 2) - 2*eta + chi * psi + 1) + eta - 1 == 0 ? chi/(2-2*eta) : (sqrt(pow(eta, 2) - 2*eta + chi * psi + 1) + eta - 1) / psi;
        dx = sqrt(pow(eta, 2) + 2*eta + chi * psi + 1) + eta + 1 == 0 ? -chi / (2 * eta + 2) : (sqrt(pow(eta, 2) + 2*eta + chi * psi + 1) + eta + 1) / psi;
        ib = sqrt(exp(lgigkernal(bx, eta, chi, psi)));
        id = dx * sqrt(exp(lgigkernal(dx, eta, chi, psi)));

        // if bx or dx is less than 0, likely psi is too closed to zero and caused an rounding error.
        // if ((bx <= 0 | dx <= 0 | id <= 0 | ib <= 0) && (eta < 0)) return 1/gen.gamma(-eta, chi);
         
        while (num_try < 1000)
        {
            u1 = gen.uniform()*ib;
            u2 = gen.uniform()*id;
            if (isinf(u1) | isinf(u2) | isnan(u1) | isnan(u2)) {
               //  cout << "u1 = " << u1 << "; u2 = " << u2 << endl;
               //  cout << "bx = " << ib << "; ib = " << ib << "; dx = " << dx << "; id = " << id << endl;
               //  cout << "eta = " << eta << "; chi = " << chi << "; psi = " << psi << endl; 
               //  cout << "c = " << c << "; d = " << d << "; n = " << n << "; sy = " << sy << endl;
                exit(1);
                }
            if (2*log(u1) <= lgigkernal(u2/u1, eta, chi, psi)) {return u2 / u1; }
            else {num_try += 1;}
        }
   
      // When psi is extremely small and the sampling can not converge, it will eventually cause overflows
      // So we try to consider psi as the case psi == 0 
      if (eta < 0) {return 1/gen.gamma(-eta, chi/2);}
      else {
         //   cout << "Warning: Sampling lambda exceeds 1000 iterations." << endl;
         //   cout << "ib = " << ib << "; bx = " << bx << "; id = " << id << "; dx = " << dx << endl;
         //   cout << "u1 = " << u1 << "; u2 = " << u2 << "; u2/u1 = " << u2/u1 << endl;
         //   cout << "eta = " << eta << "; chi = " << chi << "; psi = " << psi << endl; 
         //   cout << "c = " << c << "; d = " << d << "; n = " << n << "; sy = " << sy << endl;
           return u2/u1; 
      }
    }
    else { // draw from gig(c+r, 0, 2*(d+s)) or equivalently gamma(c+r, d+s)
        return gen.gamma(c+n, d+sy);
    }
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
//
