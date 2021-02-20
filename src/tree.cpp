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

#include "tree.h"

//--------------------
// node id
size_t tree::nid() const 
{
   if(!p) return 1; //if you don't have a parent, you are the top
   if(this==p->l) return 2*(p->nid()); //if you are a left child
   else return 2*(p->nid())+1; //else you are a right child
}
//--------------------
tree::tree_p tree::getptr(size_t nid)
{
   if(this->nid() == nid) return this; //found it
   if(l==0) return 0; //no children, did not find it
   tree_p lp = l->getptr(nid);
   if(lp) return lp; //found on left
   tree_p rp = r->getptr(nid);
   if(rp) return rp; //found on right
   return 0; //never found it
}
//--------------------
//add children to  bot node nid
bool tree::birth(size_t nid,size_t v, size_t c, double thetal, double thetar)
{
   tree_p np = getptr(nid);
   if(np==0) {
      cout << "error in birth: bottom node not found\n";
      return false; //did not find note with that nid
   }
   if(np->l!=0) {
      cout << "error in birth: found node has children\n";
      return false; //node is not a bottom node
   }

   //add children to bottom node np
   tree_p l = new tree;
   l->theta=thetal;
   tree_p r = new tree;
   r->theta=thetar;
   np->l=l;
   np->r=r;
   np->v = v; np->c=c;
   l->p = np;
   r->p = np;

   return true;
}
//--------------------
//depth of node
size_t tree::depth()
{
   if(!p) return 0; //no parents
   else return (1+p->depth());
}
//--------------------
//tree size
size_t tree::treesize()
{
   if(l==0) return 1;  //if bottom node, tree size is 1
   else return (1+l->treesize()+r->treesize());
}
//--------------------
//node type
char tree::ntype()
{
   //t:top, b:bottom, n:no grandchildren, i:internal
   if(!p) return 't';
   if(!l) return 'b';
   if(!(l->l) && !(r->l)) return 'n';
   return 'i';
}
//--------------------
//print out tree(pc=true) or node(pc=false) information
void tree::pr(bool pc) 
{
   size_t d = depth();
   size_t id = nid();

   size_t pid;
   if(!p) pid=0; //parent of top node
   else pid = p->nid();

   std::string pad(2*d,' ');
   std::string sp(", ");
   if(pc && (ntype()=='t'))
      cout << "tree size: " << treesize() << std::endl;
   cout << pad << "(id,parent): " << id << sp << pid;
   cout << sp << "(v,c): " << v << sp << c;
   cout << sp << "theta: " << theta;
   cout << sp << "type: " << ntype();
   cout << sp << "depth: " << depth();
   cout << sp << "pointer: " << this << std::endl;

   if(pc) {
      if(l) {
         l->pr(pc);
         r->pr(pc);
      }
   }
}
//--------------------
//kill children of  nog node nid
bool tree::death(size_t nid, double theta)
{
   tree_p nb = getptr(nid);
   if(nb==0) {
      cout << "error in death, nid invalid\n";
      return false;
   }
   if(nb->isnog()) {
      delete nb->l;
      delete nb->r;
      nb->l=0;
      nb->r=0;
      nb->v=0;
      nb->c=0;
      nb->theta=theta;
      return true;
   } else {
      cout << "error in death, node is not a nog node\n";
      return false;
   }
}
//--------------------
//is the node a nog node
bool tree::isnog() 
{
   bool isnog=true;
   if(l) {
      if(l->l || r->l) isnog=false; //one of the children has children.
   } else {
      isnog=false; //no children
   }
   return isnog;
}
//--------------------
size_t tree::nnogs() 
{
   if(!l) return 0; //bottom node
   if(l->l || r->l) { //not a nog
      return (l->nnogs() + r->nnogs());
   } else { //is a nog
      return 1;
   }
}
//--------------------
size_t tree::nbots() 
{
   if(l==0) { //if a bottom node
      return 1;
   } else {
      return l->nbots() + r->nbots();
   }
}
//--------------------
//get bottom nodes
void tree::getbots(npv& bv)
{
   if(l) { //have children
      l->getbots(bv);
      r->getbots(bv);
   } else {
      bv.push_back(this);
   }
}
//--------------------
//get nog nodes
void tree::getnogs(npv& nv)
{
   if(l) { //have children
      if((l->l) || (r->l)) {  //have grandchildren
         if(l->l) l->getnogs(nv);
         if(r->l) r->getnogs(nv);
      } else {
         nv.push_back(this);
      }
   }
}
//--------------------
//get all nodes
void tree::getnodes(npv& v)
{
   v.push_back(this);
   if(l) {
      l->getnodes(v);
      r->getnodes(v);
   }
}
void tree::getnodes(cnpv& v)  const
{
   v.push_back(this);
   if(l) {
      l->getnodes(v);
      r->getnodes(v);
   }
}
//--------------------
tree::tree_p tree::bn(double *x,xinfo& xi)
{
   if(l==0){
    return this;
    } //no children
   //  cout << "compare " << x[v] << " " << xi[v][c] << endl;
   if(x[v] < xi[v][c]) {

      return l->bn(x,xi);
   } else {

      return r->bn(x,xi);
   }
}
//--------------------
//find region for a given variable
void tree::rg(size_t v, int* L, int* U)
{
   if(this->p==0)  {
      return;
   }
   if((this->p)->v == v) { //does my parent use v?
      if(this == p->l) { //am I left or right child
         if((int)(p->c) <= (*U)) *U = (p->c)-1;
         p->rg(v,L,U);
      } else {
         if((int)(p->c) >= *L) *L = (p->c)+1;
         p->rg(v,L,U);
      }
   } else {
      p->rg(v,L,U);
   }
}
//--------------------
//cut back to one node
void tree::tonull()
{
   size_t ts = treesize();
   //loop invariant: ts>=1
   while(ts>1) { //if false ts=1
      npv nv;
      getnogs(nv);
      for(size_t i=0;i<nv.size();i++) {
         delete nv[i]->l;
         delete nv[i]->r;
         nv[i]->l=0;
         nv[i]->r=0;
      }
      ts = treesize(); //make invariant true
   }
   theta=0.0;
   v=0;c=0;
   p=0;l=0;r=0;
}
//--------------------
//copy tree tree o to tree n
void tree::cp(tree_p n, tree_cp o)
//assume n has no children (so we don't have to kill them)
//recursion down
{
   if(n->l) {
      cout << "cp:error node has children\n";
      return;
   }

   n->theta = o->theta;
   n->v = o->v;
   n->c = o->c;

   if(o->l) { //if o has children
      n->l = new tree;
      (n->l)->p = n;
      cp(n->l,o->l);
      n->r = new tree;
      (n->r)->p = n;
      cp(n->r,o->r);
   }
}
//--------------------------------------------------
//operators
tree& tree::operator=(const tree& rhs)
{
   if(&rhs != this) {
      tonull(); //kill left hand side (this)
      cp(this,&rhs); //copy right hand side to left hand side
   }
   return *this;
}
//--------------------------------------------------
//functions
std::ostream& operator<<(std::ostream& os, const tree& t)
{
   tree::cnpv nds;
   t.getnodes(nds);
   size_t theta_size = 1;
   os << theta_size << std::endl;
   os << nds.size() << std::endl;
   for(size_t i=0;i<nds.size();i++) {
      os << nds[i]->nid() << " ";
      os << nds[i]->getv() << " ";
      os << nds[i]->getc() << " ";
      os << nds[i]->gettheta() << std::endl;
   }
   return os;
}
std::istream& operator>>(std::istream& is, tree& t)
{
   size_t tid,pid; //tid: id of current node, pid: parent's id
   std::map<size_t,tree::tree_p> pts;  //pointers to nodes indexed by node id
   size_t nn; //number of nodes

   t.tonull(); // obliterate old tree (if there)

   //read size of theta
   size_t theta_size;
   is >> theta_size;

   //read number of nodes----------
   is >> nn;
   if(!is) {
      //cout << ">> error: unable to read number of nodes" << endl;
      return is;
   }

   //read in vector of node information----------
   std::vector<node_info> nv(nn);
   for(size_t i=0;i!=nn;i++) {
      is >> nv[i].id >> nv[i].v >> nv[i].c >> nv[i].theta;
      // cout << "values read in " << nv[i].id << " " << nv[i].v << " " << nv[i].c << " " << nv[i].theta << endl;
      if(!is) {
         //cout << ">> error: unable to read node info, on node  " << i+1 << endl;
         return is;
      }
   }
   //first node has to be the top one
   pts[1] = &t; //careful! this is not the first pts, it is pointer of id 1.
   t.setv(nv[0].v); t.setc(nv[0].c); t.settheta(nv[0].theta);
   t.p=0;

   //now loop through the rest of the nodes knowing parent is already there.
   for(size_t i=1;i!=nv.size();i++) {
      tree::tree_p np = new tree;
      np->v = nv[i].v; np->c=nv[i].c; np->theta=nv[i].theta;
      tid = nv[i].id;
      pts[tid] = np;
      pid = tid/2;
      // set pointers
      if(tid % 2 == 0) { //left child has even id
         pts[pid]->l = np;
      } else {
         pts[pid]->r = np;
      }
      np->p = pts[pid];
   }
   return is;
}
// 
std::istream& operator>>(std::istream& is, std::vector<tree> &tmat)
{
   // load trees
   // load parameters from XBART, check whether they match BART parameters or not
   bool separate_tree;
   size_t num_class, num_sweeps, num_trees, p;
   is >> separate_tree >> num_class >> num_sweeps >> num_trees >> p;

   cout << "loaded data " << separate_tree << " " << num_class << " " << num_sweeps << " " << num_trees << " " << p << endl; 
   
   size_t tid,pid; //tid: id of current node, pid: parent's id
   size_t nn; //number of nodes
   size_t theta_size;
   size_t count;
   double temp;

   // tmat is a vector of num_class * num_trees trees
   // it list the first tree of all classes first, then second tree of all classes ... 
   // tmat[itree * num_trees + iclass]

   for(size_t itree = 0; itree < num_trees; itree++)
   {
      for(size_t iclass = 0; iclass < num_class; iclass++)
      {
         cout << "loop index " << itree << " " << num_trees << " " << iclass << " " << num_class << endl;
         tmat[itree * num_trees + iclass].tonull(); // obliterate old tree (if there)

         //read size of theta
         is >> theta_size;

         //read number of nodes----------
         is >> nn;
         cout << "size of loaded tree " << itree << " " << iclass << " " << nn << endl;
         cout << "size of theta " << theta_size << endl;
         if(!is) {
            cout << ">> error: unable to read number of nodes" << endl; 
            return is;
         }

         //read in vector of node information----------
         std::vector<node_info> nv(nn);
         for(size_t i=0;i!=nn;i++) {
            // is >> nv[i].id >> nv[i].v >> nv[i].c >> nv[i].theta;

            is >> nv[i].id >> nv[i].v >> nv[i].c;

            cout << "node info of " << i << " node " << nv[i].id << " " << nv[i].v << " " << nv[i].c << endl;

            for(size_t kk = 0; kk < theta_size; kk++)
            {  
               // XBART returns a entire vector of theta, take the one we need
               if(kk == iclass){
                  is >> nv[i].theta;
               }else{
                  is >> temp;
               }
            }
            cout << "loaded theta " << nv[i].theta << endl;

            // cout << "values read in " << nv[i].id << " " << nv[i].v << " " << nv[i].c << " " << nv[i].theta << endl;
            if(!is) {
               cout << ">> error: unable to read node info, on node  " << i+1 << endl;
               return is;
            }
         }
         //first node has to be the top one
         std::map<size_t,tree::tree_p> pts;  //pointers to nodes indexed by node id
      
         pts[1] = &(tmat[itree * num_trees + iclass]); //careful! this is not the first pts, it is pointer of id 1.
         tmat[itree * num_trees + iclass].setv(nv[0].v); 
         tmat[itree * num_trees + iclass].setc(nv[0].c); 
         tmat[itree * num_trees + iclass].settheta(nv[0].theta);
         tmat[itree * num_trees + iclass].setp(0);

         //now loop through the rest of the nodes knowing parent is already there.
         for(size_t i=1;i!=nv.size();i++) {
            tree::tree_p np = &(tmat[itree * num_trees + iclass]);
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
      }
   }
   return is;
}
//--------------------
//add children to bot node *np
void tree::birthp(tree_p np,size_t v, size_t c, double thetal, double thetar)
{
   tree_p l = new tree;
   l->theta=thetal;
   tree_p r = new tree;
   r->theta=thetar;
   np->l=l;
   np->r=r;
   np->v = v; np->c=c;
   l->p = np;
   r->p = np;
}
//--------------------
//kill children of  nog node *nb
void tree::deathp(tree_p nb, double theta)
{
   delete nb->l;
   delete nb->r;
   nb->l=0;
   nb->r=0;
   nb->v=0;
   nb->c=0;
   nb->theta=theta;
}
void tree::copy_only_root(tree_p o)
//assume n has no children (so we don't have to kill them)
//NOT LIKE cp() function
//this function pointer new root to the OLD structure
{
    this->v = o->v;
    this->c = o->c;
    this->theta = o->theta;

    if (o->l)
    {
        // keep the following structure, rather than create a new tree in memory
        this->l = o->l;
        this->r = o->r;
        // also update pointers to parents
        this->l->p = this;
        this->r->p = this;
    }
    else
    {
        this->l = 0;
        this->r = 0;
    }
}
size_t tree::getbadcut(size_t v){
  tree_p par=this->getp();
  if(par->getv()==v)
    return par->getc();
  else
    return par->getbadcut(v);
}

#ifndef NoRcpp   
// instead of returning y.test, let's return trees
// this conveniently avoids the need for x.test
// loosely based on pr() 
// create an efficient list from a single tree
// tree2list calls itself recursively
Rcpp::List tree::tree2list(xinfo& xi, double center, double scale) {
  Rcpp::List res;

  // five possible scenarios
  if(l) { // tree has branches
    //double cut=xi[v][c];
    size_t var=v, cut=c;

    var++; cut++; // increment from 0-based (C) to 1-based (R) array index

    if(l->l && r->l)         // two sub-trees
      res=Rcpp::List::create(Rcpp::Named("var")=(int)var,
			     //Rcpp::Named("cut")=cut,
			     Rcpp::Named("cut")=(int)cut,
			     Rcpp::Named("type")=1,
			     Rcpp::Named("left")= l->tree2list(xi, center, scale),
			     Rcpp::Named("right")=r->tree2list(xi, center, scale));   
    else if(l->l && !(r->l)) // left sub-tree and right terminal
      res=Rcpp::List::create(Rcpp::Named("var")=(int)var,
			     //Rcpp::Named("cut")=cut,
			     Rcpp::Named("cut")=(int)cut,
			     Rcpp::Named("type")=2,
			     Rcpp::Named("left")= l->tree2list(xi, center, scale),
			     Rcpp::Named("right")=r->gettheta()*scale+center);    
    else if(!(l->l) && r->l) // left terminal and right sub-tree
      res=Rcpp::List::create(Rcpp::Named("var")=(int)var,
			     //Rcpp::Named("cut")=cut,
			     Rcpp::Named("cut")=(int)cut,
			     Rcpp::Named("type")=3,
			     Rcpp::Named("left")= l->gettheta()*scale+center,
			     Rcpp::Named("right")=r->tree2list(xi, center, scale));
    else                     // no sub-trees 
      res=Rcpp::List::create(Rcpp::Named("var")=(int)var,
			     //Rcpp::Named("cut")=cut,
			     Rcpp::Named("cut")=(int)cut,
			     Rcpp::Named("type")=0,
			     Rcpp::Named("left")= l->gettheta()*scale+center,
			     Rcpp::Named("right")=r->gettheta()*scale+center);
  }
  else // no branches
    res=Rcpp::List::create(Rcpp::Named("var")=0, // var=0 means root
			   //Rcpp::Named("cut")=0.,
			   Rcpp::Named("cut")=0,
			   Rcpp::Named("type")=0,
			   Rcpp::Named("left") =theta*scale+center,
			   Rcpp::Named("right")=theta*scale+center);

  return res;
}

// for one tree, count the number of branches for each variable
Rcpp::IntegerVector tree::tree2count(size_t nvar) {
  Rcpp::IntegerVector res(nvar);

  if(l) { // tree branches
    res[v]++;
    
    if(l->l) res+=l->tree2count(nvar); // if left sub-tree
    if(r->l) res+=r->tree2count(nvar); // if right sub-tree
  } // else no branches and nothing to do

  return res;
}
#endif

