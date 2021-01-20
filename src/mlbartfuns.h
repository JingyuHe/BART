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

#ifndef GUARD_heterbartfuns_h
#define GUARD_heterbartfuns_h

#include "tree.h"
#include "treefuns.h"
#include "info.h"
#include "bartfuns.h"

void mlgetsuff(tree& x, tree::tree_p nx, size_t v, size_t c, xinfo& xi, mlogitdinfo& di, size_t& nl, double& syl, size_t& nr, double& syr);

void mlgetsuff(tree& x, tree::tree_p l, tree::tree_p r, xinfo& xi, mlogitdinfo& mdi, size_t& nl, double& syl, size_t& nr, double& syr);

void mlallsuff(tree& x, xinfo& xi, mlogitdinfo& di, tree::npv& bnv, std::vector<size_t>& nv, std::vector<double>& syv);

double drawnodelambda(size_t n, double sy, double c, double d, rn& gen);

void drlamb(tree& t, xinfo& xi, mlogitdinfo& mdi, mlogitpinfo& mpi, rn& gen);

double mllh(size_t n, double sy, double c, double d, double z3);

void drphi(double *phi, double *allfit, size_t n, size_t k, rn& gen);

double gignorm(double eta, double chi, double psi);

double lgigkernal(double x, double eta, double chi, double psi);

double getpbShrTr(std::vector<tree>& trees, size_t tree_iter, size_t k, xinfo& xi, pinfo& pi, std::vector<tree::npv>& goodbots);

size_t bpropShrTr(tree& x, xinfo& xi, pinfo& pi, tree::npv& goodbots, double& PBx, tree::tree_p& nx, size_t& v, size_t& c, double& pr, std::vector<size_t>& nv, std::vector<double>& pv, bool aug, rn& gen);

size_t dpropShrTr(tree& x, xinfo& xi, pinfo& pi,tree::npv& goodbots, double& PBx, tree::tree_p& nx, double& pr, rn& gen);

#endif
