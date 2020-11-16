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

#include "multinomialbart.h"

//--------------------------------------------------
void multinomialbart::pr()
{
   cout << "+++++multinomialbart object:\n";
   bart::pr();
}
//--------------------------------------------------
void multinomialbart::draw(double *sigma, rn &gen)
{
   for (size_t j = 0; j < m; j++)
   {
      fit(t[j], xi, p, n, x, ftemp);
      for (size_t k = 0; k < n; k++)
      {
         allfit[k] = allfit[k] - ftemp[k];
         r[k] = y[k] - allfit[k];
      }
      multinomialbd(t[j], xi, di, pi, sigma, nv, pv, aug, gen);
      multinomialdrmu(t[j], xi, di, pi, sigma, gen);
      fit(t[j], xi, p, n, x, ftemp);
      for (size_t k = 0; k < n; k++)
         allfit[k] += ftemp[k];
   }
   if (dartOn)
   {
      draw_s(nv, lpv, theta, gen);
      draw_theta0(const_theta, theta, lpv, a, b, rho, gen);
      for (size_t j = 0; j < p; j++)
         pv[j] = ::exp(lpv[j]);
   }
}

void multinomialbart::draw2(double *sigma, rn &gen, size_t np, double *ixp, double *temp_vec)
{
   for (size_t j = 0; j < m; j++)
   {
      fit(t[j], xi, p, n, x, ftemp);
      for (size_t k = 0; k < n; k++)
      {
         allfit[k] = allfit[k] - ftemp[k];
         r[k] = y[k] - allfit[k];
      }

         this->predict(p, np, ixp, temp_vec);
   cout << "m is " << m << " test 2 " << temp_vec[0] << ' ' << temp_vec[1] << ' ' << temp_vec[2] << endl;

      multinomialbd(t[j], xi, di, pi, sigma, nv, pv, aug, gen);

               this->predict(p, np, ixp, temp_vec);
   cout << "m is " << m << " test 3 " << temp_vec[0] << ' ' << temp_vec[1] << ' ' << temp_vec[2] << endl;

      multinomialdrmu(t[j], xi, di, pi, sigma, gen);

               this->predict(p, np, ixp, temp_vec);
   cout << "m is " << m << " test 4 " << temp_vec[0] << ' ' << temp_vec[1] << ' ' << temp_vec[2] << endl;

      fit(t[j], xi, p, n, x, ftemp);

               this->predict(p, np, ixp, temp_vec);
   cout << "m is " << m << " test 5 " << temp_vec[0] << ' ' << temp_vec[1] << ' ' << temp_vec[2] << endl;

      for (size_t k = 0; k < n; k++)
         allfit[k] += ftemp[k];
   }
   if (dartOn)
   {
      draw_s(nv, lpv, theta, gen);
      draw_theta0(const_theta, theta, lpv, a, b, rho, gen);
      for (size_t j = 0; j < p; j++)
         pv[j] = ::exp(lpv[j]);
   }
}
