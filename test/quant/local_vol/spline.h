/*
 * spline.h
 *
 * simple cubic spline interpolation library without external
 * dependencies
 *
 * ---------------------------------------------------------------------
 * Copyright (C) 2011, 2014, 2016, 2021 Tino Kluge (ttk448 at gmail.com)
 *
 *  This program is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License
 *  as published by the Free Software Foundation; either version 2
 *  of the License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * ---------------------------------------------------------------------
 *
 */

#pragma once

#include <cstdio>
#include <cassert>
#include <cmath>
#include <vector>
#include <algorithm>
#include <optional>

// unnamed namespace only because the implementation is in this
// header file and we don't want to export symbols to the obj files
namespace
{

namespace tk
{

using mat_double = std::vector<std::vector<double>>;

// spline interpolation
class spline
{
public:

    // spline types
    enum spline_type {
        linear = 10,            // linear interpolation
        cspline = 30,           // cubic splines (classical C^2)
    };

    // boundary condition type for the spline end-points
    enum bd_type {
        first_deriv = 1,
        second_deriv = 2
    };

protected:
    mat_double m_xs,m_ys;                   // x,y coordinates of points
    // interpolation parameters
    // f(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3
    // where a_i = y_i, or else it won't go through grid points
    mat_double m_bs,m_cs,m_ds;              // spline coefficients

    std::vector<double> m_c0s;               // for left extrapolation
    std::vector<double> m_left_values, m_right_values;

    spline_type m_type;
    bd_type m_left, m_right;

    size_t m_nxs, m_nys;
    size_t find_closest(size_t idy, double x) const;    // closest id so that m_xs[idx(idy)][idx]<=x

public:
    spline(const mat_double& Xs, const mat_double& Ys,
           spline_type type = cspline,
           bd_type left  = second_deriv,
           std::optional<std::vector<double>> left_values = std::nullopt,
           bd_type right = second_deriv,
           std::optional<std::vector<double>> right_values = std::nullopt
          ):
        m_type(type),
        m_left(left), m_right(right)
    {
        if(left_values.has_value()) {
            m_left_values = left_values.value();
        }

        if(right_values.has_value()) {
            m_right_values = right_values.value();
        }

        this->set_points(Xs, Ys, m_type);
    }

    // set all data points (cubic_spline=false means linear interpolation)
    void set_points(const mat_double& Xs, const mat_double& Ys,
                    spline_type type=cspline);

    // evaluates the spline id_y at point x
    double operator() (size_t idy, double x) const;
};



namespace internal
{

// band matrix solver
class band_matrix
{
private:
    std::vector< std::vector<double> > m_upper;  // upper band
    std::vector< std::vector<double> > m_lower;  // lower band
public:
    band_matrix() {};                             // constructor
    band_matrix(int dim, int n_u, int n_l);       // constructor
    ~band_matrix() {};                            // destructor
    void resize(int dim, int n_u, int n_l);      // init with dim,n_u,n_l
    int dim() const;                             // matrix dimension
    int num_upper() const
    {
        return (int)m_upper.size()-1;
    }
    int num_lower() const
    {
        return (int)m_lower.size()-1;
    }
    // access operator
    double & operator () (int i, int j);            // write
    double   operator () (int i, int j) const;      // read
    // we can store an additional diagonal (in m_lower)
    double& saved_diag(int i);
    double  saved_diag(int i) const;
    void lu_decompose();
    std::vector<double> r_solve(const std::vector<double>& b) const;
    std::vector<double> l_solve(const std::vector<double>& b) const;
    std::vector<double> lu_solve(const std::vector<double>& b,
                                 bool is_lu_decomposed=false);
};

} // namespace internal




// ---------------------------------------------------------------------
// implementation part, which could be separated into a cpp file
// ---------------------------------------------------------------------

// spline implementation
// -----------------------

void spline::set_points(const mat_double& Xs, const mat_double& Ys,
                        spline_type type)
{
    m_nxs = Xs.size();
    m_nys = Ys.size();
    assert(m_nys >= 1 && (m_nxs == 1 || m_nxs == m_nys));

    m_type=type;

    m_xs = Xs;
    m_ys = Ys;

    // boudaries
    if(m_left_values.size() == 0) {
        m_left_values = std::vector<double>(m_nys, 0.);
    }
    assert(m_left_values.size() == m_nys);

    if(m_right_values.size() == 0) {
        m_right_values = std::vector<double>(m_nys, 0.);
    }
    assert(m_right_values.size() == m_nys);


    // coefficients
    m_bs.resize(m_nys);
    m_cs.resize(m_nys);
    m_ds.resize(m_nys);

    m_c0s.resize(m_nys);
    
    
    for(size_t idy = 0; idy < m_nys; idy++) {
        const size_t idx = m_nxs == 1 ? 0 : idy; //idx = 0 always if common X
        const std::vector<double>& x_idx = m_xs.at(idx);
        const std::vector<double>& y_idy = m_ys.at(idy);

        const size_t n = x_idx.size();
        assert(y_idy.size() == n && n > 2);

        // check strict monotonicity of input vector x
        if((m_nxs == 1 && idy == 0) || (m_nxs > 1)) { //do it once if common X
            for(size_t i=0; i<n-1; i++) {
                assert(x_idx[i]<x_idx[i+1]); 
            }
        }
        
        if(type==linear) {
            // linear interpolation
            m_ds[idy] = std::vector(n, 0.);
            m_cs[idy] = std::vector(n, 0.);
            std::vector<double>& b_idy = m_bs[idy];
            b_idy.resize(n);
            for(size_t i=0; i<n-1; i++) {
                b_idy[i]=(y_idy[i+1]-y_idy[i])/(x_idx[i+1]-x_idx[i]);
            }
            // ignore boundary conditions, set slope equal to the last segment
            b_idy[n-1]=b_idy[n-2];

        } else if(type==cspline) {
            // classical cubic splines which are C^2 (twice cont differentiable)
            // this requires solving an equation system

            // setting up the matrix and right hand side of the equation system
            // for the parameters b[]
            internal::band_matrix A(n,1,1);
            std::vector<double>  rhs(n);
            for(int i=1; i<n-1; i++) {
                A(i,i-1)=1.0/3.0*(x_idx[i]-x_idx[i-1]);
                A(i,i)=2.0/3.0*(x_idx[i+1]-x_idx[i-1]);
                A(i,i+1)=1.0/3.0*(x_idx[i+1]-x_idx[i]);

                rhs[i]=(y_idy[i+1]-y_idy[i])/(x_idx[i+1]-x_idx[i]) - (y_idy[i]-y_idy[i-1])/(x_idx[i]-x_idx[i-1]);
            }
        
            // boundary conditionsx
            if(m_left == spline::second_deriv) {
                // 2*c[0] = f''
                A(0,0)=2.0;
                A(0,1)=0.0;
                rhs[0]=m_left_values[idy];
            } else if(m_left == spline::first_deriv) {
                // b[0] = f', needs to be re-expressed in terms of c:
                // (2c[0]+c[1])(x[1]-x[0]) = 3 ((y[1]-y[0])/(x[1]-x[0]) - f')
                A(0,0)=2.0*(x_idx[1]-x_idx[0]);
                A(0,1)=1.0*(x_idx[1]-x_idx[0]);
                rhs[0]=3.0*((y_idy[1]-y_idy[0])/(x_idx[1]-x_idx[0])-m_left_values[idy]);
            } else {
                assert(false);
            }
            if(m_right == spline::second_deriv) {
                // 2*c[n-1] = f''
                A(n-1,n-1)=2.0;
                A(n-1,n-2)=0.0;
                rhs[n-1]=m_right_values[idy];
            } else if(m_right == spline::first_deriv) {
                // b[n-1] = f', needs to be re-expressed in terms of c:
                // (c[n-2]+2c[n-1])(x[n-1]-x[n-2])
                // = 3 (f' - (y[n-1]-y[n-2])/(x[n-1]-x[n-2]))
                A(n-1,n-1)=2.0*(x_idx[n-1]-x_idx[n-2]);
                A(n-1,n-2)=1.0*(x_idx[n-1]-x_idx[n-2]);
                rhs[n-1]=3.0*(m_right_values[idy]-(y_idy[n-1]-y_idy[n-2])/(x_idx[n-1]-x_idx[n-2]));
            } else {
                assert(false);
            }
            
            // solve the equation system to obtain the parameters c[]
            m_cs[idy] = A.lu_solve(rhs);

            // calculate parameters b[] and d[] based on c[]
            std::vector<double>& c_idy = m_cs[idy];
            //c_idy.resize(n);
            std::vector<double>& b_idy = m_bs[idy];
            b_idy.resize(n);
            std::vector<double>& d_idy = m_ds[idy];
            d_idy.resize(n);
            for(size_t i=0; i<n-1; i++) {
                d_idy[i]=1.0/3.0*(c_idy[i+1]-c_idy[i])/(x_idx[i+1]-x_idx[i]);
                b_idy[i]=(y_idy[i+1]-y_idy[i])/(x_idx[i+1]-x_idx[i])
                    - 1.0/3.0*(2.0*c_idy[i]+c_idy[i+1])*(x_idx[i+1]-x_idx[i]);
            }
            // for the right extrapolation coefficients (zero cubic term)
            // f_{n-1}(x) = y_{n-1} + b*(x-x_{n-1}) + c*(x-x_{n-1})^2
            double h=x_idx[n-1]-x_idx[n-2];
            // m_c[n-1] is determined by the boundary condition
            d_idy[n-1]=0.0;
            b_idy[n-1]=3.0*d_idy[n-2]*h*h+2.0*c_idy[n-2]*h+b_idy[n-2];   // = f'_{n-2}(x_{n-1})
            if(m_right==first_deriv)
                c_idy[n-1]=0.0;   // force linear extrapolation

        } else {
            assert(false);
        }

        m_c0s[idy] = (m_left==first_deriv) ? 0.0 : m_cs[idy][0];
        
    }
}

// return the closest id so that m_x[idx(idy)][id] <= x (return 0 if x too small)
size_t spline::find_closest(size_t idy, double x) const
{
    const size_t idx = m_nxs == 1 ? 0 : idy; //idx = 0 always if common X
    const std::vector<double>& x_idx = m_xs.at(idx);
    std::vector<double>::const_iterator it;
    it=std::upper_bound(x_idx.begin(),x_idx.end(),x);       // *it > x
    size_t id = std::max( int(it-x_idx.begin())-1, 0);   // x_idx[id] <= x
    return id;
}

double spline::operator() (size_t idy, double x) const
{
    // polynomial evaluation using Horner's scheme
    // TODO: consider more numerically accurate algorithms, e.g.:
    //   - Clenshaw
    //   - Even-Odd method by A.C.R. Newbery
    //   - Compensated Horner Scheme

    const size_t idx = m_nxs == 1 ? 0 : idy; //idx = 0 always if common X
    const std::vector<double>& x_idx = m_xs.at(idx);
    const std::vector<double>& y_idy = m_ys.at(idy);

    const size_t n = x_idx.size();
    const size_t id = find_closest(idy, x);

    double h=x-x_idx[id];
    double interpol;
    if(x<x_idx[0]) {
        // extrapolation to the left
        interpol=(m_c0s[idy]*h + m_bs[idy][0])*h + y_idy[0];
    } else if(x>x_idx[n-1]) {
        // extrapolation to the right
        interpol=(m_cs[idy][n-1]*h + m_bs[idy][n-1])*h + y_idy[n-1];
    } else {
        // interpolation
        interpol=((m_ds[idy][id]*h + m_cs[idy][id])*h + m_bs[idy][id])*h + y_idy[id];
    }
    return interpol;
}


namespace internal
{

// band_matrix implementation
// -------------------------

band_matrix::band_matrix(int dim, int n_u, int n_l)
{
    resize(dim, n_u, n_l);
}
void band_matrix::resize(int dim, int n_u, int n_l)
{
    assert(dim>0);
    assert(n_u>=0);
    assert(n_l>=0);
    m_upper.resize(n_u+1);
    m_lower.resize(n_l+1);
    for(size_t i=0; i<m_upper.size(); i++) {
        m_upper[i].resize(dim);
    }
    for(size_t i=0; i<m_lower.size(); i++) {
        m_lower[i].resize(dim);
    }
}
int band_matrix::dim() const
{
    if(m_upper.size()>0) {
        return m_upper[0].size();
    } else {
        return 0;
    }
}


// defines the new operator (), so that we can access the elements
// by A(i,j), index going from i=0,...,dim()-1
double & band_matrix::operator () (int i, int j)
{
    int k=j-i;       // what band is the entry
    assert( (i>=0) && (i<dim()) && (j>=0) && (j<dim()) );
    assert( (-num_lower()<=k) && (k<=num_upper()) );
    // k=0 -> diagonal, k<0 lower left part, k>0 upper right part
    if(k>=0)    return m_upper[k][i];
    else        return m_lower[-k][i];
}
double band_matrix::operator () (int i, int j) const
{
    int k=j-i;       // what band is the entry
    assert( (i>=0) && (i<dim()) && (j>=0) && (j<dim()) );
    assert( (-num_lower()<=k) && (k<=num_upper()) );
    // k=0 -> diagonal, k<0 lower left part, k>0 upper right part
    if(k>=0)    return m_upper[k][i];
    else        return m_lower[-k][i];
}
// second diag (used in LU decomposition), saved in m_lower
double band_matrix::saved_diag(int i) const
{
    assert( (i>=0) && (i<dim()) );
    return m_lower[0][i];
}
double & band_matrix::saved_diag(int i)
{
    assert( (i>=0) && (i<dim()) );
    return m_lower[0][i];
}

// LR-Decomposition of a band matrix
void band_matrix::lu_decompose()
{
    int  i_max,j_max;
    int  j_min;
    double x;

    // preconditioning
    // normalize column i so that a_ii=1
    for(int i=0; i<this->dim(); i++) {
        assert(this->operator()(i,i)!=0.0);
        this->saved_diag(i)=1.0/this->operator()(i,i);
        j_min=std::max(0,i-this->num_lower());
        j_max=std::min(this->dim()-1,i+this->num_upper());
        for(int j=j_min; j<=j_max; j++) {
            this->operator()(i,j) *= this->saved_diag(i);
        }
        this->operator()(i,i)=1.0;          // prevents rounding errors
    }

    // Gauss LR-Decomposition
    for(int k=0; k<this->dim(); k++) {
        i_max=std::min(this->dim()-1,k+this->num_lower());  // num_lower not a mistake!
        for(int i=k+1; i<=i_max; i++) {
            assert(this->operator()(k,k)!=0.0);
            x=-this->operator()(i,k)/this->operator()(k,k);
            this->operator()(i,k)=-x;                         // assembly part of L
            j_max=std::min(this->dim()-1,k+this->num_upper());
            for(int j=k+1; j<=j_max; j++) {
                // assembly part of R
                this->operator()(i,j)=this->operator()(i,j)+x*this->operator()(k,j);
            }
        }
    }
}

// solves Ly=b
std::vector<double> band_matrix::l_solve(const std::vector<double>& b) const
{
    assert( this->dim()==(int)b.size() );
    std::vector<double> x(this->dim());
    int j_start;
    double sum;
    for(int i=0; i<this->dim(); i++) {
        sum=0;
        j_start=std::max(0,i-this->num_lower());
        for(int j=j_start; j<i; j++) sum += this->operator()(i,j)*x[j];
        x[i]=(b[i]*this->saved_diag(i)) - sum;
    }
    return x;
}
// solves Rx=y
std::vector<double> band_matrix::r_solve(const std::vector<double>& b) const
{
    assert( this->dim()==(int)b.size() );
    std::vector<double> x(this->dim());
    int j_stop;
    double sum;
    for(int i=this->dim()-1; i>=0; i--) {
        sum=0;
        j_stop=std::min(this->dim()-1,i+this->num_upper());
        for(int j=i+1; j<=j_stop; j++) sum += this->operator()(i,j)*x[j];
        x[i]=( b[i] - sum ) / this->operator()(i,i);
    }
    return x;
}

std::vector<double> band_matrix::lu_solve(const std::vector<double>& b,
        bool is_lu_decomposed)
{
    assert( this->dim()==(int)b.size() );
    std::vector<double>  x,y;
    if(is_lu_decomposed==false) {
        this->lu_decompose();
    }
    y=this->l_solve(b);
    x=this->r_solve(y);
    return x;
}

} // namespace internal


} // namespace tk


} // namespace