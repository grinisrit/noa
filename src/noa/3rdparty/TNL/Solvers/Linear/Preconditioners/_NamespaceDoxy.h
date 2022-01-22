// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noaTNL {
   namespace Solvers {
      namespace Linear {
         /**
          * \brief Namespace for preconditioners of linear system solvers.
          *
          * This namespace contains the following preconditioners for iterative solvers linear systems.
          *
          * 1. Diagonal - is diagonal or Jacobi preconditioner - see[Netlib](http://netlib.org/linalg/html_templates/node55.html)
          * 2. ILU0 - is Incomplete LU preconditioner with the same sparsity pattern as the original matrix - see [Wikipedia](https://en.wikipedia.org/wiki/Incomplete_LU_factorization)
          * 3. ILUT - is Incomplete LU preconiditoner with thresholding - see [paper by Y. Saad](https://www-users.cse.umn.edu/~saad/PDF/umsi-92-38.pdf)
          */
         namespace Preconditioners {
         } // namespace Preconditioners
      } // namespace Linear
   } // namespace Solvers
} // namespace noaTNL
