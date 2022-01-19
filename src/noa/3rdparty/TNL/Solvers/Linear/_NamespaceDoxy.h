// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL {
   namespace Solvers {

      /**
       * \brief Namespace for linear system solvers.
       *
       * This namespace contains the following algorithms and methods for solution of linear systems.
       *
       * # Direct methods
       *
       * # Iterative methods
       *
       * ## Stationary methods
       *    1. Jacobi method - \ref TNL::Solvers::Linear::Jacobi
       *    2. Successive-overrelaxation method, SOR - \ref TNL::Solvers::Linear::SOR
       *
       * ## Krylov subspace methods
       *    1. Conjugate gradient method, CG - \ref TNL::Solvers::Linear::CG
       *    2. Biconjugate gradient stabilized method, BICGStab  - \ref TNL::Solvers::Linear::BICGStab
       *    3. BICGStab(l) method  - \ref TNL::Solvers::Linear::BICGStabL
       *    4. Transpose-free quasi-minimal residual method, TFQMR - \ref TNL::Solvers::Linear::TFQMR
       *    5. Generalized minimal residual method, GMERS - \ref TNL::Solvers::Linear::GMRES with various methods of orthogonalization
       *        1. [Classical Gramm-Schmidt, CGS](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process)
       *        2. Classical Gramm-Schmidt with reorthogonalization, CGSR
       *        3. Modified Gramm-Schmidt, MGS
       *        4. Modified Gramm-Schmidt with reorthogonalization, MGSR
       *        5. Compact WY form of the Householder reflections, CWY
       *
       */
      namespace Linear {
      } // namespace Linear
   } // namespace Solvers
} // namespace TNL
