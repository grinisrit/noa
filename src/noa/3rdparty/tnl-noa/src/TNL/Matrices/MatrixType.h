// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace noa::TNL {
namespace Matrices {

/**
 * \brief Structure for specifying type of sparse matrix.
 *
 * It is used for specification of \ref SparseMatrix type.
 */
template< bool Symmetric >
struct MatrixType
{
   static constexpr bool
   isSymmetric()
   {
      return Symmetric;
   }

   static std::string
   getSerializationType()
   {
      if( isSymmetric() )
         return "Symmetric";
      return "General";
   }
};

/**
 * \brief General non-symmetric matrix type.
 *
 * It is used for specification of \ref SparseMatrix type.
 */
struct GeneralMatrix : MatrixType< false >
{};

/**
 * \brief Symmetric matrix type.
 *
 * Symmetric matrix stores only lower part of the matrix and its diagonal. The
 * upper part is reconstructed on the fly.
 * It is used for specification of \ref SparseMatrix type.
 */
struct SymmetricMatrix : MatrixType< true >
{};

}  // namespace Matrices
}  // namespace noa::TNL
