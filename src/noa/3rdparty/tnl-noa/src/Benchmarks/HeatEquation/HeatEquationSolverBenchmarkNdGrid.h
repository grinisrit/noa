// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once


#include <iostream>
#include <fstream>
#include <numeric>
#include <type_traits>
#include <array>
#include <bitset>

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Devices/Host.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Algorithms/ParallelFor.h>

#include "HeatEquationSolverBenchmark.h"

template<bool ...> struct bool_pack {};

template <bool... Bs>
using conjunction = std::is_same<bool_pack<true, Bs...>, bool_pack<Bs..., true>>;

template<int Dimension,
         typename Index,
         typename = std::enable_if_t<(Dimension > 0)>,
         typename = std::enable_if_t<std::is_integral<Index>::value>>
using Container = TNL::Containers::StaticArray<Dimension, Index>;

template<int Dimension,
         typename Index,
         typename = std::enable_if_t<(Dimension > 0)>,
         typename = std::enable_if_t<std::is_integral<Index>::value>>
class GridEntity {
   public:
      __cuda_callable__
      inline explicit GridEntity(const Index& vertexIndex,
                                 const Container<Dimension, Index>& coordinates,
                                 const Container<Dimension, bool>& direction) :
                                 vertexIndex(vertexIndex),
                                 coordinates(coordinates),
                                 direction(direction) {}

      __cuda_callable__
      ~GridEntity() {}

      __cuda_callable__
      inline Container<Dimension, Index> getCoordinates() const noexcept {
         return coordinates;
      }

      __cuda_callable__
      inline Index getIndex() const noexcept {
         return vertexIndex;
      }
   private:
      const Index vertexIndex;
      const Container<Dimension, Index> coordinates;
      const Container<Dimension, bool> direction;
};

template<int Dimension,
         typename Index,
         typename Device = TNL::Devices::Host,
         typename = std::enable_if_t<(Dimension > 0)>>
class NdGrid {
   public:
      NdGrid() {}
      ~NdGrid() {}

      /**
       *  @brief - Specifies dimensions of the grid
       *  @param[in] dimensions - A parameter pack, which specifies points count in the specific dimension.
       *                          Most significant dimension is in the beginning of the list.
       *                          Least significant dimension is in the end of the list
       */
      template <typename... Dimensions,
                typename = std::enable_if_t<conjunction<std::is_same<Index, Dimensions>::value...>::value>,
                typename = std::enable_if_t<sizeof...(Dimensions) == Dimension>>
      void setDimensions(Dimensions... dimensions) noexcept {
         Index i = 0;

         for (auto x: { dimensions... }) {
            TNL_ASSERT_GT(x, 0, "Dimension must be positive");
            this -> dimensions[i] = x;
            i++;
         }

         refreshDimensionMaps();
      }
      /**
       * @param[in] index - index of dimension
       */
      inline __cuda_callable__ Index getDimension(Index index) const noexcept {
         TNL_ASSERT_GE(index, 0, "Index must be greater than zero");
         TNL_ASSERT_LT(index, Dimension, "Index must be less than Dimension");

         return dimensions[index];
      }
      /**
       * @param[in] indices - A dimension index pack
       */
      template <typename... DimensionIndex,
                typename = std::enable_if_t<conjunction<std::is_same<Index, DimensionIndex>::value...>::value>,
                typename = std::enable_if_t<(sizeof...(DimensionIndex) > 0)>>
      Container<sizeof...(DimensionIndex), Index> getDimensions(DimensionIndex... indices) const noexcept {
         Container<sizeof...(DimensionIndex), Index> result{indices...};

         for (std::size_t i = 0; i < sizeof...(DimensionIndex); i++)
            result[i] = this -> getDimension(result[i]);

         return result;
      }
      /**
       * @param[in] index - index of dimension
       */
      inline __cuda_callable__ Index getEntitiesCount(Index index) const noexcept {
         TNL_ASSERT_GE(index, 0, "Index must be greater than zero");
         TNL_ASSERT_LE(index, Dimension, "Index must be less than or equal to Dimension");

         return cumulativeDimensionMap(index);
      }
      /**
       * @brief - Returns the number of entities of specific dimension
       */
      template <typename... DimensionIndex,
                typename = std::enable_if_t<conjunction<std::is_same<Index, DimensionIndex>::value...>::value>,
                typename = std::enable_if_t<(sizeof...(DimensionIndex) > 0)>>
      Container<sizeof...(DimensionIndex), Index> getEntitiesCounts(DimensionIndex... indices) const noexcept {
         Container<sizeof...(DimensionIndex), Index> result{indices...};

         for (std::size_t i = 0; i < sizeof...(DimensionIndex); i++)
            result[i] = this -> getEntitiesCount(result[i]);

         return result;
      }
      /**
       * @param[in] index - index of dimension
       */
      inline __cuda_callable__ Index getEndIndex(Index index) const noexcept {
         TNL_ASSERT_GE(index, 0, "Index must be greater than zero");
         TNL_ASSERT_LT(index, Dimension, "Index must be less than or equal to Dimension");

         return this -> getDimension(index) - 1;
      }
       /**
       * @brief - Returns the last index of specific dimensions
       */
      template <typename... DimensionIndex,
         typename = std::enable_if_t<conjunction<std::is_same<Index, DimensionIndex>::value...>::value>,
         typename = std::enable_if_t<(sizeof...(DimensionIndex) > 0)>>
         Container<sizeof...(DimensionIndex), Index> getEndIndices(DimensionIndex... indices) const noexcept {
         Container<sizeof...(DimensionIndex), Index> result{ indices... };

         for (std::size_t i = 0; i < sizeof...(DimensionIndex); i++)
            result[i] = this->getEndIndex(result[i]);

         return result;
      }
      /**
       * @brief - Traversers all elements in the grid
       */
      template <typename Function, typename... FunctionArgs>
      void traverseAll(Function function, FunctionArgs... args) const noexcept {
         auto lambda = [=] __cuda_callable__(const Index index, FunctionArgs... args) mutable {
            function(index, args...);
         };

         TNL::Algorithms::ParallelFor<Device>::exec(0, cumulativeDimensionMap[0], lambda, args...);
      }
      /**
       * @brief - Traverses a grid from start index to end index.
       *
       * @param[in] start - a start index of point
       * @param[in] end - an end index of point
       * @param[in] directions - A pack of boolean vector flags with the size of the dimension.
       *   For example, let's have the 3-dimensional grid.
       *     A pack {false, false, false} will call function for all points
       *     A pack {true, false, false} will call function for edges directed over dimension at index of true
       *     A pack {true, true, false} will call function for faces directed over dimension at index of true
       *     A pack {true, true, true} will call function for cells directed over
       *
       */
      template<typename Function, typename... FunctionArgs>
      void traverse(const Container<Dimension, Index>& firstVertex,
                    const Container<Dimension, Index>& secondVertex,
                    const TNL::Containers::Array<Container<Dimension, bool>, Device>& directions,
                    Function function,
                    FunctionArgs... args) const noexcept {
         Index verticesCount = 1;
         Container<Dimension, Index> traverseRectOrigin, traverseRectDimensions,
                                     dimensionsProducts, traverseRectDimensionsProducts;

         for (Index i = 0; i < Dimension; i++) {
            traverseRectDimensions[i] = abs(secondVertex[i] - firstVertex[i]) + 1;
            verticesCount *= traverseRectDimensions[i];
            traverseRectOrigin[i] = std::min(firstVertex[i], secondVertex[i]);

            TNL_ASSERT_LT(firstVertex[i], dimensions[i], "End index must be in dimensions range");
            TNL_ASSERT_LT(secondVertex[i], dimensions[i], "Start index must be in dimensions range");
         }

         dimensionsProducts = getDimensionProducts(dimensions);
         traverseRectDimensionsProducts = getDimensionProducts(traverseRectDimensions);

         auto& grid = *this;
         auto outerFunction = [=] __cuda_callable__(Index offset,
                                                    const Container<Dimension, Index>& traverseRectOrigin,
                                                    const Container<Dimension, Index>& traverseRectDimensions,
                                                    const Container<Dimension, Index>& traverseRectDimensionsProducts,
                                                    const Container<Dimension, Index>& dimensionsProducts,
                                                    FunctionArgs... args) mutable {
            auto entity = grid.makeEntitity(offset,
                                            traverseRectOrigin, traverseRectDimensions,
                                            traverseRectDimensionsProducts, dimensionsProducts);

            function(entity, args...);
         };

         Index lowerBound = 0, upperBound = verticesCount;

         for (Index i = 0; i < directions.getSize(); i++) {
            TNL::Algorithms::ParallelFor<Device>::exec(lowerBound, upperBound, outerFunction,
                                                       traverseRectOrigin,
                                                       traverseRectDimensions,
                                                       traverseRectDimensionsProducts,
                                                       dimensionsProducts,
                                                       args...);
         }
      }
   private:
      Container<Dimension, Index> dimensions;
      /**
       * @brief - A dimension map is a store for dimension limits over all combinations of normals.
       *          First, (n choose 0) elements will contain the count of 0 dimension elements
       *          Second, (n choose 1) elements will contain the count of 1-dimension elements
       *          ....
       *
       *          For example, let's have a 3-d grid, then the map indexing will be the next:
       *            0 - 0 - count of vertices
       *            1, 2, 3 - count of edges in x, y, z plane
       *            4, 5, 6 - count of faces in xy, yz, zy plane
       *            7 - count of cells in z y x plane
       *
       * @warning - The ordering of is lexigraphical.
       */
      Container<1 << Dimension, Index> dimensionMap;
      /**
       * @brief - A cumulative map over dimensions.
       */
      Container<Dimension + 1, Index> cumulativeDimensionMap;
      /**
       * @brief - Fills dimensions map for N-dimensional Grid.
       *
       * @complexity - O(2 ^ Dimension)
       */
      void refreshDimensionMaps() noexcept {
         std::array<bool, Dimension> combinationBuffer = {};
         std::size_t j = 0;

         for (std::size_t i = 0; i < Dimension + 1; i++)
            cumulativeDimensionMap[i] = 0;

         for (std::size_t i = 0; i <= Dimension; i++) {
            std::fill(combinationBuffer.begin(), combinationBuffer.end(), false);
            std::fill(combinationBuffer.end() - i, combinationBuffer.end(), true);

            do {
               int result = 1;

               for (std::size_t k = 0; k < combinationBuffer.size(); k++)
                  result *= combinationBuffer[k] ? dimensions[Dimension - k - 1] - 1 : dimensions[Dimension - k - 1];

               dimensionMap[j] = result;
               cumulativeDimensionMap[i] += result;

               j++;
            } while (std::next_permutation(combinationBuffer.begin(), combinationBuffer.end()));
         }
      }

      __cuda_callable__ inline
      GridEntity<Dimension, Index> makeEntitity(const Index& index,
                                                const Container<Dimension, Index>& traverseRectOrigin,
                                                const Container<Dimension, Index>& traverseRectDimensions,
                                                const Container<Dimension, Index>& traverseRectDimensionsProducts,
                                                const Container<Dimension, Index>& dimensionsProducts) const {
         //Container<Dimension, Index> traverseCoordinates = 0;
         Container<Dimension, Index> traverseCoordinates = getCoordinates(index, traverseRectDimensions);
         Container<Dimension, Index> globalCoordinates = 0;

         for (Index i = 0; i < Dimension; i++)
            globalCoordinates[i] = traverseRectOrigin[i] + traverseCoordinates[i];

         const auto globalIndex = getIndex(globalCoordinates, dimensionsProducts);

         return GridEntity<Dimension, Index>(globalIndex, globalCoordinates, {});
      }
      /**
       * Calculates position in the specific boundaries
       */
      __cuda_callable__ inline
      Container<Dimension, Index> getCoordinates(const Index& index,
                                                 const Container<Dimension, Index> &dimensions) const {
         Container<Dimension, Index> coordinates = 0;
         Index tmpIndex = index;

         Index dimensionIndex = 0;

         while (tmpIndex && dimensionIndex < Dimension) {
            Index dimension = dimensions[dimensionIndex],
                  quotient = tmpIndex / dimension,
                  reminder = tmpIndex - (dimension * quotient);

            coordinates[dimensionIndex] = reminder;
            tmpIndex = quotient;

            dimensionIndex += 1;
         }

         return coordinates;
      }
      /**
       * Calculates product matrix based on the dimension
       */
      __cuda_callable__ inline
      Container<Dimension, Index> getDimensionProducts(const Container<Dimension, Index>& dimensions) const noexcept {
         Container<Dimension, Index> products = 0;

         products[0] = 1;

         for (Index i = 1; i < Dimension; i++)
            products[i] = dimensions[i - 1] * products[i - 1];

         return products;
      }
      /**
       * Calculates index based on the dimension
       */
      __cuda_callable__ inline
      Index getIndex(const Container<Dimension, Index>& coordinates,
                     const Container<Dimension, Index>& dimensionProducts) const {
         Index index = 0;

         for (Index i = 0; i < Dimension; i++)
            index += coordinates[i] * dimensionProducts[i];

         return index;
      }

};

template< typename Real = double,
          typename Device = TNL::Devices::Host,
          typename Index = int >
struct HeatEquationSolverBenchmarkNdGrid : public HeatEquationSolverBenchmark< Real, Device, Index >
{
   void exec( const Index xSize, const Index ySize )
   {
      NdGrid<2, int, Device> grid;

      grid.setDimensions( xSize, ySize );

      const Real hx = this->xDomainSize / (Real) grid.getDimension(0);
      const Real hy = this->yDomainSize / (Real) grid.getDimension(1);
      const Real hx_inv = 1.0 / (hx * hx);
      const Real hy_inv = 1.0 / (hy * hy);

      const Container<2, bool> direction{ false, false };

      Real start = 0;
      Index iterations = 0;
      auto timestep = this->timeStep ? this->timeStep : 0.1 * std::min(hx * hx, hy * hy);
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) )
      {
         auto uxView = this->aux.getView();
         auto auxView = this->ux.getView();
         auto xDimension = grid.getDimension(0);

         auto next = [=] __cuda_callable__(const GridEntity<2, int>& entity) mutable {
            auto index = entity.getIndex();
            auto element = uxView[index];
            auto center = 2 * element;

            auxView[index] = element + ((uxView[index - 1] - center + uxView[index + 1]) * hx_inv +
                                       (uxView[index - xDimension] - center + uxView[index + xDimension]) * hy_inv) * timestep;
         };
         grid.traverse( { 1, 1 },
                        { grid.getEndIndex(0) - 1, grid.getEndIndex(1) - 1 },
                        { direction },
                        next );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }
};
