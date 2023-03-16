
// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/String.h>


struct VoidOperation {
   public:

      template<typename Entity>
      __cuda_callable__ inline
      static void exec(Entity& entity) {}
};


struct GetEntityIsBoundaryOperation {
   public:
      template<typename Entity>
      __cuda_callable__ inline
      static void exec(Entity& entity) {
         bool isBoundary = entity.isBoundary();

         if (isBoundary)
            isBoundary = false;
      }
};

struct GetEntityCoordinateOperation {
   public:
      template<typename Entity>
      __cuda_callable__ inline
      static void exec(Entity& entity) {
         typename Entity::GridType::CoordinatesType coordinate = entity.getCoordinates();

         coordinate.x() += 1;
      }
};

struct GetEntityIndexOperation {
   public:
      template<typename Entity>
      __cuda_callable__ inline
      static typename Entity::IndexType exec(Entity& entity) {
         return entity.getIndex();
      }
};

struct GetEntityNormalsOperation {
   public:
      template<typename Entity>
      __cuda_callable__ inline
      static void exec(Entity& entity) {
         typename Entity::GridType::CoordinatesType coordinate = entity.getNormals();

         coordinate.x() += 1;
      }
};

struct RefreshEntityOperation {
   public:
      template<typename Entity>
      __cuda_callable__ inline
      static void exec(Entity& entity) {
         entity.refresh();
      }
};

struct GetMeshDimensionOperation {
   public:
      template<typename Entity>
      __cuda_callable__ inline
      static void exec(Entity& entity) {
         typename Entity::GridType::CoordinatesType coordinate = entity.getMesh().getDimensions();

         coordinate.x() += 1;
      }
};

struct GetOriginOperation {
   public:
      template<typename Entity>
      __cuda_callable__ inline
      static void exec(Entity& entity) {
         typename Entity::GridType::PointType coordinate = entity.getMesh().getOrigin();

         coordinate.x() += 1;
      }
};

struct GetEntitiesCountsOperation {
   public:
      template<typename Entity>
      __cuda_callable__ inline
      static void exec(Entity& entity) {
         typename Entity::GridType::EntitiesCounts coordinate = entity.getMesh().getEntitiesCounts();

         coordinate.x() += 1;
      }
};
