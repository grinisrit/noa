#pragma once

#include <TNL/Containers/StaticVector.h>
#include <TNL/Operators/Analytic/Sign.h>
#include <TNL/Functions/MeshFunctionEvaluator.h>
#include <TNL/Operators/Analytic/Sign.h>
#include <TNL/Meshes/Grid.h>
#include "CompressibleConservativeVariables.h"

namespace TNL {
template <typename Mesh>
class RiemannProblemInitialConditionSetter
{

};

template <typename MeshReal,
          typename Device,
          typename MeshIndex>
class RiemannProblemInitialConditionSetter< Meshes::Grid< 1,MeshReal, Device, MeshIndex > >
{
   public:

      typedef Meshes::Grid< 1,MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::RealType RealType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;
      static const int Dimensions = MeshType::getMeshDimension();
      typedef Containers::StaticVector< Dimensions, RealType > PointType;
      typedef Functions::MeshFunctionView< MeshType > MeshFunctionType;
      typedef Pointers::SharedPointer< MeshFunctionType > MeshFunctionPointer;
      typedef Functions::VectorField< Dimensions, MeshType > VectorFieldType;
//       for cyklus i = 0 to mesh.getDimensions().x() j pro .y() a k pro .z()
//       typedef typename MeshType::Cell CellType
//       typedef typename MeshType::CoordinatesType CoordinatesType
//       Celltype cell(mesh, CoordinatesType(i,j))
//       p59stup do density setElement(mesh.template getEntityIndex< CellType >(cell), hodnota, kterou budu zapisovat)
//       pomocn8 t59da, kterou budu specialiyovat p5es r;zn0 dimenze gridu

      void setDiscontinuity(PointType discontinuityPlacement)
      {
         this->discontinuityPlacement = discontinuityPlacement;
      };
      void setDensity(RealType NWUDensity,
                      RealType NEUDensity,
                      RealType SWUDensity,
                      RealType SEUDensity,
                      RealType NWDDensity,
                      RealType NEDDensity,
                      RealType SWDDensity,
                      RealType SEDDensity)
      {
         this->NWUDensity = NWUDensity;
         this->NEUDensity = NEUDensity;
         this->SWUDensity = SWUDensity;
         this->SEUDensity = SEUDensity;
         this->NWDDensity = NWDDensity;
         this->NEDDensity = NEDDensity;
         this->SWDDensity = SWDDensity;
         this->SEDDensity = SEDDensity;
      };

      void setMomentum(PointType NWUMomentum,
                       PointType NEUMomentum,
                       PointType SWUMomentum,
                       PointType SEUMomentum,
                       PointType NWDMomentum,
                       PointType NEDMomentum,
                       PointType SWDMomentum,
                       PointType SEDMomentum)
      {
         this->NWUMomentum = NWUMomentum;
         this->NEUMomentum = NEUMomentum;
         this->SWUMomentum = SWUMomentum;
         this->SEUMomentum = SEUMomentum;
         this->NWDMomentum = NWDMomentum;
         this->NEDMomentum = NEDMomentum;
         this->SWDMomentum = SWDMomentum;
         this->SEDMomentum = SEDMomentum;
      };

      void setEnergy(RealType NWUEnergy,
                     RealType NEUEnergy,
                     RealType SWUEnergy,
                     RealType SEUEnergy,
                     RealType NWDEnergy,
                     RealType NEDEnergy,
                     RealType SWDEnergy,
                     RealType SEDEnergy)
      {
         this->NWUEnergy = NWUEnergy;
         this->NEUEnergy = NEUEnergy;
         this->SWUEnergy = SWUEnergy;
         this->SEUEnergy = SEUEnergy;
         this->NWDEnergy = NWDEnergy;
         this->NEDEnergy = NEDEnergy;
         this->SWDEnergy = SWDEnergy;
         this->SEDEnergy = SEDEnergy;
      };

      void setGamma(RealType gamma)
      {
         this->gamma = gamma;
      };

      void placeDensity(CompressibleConservativeVariables< MeshType >& conservativeVariables)
      {
      typedef typename MeshType::Cell CellType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      MeshType mesh = (* conservativeVariables.getDensity()).getMesh();
         for( int i = 0; i < mesh.getDimensions().x(); i++)
            if ( i < this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
               {
                  CellType cell(mesh, CoordinatesType(i));
                  cell.refresh();
                  (* conservativeVariables.getDensity()).setValue(cell, this->SWDDensity);
               }
            else
               {
                  CellType cell(mesh, CoordinatesType(i));
                  cell.refresh();
                  (* conservativeVariables.getDensity()).setValue(cell, this->SEDDensity);
               }
      };

      void placeMomentum(CompressibleConservativeVariables< MeshType >& conservativeVariables)
      {
      typedef typename MeshType::Cell CellType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      MeshType mesh = (* conservativeVariables.getDensity()).getMesh();
         for( int i = 0; i < mesh.getDimensions().x(); i++)
            if ( i < this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
               {
                  CellType cell(mesh, CoordinatesType(i));
                  cell.refresh();
                  (* (* conservativeVariables.getMomentum())[ 0 ]).setValue(cell, this->SWDMomentum[ 0 ]);
               }
            else
               {
                  CellType cell(mesh, CoordinatesType(i));
                  cell.refresh();
                  (* (* conservativeVariables.getMomentum())[ 0 ]).setValue(cell, this->SEDMomentum[ 0 ]);
               }
      };

      void placeEnergy(CompressibleConservativeVariables< MeshType >& conservativeVariables)
      {
      typedef typename MeshType::Cell CellType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      MeshType mesh = (* conservativeVariables.getDensity()).getMesh();
         for( int i = 0; i < mesh.getDimensions().x(); i++)
            if ( i < this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
               {
                  CellType cell(mesh, CoordinatesType(i));
                  cell.refresh();
                  (* conservativeVariables.getEnergy()).setValue(cell, this->SWDEnergy);
               }
            else
               {
                  CellType cell(mesh, CoordinatesType(i));
                  cell.refresh();
                  (* conservativeVariables.getEnergy()).setValue(cell, this->SEDEnergy);
               }
      };

      PointType discontinuityPlacement;
      RealType  NWUDensity, NEUDensity, SWUDensity, SEUDensity, NWDDensity, NEDDensity, SWDDensity, SEDDensity;
      RealType  NWUEnergy, NEUEnergy, SWUEnergy, SEUEnergy, NWDEnergy, NEDEnergy, SWDEnergy, SEDEnergy;
      PointType NWUMomentum, NEUMomentum, SWUMomentum, SEUMomentum, NWDMomentum, NEDMomentum, SWDMomentum, SEDMomentum;
      RealType gamma;
};


template <typename MeshReal,
          typename Device,
          typename MeshIndex>
class RiemannProblemInitialConditionSetter< Meshes::Grid< 2, MeshReal, Device, MeshIndex > >
{
   public:

      typedef Meshes::Grid< 2,MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::RealType RealType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;
      static const int Dimensions = MeshType::getMeshDimension();
      typedef Containers::StaticVector< Dimensions, RealType > PointType;
      typedef Functions::MeshFunctionView< MeshType > MeshFunctionType;
      typedef Pointers::SharedPointer< MeshFunctionType > MeshFunctionPointer;
      typedef Functions::VectorField< Dimensions, MeshType > VectorFieldType;
//       for cyklus i = 0 to mesh.getDimensions().x() j pro .y() a k pro .z()
//       typedef typename MeshType::Cell CellType
//       typedef typename MeshType::CoordinatesType CoordinatesType
//       Celltype cell(mesh, CoordinatesType(i,j))
//       p59stup do density setElement(mesh.template getEntityIndex< CellType >(cell), hodnota, kterou budu zapisovat)
//       pomocn8 t59da, kterou budu specialiyovat p5es r;zn0 dimenze gridu

      void setDiscontinuity(PointType discontinuityPlacement)
      {
         this->discontinuityPlacement = discontinuityPlacement;
      };
      void setDensity(RealType NWUDensity,
                      RealType NEUDensity,
                      RealType SWUDensity,
                      RealType SEUDensity,
                      RealType NWDDensity,
                      RealType NEDDensity,
                      RealType SWDDensity,
                      RealType SEDDensity)
      {
         this->NWUDensity = NWUDensity;
         this->NEUDensity = NEUDensity;
         this->SWUDensity = SWUDensity;
         this->SEUDensity = SEUDensity;
         this->NWDDensity = NWDDensity;
         this->NEDDensity = NEDDensity;
         this->SWDDensity = SWDDensity;
         this->SEDDensity = SEDDensity;
      };

      void setMomentum(PointType NWUMomentum,
                       PointType NEUMomentum,
                       PointType SWUMomentum,
                       PointType SEUMomentum,
                       PointType NWDMomentum,
                       PointType NEDMomentum,
                       PointType SWDMomentum,
                       PointType SEDMomentum)
      {
         this->NWUMomentum = NWUMomentum;
         this->NEUMomentum = NEUMomentum;
         this->SWUMomentum = SWUMomentum;
         this->SEUMomentum = SEUMomentum;
         this->NWDMomentum = NWDMomentum;
         this->NEDMomentum = NEDMomentum;
         this->SWDMomentum = SWDMomentum;
         this->SEDMomentum = SEDMomentum;
      };

      void setEnergy(RealType NWUEnergy,
                     RealType NEUEnergy,
                     RealType SWUEnergy,
                     RealType SEUEnergy,
                     RealType NWDEnergy,
                     RealType NEDEnergy,
                     RealType SWDEnergy,
                     RealType SEDEnergy)
      {
         this->NWUEnergy = NWUEnergy;
         this->NEUEnergy = NEUEnergy;
         this->SWUEnergy = SWUEnergy;
         this->SEUEnergy = SEUEnergy;
         this->NWDEnergy = NWDEnergy;
         this->NEDEnergy = NEDEnergy;
         this->SWDEnergy = SWDEnergy;
         this->SEDEnergy = SEDEnergy;
      };

      void setGamma(RealType gamma)
      {
         this->gamma = gamma;
      };

      void placeDensity(CompressibleConservativeVariables< MeshType >& conservativeVariables)
      {
      typedef typename MeshType::Cell CellType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      MeshType mesh = (* conservativeVariables.getDensity()).getMesh();
         for( int i = 0; i < mesh.getDimensions().x(); i++)
            for( int j = 0; j < mesh.getDimensions().y(); j++)
               if ( ( i <= this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                 && ( j <= this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() ) )
                  {
                     CellType cell(mesh, CoordinatesType(i,j));
                     cell.refresh();
                     (* conservativeVariables.getDensity()).setValue(cell, this->SWDDensity);
                  }
               else
               if ( ( i > this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                 && ( j <= this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() ) )
                  {
                     CellType cell(mesh, CoordinatesType(i,j));
                     cell.refresh();
                     (* conservativeVariables.getDensity()).setValue(cell, this->SEDDensity);
                  }
               else
               if ( ( i <= this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                 && ( j > this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() ) )
                  {
                     CellType cell(mesh, CoordinatesType(i,j));
                     cell.refresh();
                     (* conservativeVariables.getDensity()).setValue(cell, this->NWDDensity);
                  }
               else
               if ( ( i > this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                 && ( j > this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() ) )
                  {
                     CellType cell(mesh, CoordinatesType(i,j));
                     cell.refresh();
                     (* conservativeVariables.getDensity()).setValue(cell, this->NEDDensity);
                  }
      };

      void placeMomentum(CompressibleConservativeVariables< MeshType >& conservativeVariables)
      {
      typedef typename MeshType::Cell CellType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      MeshType mesh = (* conservativeVariables.getDensity()).getMesh();
         for( int i = 0; i < mesh.getDimensions().x(); i++)
            for( int j = 0; j < mesh.getDimensions().y(); j++)
               if ( ( i <= this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                 && ( j <= this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() ) )
                  {
                     CellType cell(mesh, CoordinatesType(i,j));
                     cell.refresh();
                     (* (* conservativeVariables.getMomentum())[ 0 ]).setValue(cell, this->SWDMomentum[ 0 ]);
                     (* (* conservativeVariables.getMomentum())[ 1 ]).setValue(cell, this->SWDMomentum[ 1 ]);
                  }
               else
               if ( ( i > this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                 && ( j <= this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() ) )
                  {
                     CellType cell(mesh, CoordinatesType(i,j));
                     cell.refresh();
                     (* (* conservativeVariables.getMomentum())[ 0 ]).setValue(cell, this->SEDMomentum[ 0 ]);
                     (* (* conservativeVariables.getMomentum())[ 1 ]).setValue(cell, this->SEDMomentum[ 1 ]);
                  }
               else
               if ( ( i <= this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                 && ( j > this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() ) )
                  {
                     CellType cell(mesh, CoordinatesType(i,j));
                     cell.refresh();
                     (* (* conservativeVariables.getMomentum())[ 0 ]).setValue(cell, this->NWDMomentum[ 0 ]);
                     (* (* conservativeVariables.getMomentum())[ 1 ]).setValue(cell, this->NWDMomentum[ 1 ]);
                  }
               else
               if ( ( i > this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                 && ( j > this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() ) )
                  {
                     CellType cell(mesh, CoordinatesType(i,j));
                     cell.refresh();
                     (* (* conservativeVariables.getMomentum())[ 0 ]).setValue(cell, this->NEDMomentum[ 0 ]);
                     (* (* conservativeVariables.getMomentum())[ 1 ]).setValue(cell, this->NEDMomentum[ 1 ]);
                  }
      };

      void placeEnergy(CompressibleConservativeVariables< MeshType >& conservativeVariables)
      {
      typedef typename MeshType::Cell CellType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      MeshType mesh = (* conservativeVariables.getDensity()).getMesh();
         for( int i = 0; i < mesh.getDimensions().x(); i++)
            for( int j = 0; j < mesh.getDimensions().y(); j++)
               if ( ( i <= this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                 && ( j <= this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() ) )
                  {
                     CellType cell(mesh, CoordinatesType(i,j));
                     cell.refresh();
                     (* conservativeVariables.getEnergy()).setValue(cell, this->SWDEnergy);
                  }
               else
               if ( ( i > this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                 && ( j <= this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() ) )
                  {
                     CellType cell(mesh, CoordinatesType(i,j));
                     cell.refresh();
                     (* conservativeVariables.getEnergy()).setValue(cell, this->SEDEnergy);
                  }
               else
               if ( ( i <= this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                 && ( j > this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() ) )
                  {
                     CellType cell(mesh, CoordinatesType(i,j));
                     cell.refresh();
                     (* conservativeVariables.getEnergy()).setValue(cell, this->NWDEnergy);
                  }
               else
               if ( ( i > this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                 && ( j > this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() ) )
                  {
                     CellType cell(mesh, CoordinatesType(i,j));
                     cell.refresh();
                     (* conservativeVariables.getEnergy()).setValue(cell, this->NEDEnergy);
                  }
      };

      PointType discontinuityPlacement;
      RealType  NWUDensity, NEUDensity, SWUDensity, SEUDensity, NWDDensity, NEDDensity, SWDDensity, SEDDensity;
      RealType  NWUEnergy, NEUEnergy, SWUEnergy, SEUEnergy, NWDEnergy, NEDEnergy, SWDEnergy, SEDEnergy;
      PointType NWUMomentum, NEUMomentum, SWUMomentum, SEUMomentum, NWDMomentum, NEDMomentum, SWDMomentum, SEDMomentum;
      RealType gamma;
};

template <typename MeshReal,
          typename Device,
          typename MeshIndex>
class RiemannProblemInitialConditionSetter< Meshes::Grid< 3, MeshReal, Device, MeshIndex > >
{
   public:

      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::RealType RealType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;
      static const int Dimensions = MeshType::getMeshDimension();
      typedef Containers::StaticVector< Dimensions, RealType > PointType;
      typedef Functions::MeshFunctionView< MeshType > MeshFunctionType;
      typedef Pointers::SharedPointer< MeshFunctionType > MeshFunctionPointer;
      typedef Functions::VectorField< Dimensions, MeshType > VectorFieldType;
//       for cyklus i = 0 to mesh.getDimensions().x() j pro .y() a k pro .z()
//       typedef typename MeshType::Cell CellType
//       typedef typename MeshType::CoordinatesType CoordinatesType
//       Celltype cell(mesh, CoordinatesType(i,j))
//       p59stup do density setElement(mesh.template getEntityIndex< CellType >(cell), hodnota, kterou budu zapisovat)
//       pomocn8 t59da, kterou budu specialiyovat p5es r;zn0 dimenze gridu

      void setDiscontinuity(PointType discontinuityPlacement)
      {
         this->discontinuityPlacement = discontinuityPlacement;
      };
      void setDensity(RealType NWUDensity,
                      RealType NEUDensity,
                      RealType SWUDensity,
                      RealType SEUDensity,
                      RealType NWDDensity,
                      RealType NEDDensity,
                      RealType SWDDensity,
                      RealType SEDDensity)
      {
         this->NWUDensity = NWUDensity;
         this->NEUDensity = NEUDensity;
         this->SWUDensity = SWUDensity;
         this->SEUDensity = SEUDensity;
         this->NWDDensity = NWDDensity;
         this->NEDDensity = NEDDensity;
         this->SWDDensity = SWDDensity;
         this->SEDDensity = SEDDensity;
      };

      void setMomentum(PointType NWUMomentum,
                       PointType NEUMomentum,
                       PointType SWUMomentum,
                       PointType SEUMomentum,
                       PointType NWDMomentum,
                       PointType NEDMomentum,
                       PointType SWDMomentum,
                       PointType SEDMomentum)
      {
         this->NWUMomentum = NWUMomentum;
         this->NEUMomentum = NEUMomentum;
         this->SWUMomentum = SWUMomentum;
         this->SEUMomentum = SEUMomentum;
         this->NWDMomentum = NWDMomentum;
         this->NEDMomentum = NEDMomentum;
         this->SWDMomentum = SWDMomentum;
         this->SEDMomentum = SEDMomentum;
      };

      void setEnergy(RealType NWUEnergy,
                     RealType NEUEnergy,
                     RealType SWUEnergy,
                     RealType SEUEnergy,
                     RealType NWDEnergy,
                     RealType NEDEnergy,
                     RealType SWDEnergy,
                     RealType SEDEnergy)
      {
         this->NWUEnergy = NWUEnergy;
         this->NEUEnergy = NEUEnergy;
         this->SWUEnergy = SWUEnergy;
         this->SEUEnergy = SEUEnergy;
         this->NWDEnergy = NWDEnergy;
         this->NEDEnergy = NEDEnergy;
         this->SWDEnergy = SWDEnergy;
         this->SEDEnergy = SEDEnergy;
      };

      void setGamma(RealType gamma)
      {
         this->gamma = gamma;
      };

      void placeDensity(CompressibleConservativeVariables< MeshType >& conservativeVariables)
      {
      typedef typename MeshType::Cell CellType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      MeshType mesh = (* conservativeVariables.getDensity()).getMesh();
         for( int i = 0; i < mesh.getDimensions().x(); i++)
            for( int j = 0; j < mesh.getDimensions().y(); j++)
               for ( int k = 0; k < mesh.getDimensions().z(); k++)
                  if ( ( i <= this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j <= this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k <= this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* conservativeVariables.getDensity()).setValue(cell, this->SWDDensity);
                     }
                  else
                  if ( ( i > this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j <= this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k <= this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* conservativeVariables.getDensity()).setValue(cell, this->SEDDensity);
                     }
                  else
                  if ( ( i <= this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j > this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k <= this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* conservativeVariables.getDensity()).setValue(cell, this->NWDDensity);
                     }
                  else
                  if ( ( i > this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j > this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k <= this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* conservativeVariables.getDensity()).setValue(cell, this->NEDDensity);
                     }
                  else
                  if ( ( i <= this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j <= this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k > this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* conservativeVariables.getDensity()).setValue(cell, this->SWUDensity);
                     }
                  else
                  if ( ( i > this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j <= this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k > this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* conservativeVariables.getDensity()).setValue(cell, this->SEUDensity);
                     }
                  else
                  if ( ( i <= this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j > this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k > this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* conservativeVariables.getDensity()).setValue(cell, this->SWUDensity);
                     }
                  else
                  if ( ( i > this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j > this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k > this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* conservativeVariables.getDensity()).setValue(cell, this->SEUDensity);
                     }
      };

      void placeMomentum(CompressibleConservativeVariables< MeshType >& conservativeVariables)
      {
      typedef typename MeshType::Cell CellType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      MeshType mesh = (* conservativeVariables.getDensity()).getMesh();
         for( int i = 0; i < mesh.getDimensions().x(); i++)
            for( int j = 0; j < mesh.getDimensions().y(); j++)
               for ( int k = 0; k < mesh.getDimensions().z(); k++)
                  if ( ( i <= this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j <= this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k <= this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* (* conservativeVariables.getMomentum())[ 0 ]).setValue(cell, this->SWDMomentum[ 0 ]);
                        (* (* conservativeVariables.getMomentum())[ 1 ]).setValue(cell, this->SWDMomentum[ 1 ]);
                        (* (* conservativeVariables.getMomentum())[ 2 ]).setValue(cell, this->SWDMomentum[ 2 ]);
                     }
                  else
                  if ( ( i > this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j <= this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k <= this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* (* conservativeVariables.getMomentum())[ 0 ]).setValue(cell, this->SEDMomentum[ 0 ]);
                        (* (* conservativeVariables.getMomentum())[ 1 ]).setValue(cell, this->SEDMomentum[ 1 ]);
                        (* (* conservativeVariables.getMomentum())[ 2 ]).setValue(cell, this->SEDMomentum[ 2 ]);
                     }
                  else
                  if ( ( i <= this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j > this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k <= this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* (* conservativeVariables.getMomentum())[ 0 ]).setValue(cell, this->NWDMomentum[ 0 ]);
                        (* (* conservativeVariables.getMomentum())[ 1 ]).setValue(cell, this->NWDMomentum[ 1 ]);
                        (* (* conservativeVariables.getMomentum())[ 2 ]).setValue(cell, this->NWDMomentum[ 2 ]);
                     }
                  else
                  if ( ( i > this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j > this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k <= this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* (* conservativeVariables.getMomentum())[ 0 ]).setValue(cell, this->NEDMomentum[ 0 ]);
                        (* (* conservativeVariables.getMomentum())[ 1 ]).setValue(cell, this->NEDMomentum[ 1 ]);
                        (* (* conservativeVariables.getMomentum())[ 2 ]).setValue(cell, this->NEDMomentum[ 2 ]);
                     }
                  else
                  if ( ( i <= this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j <= this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k > this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* (* conservativeVariables.getMomentum())[ 0 ]).setValue(cell, this->SWUMomentum[ 0 ]);
                        (* (* conservativeVariables.getMomentum())[ 1 ]).setValue(cell, this->SWUMomentum[ 1 ]);
                        (* (* conservativeVariables.getMomentum())[ 2 ]).setValue(cell, this->SWUMomentum[ 2 ]);
                     }
                  else
                  if ( ( i > this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j <= this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k > this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* (* conservativeVariables.getMomentum())[ 0 ]).setValue(cell, this->SEUMomentum[ 0 ]);
                        (* (* conservativeVariables.getMomentum())[ 1 ]).setValue(cell, this->SEUMomentum[ 1 ]);
                        (* (* conservativeVariables.getMomentum())[ 2 ]).setValue(cell, this->SEUMomentum[ 2 ]);
                     }
                  else
                  if ( ( i <= this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j > this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k > this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* (* conservativeVariables.getMomentum())[ 0 ]).setValue(cell, this->SWUMomentum[ 0 ]);
                        (* (* conservativeVariables.getMomentum())[ 1 ]).setValue(cell, this->SWUMomentum[ 1 ]);
                        (* (* conservativeVariables.getMomentum())[ 2 ]).setValue(cell, this->SWUMomentum[ 2 ]);
                     }
                  else
                  if ( ( i > this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j > this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k > this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* (* conservativeVariables.getMomentum())[ 0 ]).setValue(cell, this->SEUMomentum[ 0 ]);
                        (* (* conservativeVariables.getMomentum())[ 1 ]).setValue(cell, this->SEUMomentum[ 1 ]);
                        (* (* conservativeVariables.getMomentum())[ 2 ]).setValue(cell, this->SEUMomentum[ 2 ]);
                     }
      };

      void placeEnergy(CompressibleConservativeVariables< MeshType >& conservativeVariables)
      {
      typedef typename MeshType::Cell CellType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      MeshType mesh = (* conservativeVariables.getDensity()).getMesh();
         for( int i = 0; i < mesh.getDimensions().x(); i++)
            for( int j = 0; j < mesh.getDimensions().y(); j++)
               for ( int k = 0; k < mesh.getDimensions().z(); k++)
                  if ( ( i <= this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j <= this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k <= this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* conservativeVariables.getEnergy()).setValue(cell, this->SWDEnergy);
                     }
                  else
                  if ( ( i > this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j <= this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k <= this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* conservativeVariables.getEnergy()).setValue(cell, this->SEDEnergy);
                     }
                  else
                  if ( ( i <= this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j > this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k <= this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* conservativeVariables.getEnergy()).setValue(cell, this->NWDEnergy);
                     }
                  else
                  if ( ( i > this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j > this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k <= this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* conservativeVariables.getEnergy()).setValue(cell, this->NEDEnergy);
                     }
                  else
                  if ( ( i <= this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j <= this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k > this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* conservativeVariables.getEnergy()).setValue(cell, this->SWUEnergy);
                     }
                  else
                  if ( ( i > this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j <= this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k > this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* conservativeVariables.getEnergy()).setValue(cell, this->SEUEnergy);
                     }
                  else
                  if ( ( i <= this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j > this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k > this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* conservativeVariables.getEnergy()).setValue(cell, this->SWUEnergy);
                     }
                  else
                  if ( ( i > this->discontinuityPlacement[ 0 ] * mesh.getDimensions().x() )
                    && ( j > this->discontinuityPlacement[ 1 ] * mesh.getDimensions().y() )
                    && ( k > this->discontinuityPlacement[ 2 ] * mesh.getDimensions().z() ) )
                     {
                        CellType cell(mesh, CoordinatesType(i,j,k));
                        cell.refresh();
                        (* conservativeVariables.getEnergy()).setValue(cell, this->SEUEnergy);
                     }
      };

      PointType discontinuityPlacement;
      RealType  NWUDensity, NEUDensity, SWUDensity, SEUDensity, NWDDensity, NEDDensity, SWDDensity, SEDDensity;
      RealType  NWUEnergy, NEUEnergy, SWUEnergy, SEUEnergy, NWDEnergy, NEDEnergy, SWDEnergy, SEDEnergy;
      PointType NWUMomentum, NEUMomentum, SWUMomentum, SEUMomentum, NWDMomentum, NEDMomentum, SWDMomentum, SEDMomentum;
      RealType gamma;
};

template< typename Mesh >
class RiemannProblemInitialCondition
{
   public:

      typedef Mesh MeshType;
      typedef typename MeshType::RealType RealType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;
      static const int Dimensions = MeshType::getMeshDimension();
      typedef Containers::StaticVector< Dimensions, RealType > PointType;
      typedef Functions::MeshFunctionView< MeshType > MeshFunctionType;
      typedef Pointers::SharedPointer<  MeshFunctionType > MeshFunctionPointer;
      typedef Functions::VectorField< Dimensions, MeshType > VectorFieldType;

      RiemannProblemInitialCondition()
         : discontinuityPlacement( 0.5 ),
           leftDensity( 1.0 ), rightDensity( 1.0 ),
           leftVelocity( -2.0 ), rightVelocity( 2.0 ),
           leftPressure( 0.4 ), rightPressure( 0.4 ),
           gamma( 1.67 ){}

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         config.addEntry< double >( prefix + "discontinuity-placement-x", "x-coordinate of the discontinuity placement.", 0.5 );
         config.addEntry< double >( prefix + "discontinuity-placement-y", "y-coordinate of the discontinuity placement.", 0.5 );
         config.addEntry< double >( prefix + "discontinuity-placement-z", "z-coordinate of the discontinuity placement.", 0.5 );
/*
         config.addEntry< double >( prefix + "left-density", "Density on the left side of the discontinuity.", 1.0 );
         config.addEntry< double >( prefix + "right-density", "Density on the right side of the discontinuity.", 0.0 );
         config.addEntry< double >( prefix + "left-velocity-x", "x-coordinate of the velocity on the left side of the discontinuity.", 1.0 );
         config.addEntry< double >( prefix + "left-velocity-y", "y-coordinate of the velocity on the left side of the discontinuity.", 1.0 );
         config.addEntry< double >( prefix + "left-velocity-z", "z-coordinate of the velocity on the left side of the discontinuity.", 1.0 );
         config.addEntry< double >( prefix + "right-velocity-x", "x-coordinate of the velocity on the right side of the discontinuity.", 0.0 );
         config.addEntry< double >( prefix + "right-velocity-y", "y-coordinate of the velocity on the right side of the discontinuity.", 0.0 );
         config.addEntry< double >( prefix + "right-velocity-z", "z-coordinate of the velocity on the right side of the discontinuity.", 0.0 );
         config.addEntry< double >( prefix + "left-pressure", "Pressure on the left side of the discontinuity.", 1.0 );
         config.addEntry< double >( prefix + "right-pressure", "Pressure on the right side of the discontinuity.", 0.0 );
*/
         config.addEntry< double >( prefix + "NWU-density", "This sets a value of northwest up density.", 1.0  );
         config.addEntry< double >( prefix + "NWU-velocity-x", "This sets a value of northwest up x velocity.", 1.0  );
         config.addEntry< double >( prefix + "NWU-velocity-y", "This sets a value of northwest up y velocity.", 1.0  );
         config.addEntry< double >( prefix + "NWU-velocity-z", "This sets a value of northwest up z velocity.", 1.0  );
         config.addEntry< double >( prefix + "NWU-pressure", "This sets a value of northwest up pressure.", 1.0  );
         config.addEntry< double >( prefix + "SWU-density", "This sets a value of southwest up density.", 1.0  );
         config.addEntry< double >( prefix + "SWU-velocity-x", "This sets a value of southwest up x velocity.", 1.0  );
         config.addEntry< double >( prefix + "SWU-velocity-y", "This sets a value of southwest up y velocity.", 1.0  );
         config.addEntry< double >( prefix + "SWU-velocity-z", "This sets a value of southwest up z velocity.", 1.0  );
         config.addEntry< double >( prefix + "SWU-pressure", "This sets a value of southwest up pressure.", 1.0  );
         config.addEntry< double >( prefix + "NWD-density", "This sets a value of northwest down density.", 1.0  );
         config.addEntry< double >( prefix + "NWD-velocity-x", "This sets a value of northwest down x velocity.", 1.0  );
         config.addEntry< double >( prefix + "NWD-velocity-y", "This sets a value of northwest down y velocity.", 1.0  );
         config.addEntry< double >( prefix + "NWD-velocity-z", "This sets a value of northwest down z velocity.", 1.0  );
         config.addEntry< double >( prefix + "NWD-pressure", "This sets a value of northwest down pressure.", 1.0  );
         config.addEntry< double >( prefix + "SWD-density", "This sets a value of southwest down density.", 1.0  );
         config.addEntry< double >( prefix + "SWD-velocity-x", "This sets a value of southwest down x velocity.", 1.0  );
         config.addEntry< double >( prefix + "SWD-velocity-y", "This sets a value of southwest down y velocity.", 1.0  );
         config.addEntry< double >( prefix + "SWD-velocity-z", "This sets a value of southwest down z velocity.", 1.0  );
         config.addEntry< double >( prefix + "SWD-pressure", "This sets a value of southwest down pressure.", 1.0  );
         config.addEntry< double >( prefix + "NEU-density", "This sets a value of northeast up density.", 1.0  );
         config.addEntry< double >( prefix + "NEU-velocity-x", "This sets a value of northeast up x velocity.", 1.0  );
         config.addEntry< double >( prefix + "NEU-velocity-y", "This sets a value of northeast up y velocity.", 1.0  );
         config.addEntry< double >( prefix + "NEU-velocity-z", "This sets a value of northeast up z velocity.", 1.0  );
         config.addEntry< double >( prefix + "NEU-pressure", "This sets a value of northeast up pressure.", 1.0  );
         config.addEntry< double >( prefix + "SEU-density", "This sets a value of southeast up density.", 1.0  );
         config.addEntry< double >( prefix + "SEU-velocity-x", "This sets a value of southeast up x velocity.", 1.0  );
         config.addEntry< double >( prefix + "SEU-velocity-y", "This sets a value of southeast up y velocity.", 1.0  );
         config.addEntry< double >( prefix + "SEU-velocity-z", "This sets a value of southeast up z velocity.", 1.0  );
         config.addEntry< double >( prefix + "SEU-pressure", "This sets a value of southeast up pressure.", 1.0  );
         config.addEntry< double >( prefix + "NED-density", "This sets a value of northeast down density.", 1.0  );
         config.addEntry< double >( prefix + "NED-velocity-x", "This sets a value of northeast down x velocity.", 1.0  );
         config.addEntry< double >( prefix + "NED-velocity-y", "This sets a value of northeast down y velocity.", 1.0  );
         config.addEntry< double >( prefix + "NED-velocity-z", "This sets a value of northeast down z velocity.", 1.0  );
         config.addEntry< double >( prefix + "NED-pressure", "This sets a value of northeast down pressure.", 1.0  );
         config.addEntry< double >( prefix + "SED-density", "This sets a value of southeast down density.", 1.0  );
         config.addEntry< double >( prefix + "SED-velocity-x", "This sets a value of southeast down x velocity.", 1.0  );
         config.addEntry< double >( prefix + "SED-velocity-y", "This sets a value of southeast down y velocity.", 1.0  );
         config.addEntry< double >( prefix + "SED-velocity-z", "This sets a value of southeast down z velocity.", 1.0  );
         config.addEntry< double >( prefix + "SED-pressure", "This sets a value of southeast down pressure.", 1.0  );
         config.addEntry< double >( prefix + "gamma", "Gamma in the ideal gas state equation.", 1.4 );

         config.addEntry< String >( prefix + "initial", " One of predefined initial condition.", "none");
            config.addEntryEnum< String >( "none" );
            config.addEntryEnum< String >( "1D_2" );
            config.addEntryEnum< String >( "1D_3a" );
            config.addEntryEnum< String >( "1D_4" );
            config.addEntryEnum< String >( "1D_5" );
            config.addEntryEnum< String >( "1D_6" );
            config.addEntryEnum< String >( "1D_Noh" );
            config.addEntryEnum< String >( "1D_peak" );
            config.addEntryEnum< String >( "2D_3" );
            config.addEntryEnum< String >( "2D_4" );
            config.addEntryEnum< String >( "2D_6" );
            config.addEntryEnum< String >( "2D_12" );
            config.addEntryEnum< String >( "2D_15" );
            config.addEntryEnum< String >( "2D_17" );
      }

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         String initial = parameters.getParameter< String >( prefix + "initial" );
         if(initial == prefix + "none")
            {
               this->discontinuityPlacement = parameters.getXyz< PointType >( prefix + "discontinuity-placement" );
               this->gamma = parameters.getParameter< double >( prefix + "gamma" );
/*
               this->leftVelocity = parameters.getXyz< PointType >( prefix + "left-velocity" );
               this->rightVelocity = parameters.getXyz< PointType >( prefix + "right-velocity" );
               this->leftDensity = parameters.getParameter< double >( prefix + "left-density" );
               this->rightDensity = parameters.getParameter< double >( prefix + "right-density" );
               this->leftPressure = parameters.getParameter< double >( prefix + "left-pressure" );
               this->rightPressure = parameters.getParameter< double >( prefix + "right-pressure" );
*/

               this->NWUDensity = parameters.getParameter< RealType >( prefix + "NWU-density" );
               this->NWUVelocity = parameters.getXyz< PointType >( prefix + "NWU-velocity" );
               this->NWUPressure = parameters.getParameter< RealType >( prefix + "NWU-pressure" );
               this->NWUEnergy = Energy( NWUDensity, NWUPressure, gamma, NWUVelocity);
               this->NWUMomentum = NWUVelocity * NWUDensity;

               this->SWUDensity = parameters.getParameter< RealType >( prefix + "SWU-density" );
               this->SWUVelocity = parameters.getXyz< PointType >( prefix + "SWU-velocity" );
               this->SWUPressure = parameters.getParameter< RealType >( prefix + "SWU-pressure" );
               this->SWUEnergy = Energy( SWUDensity, SWUPressure, gamma, SWUVelocity);
               this->SWUMomentum = SWUVelocity * SWUDensity;

               this->NWDDensity = parameters.getParameter< RealType >( prefix + "NWD-density" );
               this->NWDVelocity = parameters.getXyz< PointType >( prefix + "NWD-velocity" );
               this->NWDPressure = parameters.getParameter< RealType >( prefix + "NWD-pressure" );
               this->SWUEnergy = Energy( NWDDensity, NWDPressure, gamma, NWDVelocity);
               this->NWDMomentum = NWDVelocity * NWDDensity;

               this->SWDDensity = parameters.getParameter< RealType >( prefix + "SWD-density" );
               this->SWDVelocity = parameters.getXyz< PointType >( prefix + "SWD-velocity" );
               this->SWDPressure = parameters.getParameter< RealType >( prefix + "SWD-pressure" );
               this->SWDEnergy = Energy( SWDDensity, SWDPressure, gamma, SWDVelocity);
               this->SWDMomentum = SWDVelocity * SWDDensity;

               this->NEUDensity = parameters.getParameter< RealType >( prefix + "NEU-density" );
               this->NEUVelocity = parameters.getXyz< PointType >( prefix + "NEU-velocity" );
               this->NEUPressure = parameters.getParameter< RealType >( prefix + "NEU-pressure" );
               this->NEUEnergy = Energy( NEUDensity, NEUPressure, gamma, NEUVelocity);
               this->NEUMomentum = NEUVelocity * NEUDensity;

               this->SEUDensity = parameters.getParameter< RealType >( prefix + "SEU-density" );
               this->SEUVelocity = parameters.getXyz< PointType >( prefix + "SEU-velocity" );
               this->SEUPressure = parameters.getParameter< RealType >( prefix + "SEU-pressure" );
               this->SEUEnergy = Energy( SEUDensity, SEUPressure, gamma, SEUVelocity);
               this->SEUMomentum = SEUVelocity * SEUDensity;

               this->NEDDensity = parameters.getParameter< RealType >( prefix + "NED-density" );
               this->NEDVelocity = parameters.getXyz< PointType >( prefix + "NED-velocity" );
               this->NEDPressure = parameters.getParameter< RealType >( prefix + "NED-pressure" );
               this->NEDEnergy = Energy( NEDDensity, NEDPressure, gamma, NEDVelocity);
               this->NEDMomentum = NEDVelocity * NEDDensity;

               this->SEDDensity = parameters.getParameter< RealType >( prefix + "SED-density" );
               this->SEDVelocity = parameters.getXyz< PointType >( prefix + "SED-velocity" );
               this->SEDPressure = parameters.getParameter< RealType >( prefix + "SED-pressure" );
               this->SEDEnergy = Energy( SEDDensity, SEDPressure, gamma, SEDVelocity);
               this->SEDMomentum = SEDVelocity * SEDDensity;

           }
         if(initial == prefix + "1D_2")
           predefinedInitialCondition( 1.4, 0.5, 0.0, 0.0, // double preGamma,       double preDiscX,       double preDiscY,       double preDiscZ,
                                       0.0, 0.0, 0.0, 1.0, //double preNWUDensity,  double preSWUDensity,  double preNWDDensity,  double preSWDDensity,
                                       0.0, 0.0, 0.0, 1.0, //double preNEUDensity,  double preSEUDensity,  double preNEDDensity,  double preSEDDensity,
                                       0.0, 0.0, 0.0, 0.4, //double preNWUPressure, double preSWUPressure, double preNWDPressure, double preSWDPressure,
                                       0.0, 0.0, 0.0, 0.4, //double preNEUPressure, double preSEUPressure, double preNEDPressure, double preSEDPressure,
                                       0.0, 0.0, 0.0, //double preNWUVelocityX, double preNWUVelocityY,double preNWUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSWUVelocityX, double preSWUVelocityY,double preSWUVelocityZ,
                                       0.0, 0.0, 0.0, //double preNWDVelocityX, double preNWDVelocityY,double preNWDVelocityZ,
                                       -2.0, 0.0, 0.0, //double preSWDVelocityX, double preSWDVelocityY,double preSWDVelocityZ,
                                       0.0, 0.0, 0.0, //double preNEUVelocityX, double preNEUVelocityY,double preNEUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSEUVelocityX, double preSEUVelocityY,double preSEUVelocityZ,
                                       0.0, 0.0, 0.0, //double preNEDVelocityX, double preNEDVelocityY,double preNEDVelocityZ,
                                       2.0, 0.0, 0.0 //double preSEDVelocityX, double preSEDVelocityY,double preSEDVelocityZ
                                       );
      if(initial == prefix + "1D_3a")
           predefinedInitialCondition( 1.4, 0.8, 0.0, 0.0, // double preGamma,       double preDiscX,       double preDiscY,       double preDiscZ,
                                       0.0, 0.0, 0.0, 1.0, //double preNWUDensity,  double preSWUDensity,  double preNWDDensity,  double preSWDDensity,
                                       0.0, 0.0, 0.0, 1.0, //double preNEUDensity,  double preSEUDensity,  double preNEDDensity,  double preSEDDensity,
                                       0.0, 0.0, 0.0, 1000.0, //double preNWUPressure, double preSWUPressure, double preNWDPressure, double preSWDPressure,
                                       0.0, 0.0, 0.0, 0.01, //double preNEUPressure, double preSEUPressure, double preNEDPressure, double preSEDPressure,
                                       0.0, 0.0, 0.0, //double preNWUVelocityX, double preNWUVelocityY,double preNWUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSWUVelocityX, double preSWUVelocityY,double preSWUVelocityZ,
                                       0.0, 0.0, 0.0, //double preNWDVelocityX, double preNWDVelocityY,double preNWDVelocityZ,
                                       -19.59745, 0.0, 0.0, //double preSWDVelocityX, double preSWDVelocityY,double preSWDVelocityZ,
                                       0.0, 0.0, 0.0, //double preNEUVelocityX, double preNEUVelocityY,double preNEUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSEUVelocityX, double preSEUVelocityY,double preSEUVelocityZ,
                                       0.0, 0.0, 0.0, //double preNEDVelocityX, double preNEDVelocityY,double preNEDVelocityZ,
                                       -19.59745, 0.0, 0.0 //double preSEDVelocityX, double preSEDVelocityY,double preSEDVelocityZ
                                       );
      if(initial == prefix + "1D_4")
           predefinedInitialCondition( 1.666, 0.4, 0.0, 0.0, // double preGamma,       double preDiscX,       double preDiscY,       double preDiscZ,
                                       0.0, 0.0, 0.0, 5.99924, //double preNWUDensity,  double preSWUDensity,  double preNWDDensity,  double preSWDDensity,
                                       0.0, 0.0, 0.0, 5.99242, //double preNEUDensity,  double preSEUDensity,  double preNEDDensity,  double preSEDDensity,
                                       0.0, 0.0, 0.0, 460.894, //double preNWUPressure, double preSWUPressure, double preNWDPressure, double preSWDPressure,
                                       0.0, 0.0, 0.0, 46.095, //double preNEUPressure, double preSEUPressure, double preNEDPressure, double preSEDPressure,
                                       0.0, 0.0, 0.0, //double preNWUVelocityX, double preNWUVelocityY,double preNWUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSWUVelocityX, double preSWUVelocityY,double preSWUVelocityZ,
                                       0.0, 0.0, 0.0, //double preNWDVelocityX, double preNWDVelocityY,double preNWDVelocityZ,
                                       19.5975, 0.0, 0.0, //double preSWDVelocityX, double preSWDVelocityY,double preSWDVelocityZ,
                                       0.0, 0.0, 0.0, //double preNEUVelocityX, double preNEUVelocityY,double preNEUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSEUVelocityX, double preSEUVelocityY,double preSEUVelocityZ,
                                       0.0, 0.0, 0.0, //double preNEDVelocityX, double preNEDVelocityY,double preNEDVelocityZ,
                                       -6.19633, 0.0, 0.0 //double preSEDVelocityX, double preSEDVelocityY,double preSEDVelocityZ
                                       );
      if(initial == prefix + "1D_5")
           predefinedInitialCondition( 1.4, 0.5, 0.0, 0.0, // double preGamma,       double preDiscX,       double preDiscY,       double preDiscZ,
                                       0.0, 0.0, 0.0, 1.4, //double preNWUDensity,  double preSWUDensity,  double preNWDDensity,  double preSWDDensity,
                                       0.0, 0.0, 0.0, 1.0, //double preNEUDensity,  double preSEUDensity,  double preNEDDensity,  double preSEDDensity,
                                       0.0, 0.0, 0.0, 1.0, //double preNWUPressure, double preSWUPressure, double preNWDPressure, double preSWDPressure,
                                       0.0, 0.0, 0.0, 1.0, //double preNEUPressure, double preSEUPressure, double preNEDPressure, double preSEDPressure,
                                       0.0, 0.0, 0.0, //double preNWUVelocityX, double preNWUVelocityY,double preNWUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSWUVelocityX, double preSWUVelocityY,double preSWUVelocityZ,
                                       0.0, 0.0, 0.0, //double preNWDVelocityX, double preNWDVelocityY,double preNWDVelocityZ,
                                       0.0, 0.0, 0.0, //double preSWDVelocityX, double preSWDVelocityY,double preSWDVelocityZ,
                                       0.0, 0.0, 0.0, //double preNEUVelocityX, double preNEUVelocityY,double preNEUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSEUVelocityX, double preSEUVelocityY,double preSEUVelocityZ,
                                       0.0, 0.0, 0.0, //double preNEDVelocityX, double preNEDVelocityY,double preNEDVelocityZ,
                                       0.0, 0.0, 0.0 //double preSEDVelocityX, double preSEDVelocityY,double preSEDVelocityZ
                                       );
      if(initial == prefix + "1D_6")
           predefinedInitialCondition( 1.4, 0.5, 0.0, 0.0, // double preGamma,       double preDiscX,       double preDiscY,       double preDiscZ,
                                       0.0, 0.0, 0.0, 1.4, //double preNWUDensity,  double preSWUDensity,  double preNWDDensity,  double preSWDDensity,
                                       0.0, 0.0, 0.0, 1.0, //double preNEUDensity,  double preSEUDensity,  double preNEDDensity,  double preSEDDensity,
                                       0.0, 0.0, 0.0, 0.1, //double preNWUPressure, double preSWUPressure, double preNWDPressure, double preSWDPressure,
                                       0.0, 0.0, 0.0, 0.1, //double preNEUPressure, double preSEUPressure, double preNEDPressure, double preSEDPressure,
                                       0.0, 0.0, 0.0, //double preNWUVelocityX, double preNWUVelocityY,double preNWUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSWUVelocityX, double preSWUVelocityY,double preSWUVelocityZ,
                                       0.0, 0.0, 0.0, //double preNWDVelocityX, double preNWDVelocityY,double preNWDVelocityZ,
                                       1.0, 0.0, 0.0, //double preSWDVelocityX, double preSWDVelocityY,double preSWDVelocityZ,
                                       0.0, 0.0, 0.0, //double preNEUVelocityX, double preNEUVelocityY,double preNEUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSEUVelocityX, double preSEUVelocityY,double preSEUVelocityZ,
                                       0.0, 0.0, 0.0, //double preNEDVelocityX, double preNEDVelocityY,double preNEDVelocityZ,
                                       1.0, 0.0, 0.0 //double preSEDVelocityX, double preSEDVelocityY,double preSEDVelocityZ
                                       );
      if(initial == prefix + "1D_Noh")
           predefinedInitialCondition( 1.4, 0.5, 0.0, 0.0, // double preGamma,       double preDiscX,       double preDiscY,       double preDiscZ,
                                       0.0, 0.0, 0.0, 1.0, //double preNWUDensity,  double preSWUDensity,  double preNWDDensity,  double preSWDDensity,
                                       0.0, 0.0, 0.0, 1.0, //double preNEUDensity,  double preSEUDensity,  double preNEDDensity,  double preSEDDensity,
                                       0.0, 0.0, 0.0, 0.000001, //double preNWUPressure, double preSWUPressure, double preNWDPressure, double preSWDPressure,
                                       0.0, 0.0, 0.0, 0.000001, //double preNEUPressure, double preSEUPressure, double preNEDPressure, double preSEDPressure,
                                       0.0, 0.0, 0.0, //double preNWUVelocityX, double preNWUVelocityY,double preNWUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSWUVelocityX, double preSWUVelocityY,double preSWUVelocityZ,
                                       0.0, 0.0, 0.0, //double preNWDVelocityX, double preNWDVelocityY,double preNWDVelocityZ,
                                       1.0, 0.0, 0.0, //double preSWDVelocityX, double preSWDVelocityY,double preSWDVelocityZ,
                                       0.0, 0.0, 0.0, //double preNEUVelocityX, double preNEUVelocityY,double preNEUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSEUVelocityX, double preSEUVelocityY,double preSEUVelocityZ,
                                       0.0, 0.0, 0.0, //double preNEDVelocityX, double preNEDVelocityY,double preNEDVelocityZ,
                                       -1.0, 0.0, 0.0 //double preSEDVelocityX, double preSEDVelocityY,double preSEDVelocityZ
                                       );
      if(initial == prefix + "1D_peak")
           predefinedInitialCondition( 1.4, 0.5, 0.0, 0.0, // double preGamma,       double preDiscX,       double preDiscY,       double preDiscZ,
                                       0.0, 0.0, 0.0, 0.12612, //double preNWUDensity,  double preSWUDensity,  double preNWDDensity,  double preSWDDensity,
                                       0.0, 0.0, 0.0, 6.5915, //double preNEUDensity,  double preSEUDensity,  double preNEDDensity,  double preSEDDensity,
                                       0.0, 0.0, 0.0, 782.929, //double preNWUPressure, double preSWUPressure, double preNWDPressure, double preSWDPressure,
                                       0.0, 0.0, 0.0, 3.15449, //double preNEUPressure, double preSEUPressure, double preNEDPressure, double preSEDPressure,
                                       0.0, 0.0, 0.0, //double preNWUVelocityX, double preNWUVelocityY,double preNWUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSWUVelocityX, double preSWUVelocityY,double preSWUVelocityZ,
                                       0.0, 0.0, 0.0, //double preNWDVelocityX, double preNWDVelocityY,double preNWDVelocityZ,
                                       8.90470, 0.0, 0.0, //double preSWDVelocityX, double preSWDVelocityY,double preSWDVelocityZ,
                                       0.0, 0.0, 0.0, //double preNEUVelocityX, double preNEUVelocityY,double preNEUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSEUVelocityX, double preSEUVelocityY,double preSEUVelocityZ,
                                       0.0, 0.0, 0.0, //double preNEDVelocityX, double preNEDVelocityY,double preNEDVelocityZ,
                                       2.26542, 0.0, 0.0 //double preSEDVelocityX, double preSEDVelocityY,double preSEDVelocityZ
                                       );
      if(initial == prefix + "2D_3")
           predefinedInitialCondition( 1.666, 0.5, 0.5, 0.0, // double preGamma,       double preDiscX,       double preDiscY,       double preDiscZ,
                                       0.0, 0.0, 0.5323, 0.138, //double preNWUDensity,  double preSWUDensity,  double preNWDDensity,  double preSWDDensity,
                                       0.0, 0.0, 1.5, 0.5323, //double preNEUDensity,  double preSEUDensity,  double preNEDDensity,  double preSEDDensity,
                                       0.0, 0.0, 0.3, 0.029, //double preNWUPressure, double preSWUPressure, double preNWDPressure, double preSWDPressure,
                                       0.0, 0.0, 1.5, 0.3, //double preNEUPressure, double preSEUPressure, double preNEDPressure, double preSEDPressure,
                                       0.0, 0.0, 0.0, //double preNWUVelocityX, double preNWUVelocityY,double preNWUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSWUVelocityX, double preSWUVelocityY,double preSWUVelocityZ,
                                       1.206, 0.0, 0.0, //double preNWDVelocityX, double preNWDVelocityY,double preNWDVelocityZ,
                                       1.206, 1.206, 0.0, //double preSWDVelocityX, double preSWDVelocityY,double preSWDVelocityZ,
                                       0.0, 0.0, 0.0, //double preNEUVelocityX, double preNEUVelocityY,double preNEUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSEUVelocityX, double preSEUVelocityY,double preSEUVelocityZ,
                                       0.0, 0.0, 0.0, //double preNEDVelocityX, double preNEDVelocityY,double preNEDVelocityZ,
                                       0.0, 1.206, 0.0 //double preSEDVelocityX, double preSEDVelocityY,double preSEDVelocityZ
                                       );
      if(initial == prefix + "2D_4")
           predefinedInitialCondition( 1.666, 0.5, 0.5, 0.0, // double preGamma,       double preDiscX,       double preDiscY,       double preDiscZ,
                                       0.0, 0.0, 0.5065, 1.1, //double preNWUDensity,  double preSWUDensity,  double preNWDDensity,  double preSWDDensity,
                                       0.0, 0.0, 1.1, 0.5065, //double preNEUDensity,  double preSEUDensity,  double preNEDDensity,  double preSEDDensity,
                                       0.0, 0.0, 0.35, 1.1, //double preNWUPressure, double preSWUPressure, double preNWDPressure, double preSWDPressure,
                                       0.0, 0.0, 1.1, 0.35, //double preNEUPressure, double preSEUPressure, double preNEDPressure, double preSEDPressure,
                                       0.0, 0.0, 0.0, //double preNWUVelocityX, double preNWUVelocityY,double preNWUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSWUVelocityX, double preSWUVelocityY,double preSWUVelocityZ,
                                       0.8939, 0.0, 0.0, //double preNWDVelocityX, double preNWDVelocityY,double preNWDVelocityZ,
                                       0.8939, 0.8939, 0.0, //double preSWDVelocityX, double preSWDVelocityY,double preSWDVelocityZ,
                                       0.0, 0.0, 0.0, //double preNEUVelocityX, double preNEUVelocityY,double preNEUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSEUVelocityX, double preSEUVelocityY,double preSEUVelocityZ,
                                       0.0, 0.0, 0.0, //double preNEDVelocityX, double preNEDVelocityY,double preNEDVelocityZ,
                                       0.0, 0.8939, 0.0 //double preSEDVelocityX, double preSEDVelocityY,double preSEDVelocityZ
                                       );

      if(initial == prefix + "2D_6")
           predefinedInitialCondition( 1.666, 0.5, 0.5, 0.0, // double preGamma,       double preDiscX,       double preDiscY,       double preDiscZ,
                                       0.0, 0.0, 2.0, 1.0, //double preNWUDensity,  double preSWUDensity,  double preNWDDensity,  double preSWDDensity,
                                       0.0, 0.0, 1.0, 3.0, //double preNEUDensity,  double preSEUDensity,  double preNEDDensity,  double preSEDDensity,
                                       0.0, 0.0, 1.0, 1.0, //double preNWUPressure, double preSWUPressure, double preNWDPressure, double preSWDPressure,
                                       0.0, 0.0, 1.0, 1.0, //double preNEUPressure, double preSEUPressure, double preNEDPressure, double preSEDPressure,
                                       0.0, 0.0, 0.0, //double preNWUVelocityX, double preNWUVelocityY,double preNWUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSWUVelocityX, double preSWUVelocityY,double preSWUVelocityZ,
                                       0.75, 0.5, 0.0, //double preNWDVelocityX, double preNWDVelocityY,double preNWDVelocityZ,
                                       -0.75, 0.5, 0.0, //double preSWDVelocityX, double preSWDVelocityY,double preSWDVelocityZ,
                                       0.0, 0.0, 0.0, //double preNEUVelocityX, double preNEUVelocityY,double preNEUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSEUVelocityX, double preSEUVelocityY,double preSEUVelocityZ,
                                       0.75, -0.5, 0.0, //double preNEDVelocityX, double preNEDVelocityY,double preNEDVelocityZ,
                                       -0.75, -0.5, 0.0 //double preSEDVelocityX, double preSEDVelocityY,double preSEDVelocityZ
                                       );
      if(initial == prefix + "2D_12")
           predefinedInitialCondition( 1.666, 0.5, 0.5, 0.0, // double preGamma,       double preDiscX,       double preDiscY,       double preDiscZ,
                                       0.0, 0.0, 1.0, 0.8, //double preNWUDensity,  double preSWUDensity,  double preNWDDensity,  double preSWDDensity,
                                       0.0, 0.0, 0.5313, 1.0, //double preNEUDensity,  double preSEUDensity,  double preNEDDensity,  double preSEDDensity,
                                       0.0, 0.0, 1.0, 1.0, //double preNWUPressure, double preSWUPressure, double preNWDPressure, double preSWDPressure,
                                       0.0, 0.0, 0.4, 1.0, //double preNEUPressure, double preSEUPressure, double preNEDPressure, double preSEDPressure,
                                       0.0, 0.0, 0.0, //double preNWUVelocityX, double preNWUVelocityY,double preNWUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSWUVelocityX, double preSWUVelocityY,double preSWUVelocityZ,
                                       0.7276, 0.0, 0.0, //double preNWDVelocityX, double preNWDVelocityY,double preNWDVelocityZ,
                                       0.0, 0.0, 0.0, //double preSWDVelocityX, double preSWDVelocityY,double preSWDVelocityZ,
                                       0.0, 0.0, 0.0, //double preNEUVelocityX, double preNEUVelocityY,double preNEUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSEUVelocityX, double preSEUVelocityY,double preSEUVelocityZ,
                                       0.0, 0.0, 0.0, //double preNEDVelocityX, double preNEDVelocityY,double preNEDVelocityZ,
                                       0.0, 0.7276, 0.0 //double preSEDVelocityX, double preSEDVelocityY,double preSEDVelocityZ
                                       );

      if(initial == prefix + "2D_15")
           predefinedInitialCondition( 1.666, 0.5, 0.5, 0.0, // double preGamma,       double preDiscX,       double preDiscY,       double preDiscZ,
                                       0.0, 0.0, 0.5197, 0.8, //double preNWUDensity,  double preSWUDensity,  double preNWDDensity,  double preSWDDensity,
                                       0.0, 0.0, 1.0, 0.5313, //double preNEUDensity,  double preSEUDensity,  double preNEDDensity,  double preSEDDensity,
                                       0.0, 0.0, 0.4, 0.4, //double preNWUPressure, double preSWUPressure, double preNWDPressure, double preSWDPressure,
                                       0.0, 0.0, 1.0, 0.4, //double preNEUPressure, double preSEUPressure, double preNEDPressure, double preSEDPressure,
                                       0.0, 0.0, 0.0, //double preNWUVelocityX, double preNWUVelocityY,double preNWUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSWUVelocityX, double preSWUVelocityY,double preSWUVelocityZ,
                                       -0.6259, -0.3, 0.0, //double preNWDVelocityX, double preNWDVelocityY,double preNWDVelocityZ,
                                       0.1, -0.3, 0.0, //double preSWDVelocityX, double preSWDVelocityY,double preSWDVelocityZ,
                                       0.0, 0.0, 0.0, //double preNEUVelocityX, double preNEUVelocityY,double preNEUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSEUVelocityX, double preSEUVelocityY,double preSEUVelocityZ,
                                       0.1, -0.3, 0.0, //double preNEDVelocityX, double preNEDVelocityY,double preNEDVelocityZ,
                                       0.1, 0.4276, 0.0 //double preSEDVelocityX, double preSEDVelocityY,double preSEDVelocityZ
                                       );
      if(initial == prefix + "2D_17")
           predefinedInitialCondition( 1.666, 0.5, 0.5, 0.0, // double preGamma,       double preDiscX,       double preDiscY,       double preDiscZ,
                                       0.0, 0.0, 2.0, 1.0625, //double preNWUDensity,  double preSWUDensity,  double preNWDDensity,  double preSWDDensity,
                                       0.0, 0.0, 1.0, 0.5197, //double preNEUDensity,  double preSEUDensity,  double preNEDDensity,  double preSEDDensity,
                                       0.0, 0.0, 1.0, 0.4, //double preNWUPressure, double preSWUPressure, double preNWDPressure, double preSWDPressure,
                                       0.0, 0.0, 1.0, 0.4, //double preNEUPressure, double preSEUPressure, double preNEDPressure, double preSEDPressure,
                                       0.0, 0.0, 0.0, //double preNWUVelocityX, double preNWUVelocityY,double preNWUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSWUVelocityX, double preSWUVelocityY,double preSWUVelocityZ,
                                       0.0, -0.3, 0.0, //double preNWDVelocityX, double preNWDVelocityY,double preNWDVelocityZ,
                                       0.0, 0.2145, 0.0, //double preSWDVelocityX, double preSWDVelocityY,double preSWDVelocityZ,
                                       0.0, 0.0, 0.0, //double preNEUVelocityX, double preNEUVelocityY,double preNEUVelocityZ,
                                       0.0, 0.0, 0.0, //double preSEUVelocityX, double preSEUVelocityY,double preSEUVelocityZ,
                                       0.0, -0.4, 0.0, //double preNEDVelocityX, double preNEDVelocityY,double preNEDVelocityZ,
                                       0.0, 1.1259, 0.0 //double preSEDVelocityX, double preSEDVelocityY,double preSEDVelocityZ
                                       );
         return true;
      }

      void setDiscontinuityPlacement( const PointType& v )
      {
         this->discontinuityPlacement = v;
      }

      const PointType& getDiscontinuityPlasement() const
      {
         return this->discontinuityPlacement;
      }

      void setLeftDensity( const RealType& leftDensity )
      {
         this->leftDensity = leftDensity;
      }

      const RealType& getLeftDensity() const
      {
         return this->leftDensity;
      }

      void setRightDensity( const RealType& rightDensity )
      {
         this->rightDensity = rightDensity;
      }

      const RealType& getRightDensity() const
      {
         return this->rightDensity;
      }

      void setLeftVelocity( const PointType& leftVelocity )
      {
         this->leftVelocity = leftVelocity;
      }

      const PointType& getLeftVelocity() const
      {
         return this->leftVelocity;
      }

      void setRightVelocity( const RealType& rightVelocity )
      {
         this->rightVelocity = rightVelocity;
      }

      const PointType& getRightVelocity() const
      {
         return this->rightVelocity;
      }

      void setLeftPressure( const RealType& leftPressure )
      {
         this->leftPressure = leftPressure;
      }

      const RealType& getLeftPressure() const
      {
         return this->leftPressure;
      }

      void setRightPressure( const RealType& rightPressure )
      {
         this->rightPressure = rightPressure;
      }

      const RealType& getRightPressure() const
      {
         return this->rightPressure;
      }


      void predefinedInitialCondition( double preGamma,       double preDiscX,       double preDiscY,       double preDiscZ,
                                       double preNWUDensity,  double preSWUDensity,  double preNWDDensity,  double preSWDDensity,
                                       double preNEUDensity,  double preSEUDensity,  double preNEDDensity,  double preSEDDensity,
                                       double preNWUPressure, double preSWUPressure, double preNWDPressure, double preSWDPressure,
                                       double preNEUPressure, double preSEUPressure, double preNEDPressure, double preSEDPressure,
                                       double preNWUVelocityX, double preNWUVelocityY,double preNWUVelocityZ,
                                       double preSWUVelocityX, double preSWUVelocityY,double preSWUVelocityZ,
                                       double preNWDVelocityX, double preNWDVelocityY,double preNWDVelocityZ,
                                       double preSWDVelocityX, double preSWDVelocityY,double preSWDVelocityZ,
                                       double preNEUVelocityX, double preNEUVelocityY,double preNEUVelocityZ,
                                       double preSEUVelocityX, double preSEUVelocityY,double preSEUVelocityZ,
                                       double preNEDVelocityX, double preNEDVelocityY,double preNEDVelocityZ,
                                       double preSEDVelocityX, double preSEDVelocityY,double preSEDVelocityZ
                                       )

      {
         this->discontinuityPlacement = PointLoad(preDiscX, preDiscY, preDiscZ);
         this->gamma = preGamma;

         this->NWUDensity = preNWUDensity;
         this->NWUVelocity = PointLoad(preNWUVelocityX, preNWUVelocityY, preNWUVelocityZ);
         this->NWUPressure = preNWUPressure;
         this->NWUEnergy = Energy( NWUDensity, NWUPressure, gamma, NWUVelocity);
         this->NWUMomentum = NWUVelocity * NWUDensity;

         this->SWUDensity = preNWUDensity;
         this->SWUVelocity = PointLoad(preSWUVelocityX, preSWUVelocityY, preSWUVelocityZ);
         this->SWUPressure = preSWUPressure;
         this->SWUEnergy = Energy( SWUDensity, SWUPressure, gamma, SWUVelocity);
         this->SWUMomentum = SWUVelocity * SWUDensity;

         this->NWDDensity = preNWDDensity;
         this->NWDVelocity = PointLoad(preNWDVelocityX, preNWDVelocityY, preNWDVelocityZ);
         this->NWDPressure = preNWDPressure;
         this->NWDEnergy = Energy( NWDDensity, NWDPressure, gamma, NWDVelocity);
         this->NWDMomentum = NWDVelocity * NWDDensity;

         this->SWDDensity = preSWDDensity;
         this->SWDVelocity = PointLoad(preSWDVelocityX, preSWDVelocityY, preSWDVelocityZ);
         this->SWDPressure = preSWDPressure;
         this->SWDEnergy = Energy( SWDDensity, SWDPressure, gamma, SWDVelocity);
         this->SWDMomentum = SWDVelocity * SWDDensity;

         this->NEUDensity = preNEUDensity;
         this->NEUVelocity = PointLoad(preNEUVelocityX, preNEUVelocityY, preNEUVelocityZ);
         this->NEUPressure = preNEUPressure;
         this->NEUEnergy = Energy( NEUDensity, NEUPressure, gamma, NEUVelocity);
         this->NEUMomentum = NEUVelocity * NEUDensity;

         this->SEUDensity = preSEUDensity;
         this->SEUVelocity = PointLoad(preSEUVelocityX, preSEUVelocityY, preSEUVelocityZ);
         this->SEUPressure = preSEUPressure;
         this->SEUEnergy = Energy( SEUDensity, SEUPressure, gamma, SEUVelocity);
         this->SEUMomentum = SEUVelocity * SEUDensity;

         this->NEDDensity = preNEDDensity;
         this->NEDVelocity = PointLoad(preNEDVelocityX, preNEDVelocityY, preNEDVelocityZ);
         this->NEDPressure = preNEDPressure;
         this->NEDEnergy = Energy( NEDDensity, NEDPressure, gamma, NEDVelocity);
         this->NEDMomentum = NEDVelocity * NEDDensity;

         this->SEDDensity = preSEDDensity;
         this->SEDVelocity = PointLoad(preSEDVelocityX, preSEDVelocityY, preSEDVelocityZ);
         this->SEDPressure = preSEDPressure;
         this->SEDEnergy = Energy( SEDDensity, SEDPressure, gamma, SEDVelocity);
         this->SEDMomentum = SEDVelocity * SEDDensity;

         std::cout << this->SEDEnergy;
         std::cout << this->SWDEnergy;

      }

      PointType PointLoad( RealType ValueX, RealType ValueY, RealType ValueZ)
      {
         PointType point;
         switch (Dimensions)
         {
            case 1: point[ 0 ] = ValueX;
                    break;
            case 2: point[ 0 ] = ValueX;
                    point[ 1 ] = ValueY;
                    break;
            case 3: point[ 0 ] = ValueX;
                    point[ 1 ] = ValueY;
                    point[ 2 ] = ValueZ;
                    break;
         }
         return point;
      }

      RealType Energy( RealType Density, RealType Pressure, RealType gamma, PointType Velocity)
      {
         RealType energy;
         switch (Dimensions)
         {
            case 1: energy = (Pressure / (gamma -1.0) + 0.5 * Density * (std::pow(Velocity[ 0 ], 2 )));
                    break;
            case 2: energy = (Pressure / (gamma -1.0) + 0.5 * Density * (std::pow(Velocity[ 0 ], 2 ) + std::pow(Velocity[ 1 ], 2 )));
                    break;
            case 3: energy = (Pressure / (gamma -1.0) + 0.5 * Density * (std::pow(Velocity[ 0 ], 2 ) + std::pow(Velocity[ 1 ], 2 ) + std::pow(Velocity[ 3 ], 2 )));
                    break; // druhou mocninu ps8t jako sou4in
         }
         return energy;
      }

      void setInitialCondition( CompressibleConservativeVariables< MeshType >& conservativeVariables,
                                const PointType& center = PointType( 0.0 ) )
      {
         RiemannProblemInitialConditionSetter<MeshType>* variablesSetter = new RiemannProblemInitialConditionSetter<MeshType>;
         variablesSetter->setGamma(this->gamma);
         variablesSetter->setDensity(this->NWUDensity,
                                     this->NEUDensity,
                                     this->SWUDensity,
                                     this->SEUDensity,
                                     this->NWDDensity,
                                     this->NEDDensity,
                                     this->SWDDensity,
                                     this->SEDDensity);
         variablesSetter->setMomentum(this->NWUMomentum,
                                      this->NEUMomentum,
                                      this->SWUMomentum,
                                      this->SEUMomentum,
                                      this->NWDMomentum,
                                      this->NEDMomentum,
                                      this->SWDMomentum,
                                      this->SEDMomentum);
         variablesSetter->setEnergy(this->NWUEnergy,
                                    this->NEUEnergy,
                                    this->SWUEnergy,
                                    this->SEUEnergy,
                                    this->NWDEnergy,
                                    this->NEDEnergy,
                                    this->SWDEnergy,
                                    this->SEDEnergy);
         variablesSetter->setDiscontinuity(this->discontinuityPlacement);
         variablesSetter->placeDensity(conservativeVariables);
         variablesSetter->placeMomentum(conservativeVariables);
         variablesSetter->placeEnergy(conservativeVariables);

//       for cyklus i = 0 to mesh.getDimensions().x() j pro .y() a k pro .z()
//       typedef typename MeshType::Cell CellType
//       typedef typename MeshType::CoordinatesType CoordinatesType
//       Celltype cell(mesh, CoordinatesType(i,j))
//       p59stup do density setElement(mesh.template getEntityIndex< CellType >(cell), hodnota, kterou budu zapisovat)
//       pomocn8 t59da, kterou budu specialiyovat p5es r;zn0 dimenze gridu

/*
         typedef Functions::Analytic::VectorNorm< Dimensions, RealType > VectorNormType;
         typedef Operators::Analytic::Sign< Dimensions, RealType > SignType;
         typedef Functions::OperatorFunction< SignType, VectorNormType > InitialConditionType;
         typedef Pointers::SharedPointer<  InitialConditionType, DeviceType > InitialConditionPointer;

         InitialConditionPointer initialCondition;
         initialCondition->getFunction().setCenter( center );
         initialCondition->getFunction().setMaxNorm( true );
         initialCondition->getFunction().setRadius( discontinuityPlacement[ 0 ] );
         discontinuityPlacement *= 1.0 / discontinuityPlacement[ 0 ];
         for( int i = 1; i < Dimensions; i++ )
            discontinuityPlacement[ i ] = 1.0 / discontinuityPlacement[ i ];
         initialCondition->getFunction().setAnisotropy( discontinuityPlacement );
         initialCondition->getFunction().setMultiplicator( -1.0 );

         Functions::MeshFunctionEvaluator< MeshFunctionType, InitialConditionType > evaluator;
*/
         /****
          * Density
          */
/*
         conservativeVariables.getDensity()->write( "density.gplt", "gnuplot" );
*/
/*
         initialCondition->getOperator().setPositiveValue( leftDensity );
         initialCondition->getOperator().setNegativeValue( rightDensity );
         evaluator.evaluate( conservativeVariables.getDensity(), initialCondition );
         conservativeVariables.getDensity()->write( "density.gplt", "gnuplot" );
*/
         /****
          * Momentum
          */

/*
         for( int i = 0; i < Dimensions; i++ )
         {
            initialCondition->getOperator().setPositiveValue( leftDensity * leftVelocity[ i ] );
            initialCondition->getOperator().setNegativeValue( rightDensity * rightVelocity[ i ] );
            evaluator.evaluate( conservativeVariables.getMomentum()[ i ], initialCondition );
         }
*/
         /****
          * Energy
          */
/*
         conservativeVariables.getEnergy()->write( "energy-init", "gnuplot" );
*/
/*
         const RealType leftKineticEnergy = leftVelocity.lpNorm( 2.0 );
         const RealType rightKineticEnergy = rightVelocity.lpNorm( 2.0 );
         const RealType leftEnergy = leftPressure / ( gamma - 1.0 ) + 0.5 * leftDensity * leftKineticEnergy * leftKineticEnergy;
         const RealType rightEnergy = rightPressure / ( gamma - 1.0 ) + 0.5 * rightDensity * rightKineticEnergy * rightKineticEnergy;
         initialCondition->getOperator().setPositiveValue( leftEnergy );
         initialCondition->getOperator().setNegativeValue( rightEnergy );
         evaluator.evaluate( (* conservativeVariables.getEnergy()), initialCondition );
         (* conservativeVariables.getEnergy())->write( "energy-init", "gnuplot" );
*/
      }


   protected:

      PointType discontinuityPlacement;
      PointType NWUVelocity, NEUVelocity, SWUVelocity, SEUVelocity, NWDVelocity, NEDVelocity, SWDVelocity, SEDVelocity;
      RealType  NWUDensity, NEUDensity, SWUDensity, SEUDensity, NWDDensity, NEDDensity, SWDDensity, SEDDensity;
      RealType  NWUPressure, NEUPressure, SWUPressure, SEUPressure, NWDPressure, NEDPressure, SWDPressure, SEDPressure;
      RealType  NWUEnergy, NEUEnergy, SWUEnergy, SEUEnergy, NWDEnergy, NEDEnergy, SWDEnergy, SEDEnergy;
      PointType NWUMomentum, NEUMomentum, SWUMomentum, SEUMomentum, NWDMomentum, NEDMomentum, SWDMomentum, SEDMomentum;
      RealType  leftDensity, rightDensity;
      PointType leftVelocity, rightVelocity;
      RealType  leftPressure, rightPressure;

      RealType gamma; // gamma in the ideal gas state equation
};

} //namespace TNL
