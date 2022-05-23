#pragma once


#include <iostream>
#include <sstream>
#include <TNL/Functions/MeshFunction.h>


using namespace TNL;
using namespace TNL::Functions;

template <typename Real,
        int Dim>
class LinearFunction{};


template <typename Real,
        int Dim>
class ConstFunction{};

template < typename GridType,
        typename DofType,
        int Dim=GridType::getMeshDimension()>
class Printer
{};

//=================================1D=====================================================
template <typename Real>
class LinearFunction<Real,1> : public Functions::Domain< 1, Functions::MeshDomain >
{
   public:
      typedef Real RealType;
      LinearFunction( )
      {};

      template< typename EntityType >
      __cuda_callable__ RealType operator()( const EntityType& meshEntity,
                                  const RealType& time = 0.0 ) const
      {
         return meshEntity.getCenter().x();
         
      }
};

template <typename Real>
class ConstFunction<Real,1> : public Functions::Domain< 1, Functions::MeshDomain >
{
   public:
      typedef Real RealType;
          
          Real Number;
          
      ConstFunction( )
      {};

      template< typename EntityType >
      __cuda_callable__ RealType operator()( const EntityType& meshEntity,
                                  const RealType& time = 0.0 ) const
      {
         return Number;
         
      }
};

template<typename GridType, typename DofType>
class Printer< GridType,DofType,1>
{
    public:
    void static print_dof(int rank, GridType grid, DofType dof)
    {
    std::stringstream sout;
    for(int i=0;i<dof.getSize();i++) 
        sout<< dof[i] << " ";
    
    std::cout << rank << ":   " << sout.str() << std::endl;
    };
};

//=================================2D======================================================

template <typename Real>
class LinearFunction<Real,2> : public Functions::Domain< 2, Functions::MeshDomain >
{
   public:
      typedef Real RealType;
      LinearFunction( )
      {};

      template< typename EntityType >
      __cuda_callable__ RealType operator()( const EntityType& meshEntity,
                                  const RealType& time = 0.0 ) const
      {
         //return meshEntity.getCoordinates().y()*10+meshEntity.getCoordinates().x();
         return meshEntity.getCenter().y()*100+meshEntity.getCenter().x();
      }
};

template <typename Real>
class ConstFunction<Real,2> : public Functions::Domain< 2, Functions::MeshDomain >
{
   public:
          typedef Real RealType;
          
          Real Number;
      ConstFunction( )
      {};
          
      template< typename EntityType >
      __cuda_callable__ RealType operator()( const EntityType& meshEntity,
                                  const RealType& time = 0.0 ) const
      {
         //return meshEntity.getCoordinates().y()*10+meshEntity.getCoordinates().x();
         return this->Number;
         
      }
};

template<typename GridType, typename DofType>
class Printer< GridType,DofType,2>
{
    public:
    void static print_dof(int rank,GridType grid, DofType dof)
    {
    int maxx=grid.getDimensions().x();
    int maxy=grid.getDimensions().y();
    std::stringstream sout;
        sout<< rank<<":" <<std::endl;
    for(int j=0;j<maxy;j++)
    {
        for(int i=0;i<maxx;i++)
            sout<< dof[j*maxx+i] << " ";
        sout<<std::endl;
    }
    std::cout << sout.str() << std::endl<< std::endl;
    };
};

//============================3D============================================================
template <typename Real>
class LinearFunction<Real,3> : public Functions::Domain< 3, Functions::MeshDomain >
{
   public:
      typedef Real RealType;
      LinearFunction( )
      {};

      template< typename EntityType >
      __cuda_callable__ RealType operator()( const EntityType& meshEntity,
                                  const RealType& time = 0.0 ) const
      {
         //return meshEntity.getCoordinates().y()*10+meshEntity.getCoordinates().x();
         return meshEntity.getCenter().z()*10000+meshEntity.getCenter().y()*100+meshEntity.getCenter().x();
      }
};

template <typename Real>
class ConstFunction<Real,3> : public Functions::Domain< 3, Functions::MeshDomain >
{
   public:
          typedef Real RealType;
          
          Real Number;
      ConstFunction( )
      {};
          
      template< typename EntityType >
      __cuda_callable__ RealType operator()( const EntityType& meshEntity,
                                  const RealType& time = 0.0 ) const
      {
         //return meshEntity.getCoordinates().y()*10+meshEntity.getCoordinates().x();
         return this->Number;
         
      }
};

template<typename GridType, typename DofType>
class Printer< GridType,DofType,3>
{
    public:
    static void print_dof(int rank,GridType grid, DofType dof)
    {
      //print local dof
      int maxx=grid.getDimensions().x();
      int maxy=grid.getDimensions().y();
      int maxz=grid.getDimensions().z();

      std::stringstream sout;
      sout<< rank<<":"  <<std::endl;
      for(int k=0;k<maxz;k++)
      {
            for(int j=0;j<maxy;j++)
            {
                    for(int ii=0;ii<k;ii++)
                            sout<<"  ";
                    for(int i=0;i<maxx;i++)
                    {
                            sout <<dof[k*maxx*maxy+maxx*j+i]<<"  ";
                    }
                    sout << std::endl;
            }
      }
      std::cout << sout.str()<< std::endl<<std::endl;
    };
};
