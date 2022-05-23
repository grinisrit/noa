#pragma once 




//#include <mesh/grids/Grid2D.h>

#include <core/tnlObject.h>
#include <core/Devices::Host.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Containers/Vector.h>

template< int Dimension,
          typename Real = double,
          typename Device = Devices::Host,
          typename Index = int >
class Grid : public tnlObject
{
};

//#include <mesh/grids/Grid1D.h>
//#include <mesh/grids/Grid3D.h>


#include <mesh/grids/GridEntityTopology.h>
#include <mesh/grids/GridEntityGetter.h>
#include <mesh/grids/GridEntityConfig.h>
#include <mesh/grids/NeighborGridEntityGetter.h>
#include <core/tnlLogger.h>

// TODO: remove this
//#include <../tests/benchmarks/heat-equation-benchmark/tnlTestGridEntity.h>
//#include <../tests/benchmarks/heat-equation-benchmark/tnlTestNeighborGridEntityGetter2D_impl.h>
/////

template< typename Real,
          typename Device,
          typename Index >
class Meshes::Grid< 2, Real, Device, Index > : public tnlObject
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef Containers::StaticVector< 2, Real > PointType;
   typedef Containers::StaticVector< 2, Index > CoordinatesType;
   
   static const int meshDimension = 2;

   //template< int EntityDimension, 
   //          typename Config = GridEntityNoStencilStorage >//CrossStencilStorage< 1 > >
   //using MeshEntity = GridEntity< ThisType, EntityDimension, Config >;
   
   //typedef MeshEntity< meshDimension, GridEntityCrossStencilStorage< 1 > > Cell;
   //typedef MeshEntity< meshDimension - 1, GridEntityNoStencilStorage > Face;
   //typedef MeshEntity< 0 > Point;
   

   // TODO: remove this
   template< int EntityDimension, 
             typename Config = GridEntityNoStencilStorage >//CrossStencilStorage< 1 > >
   using TestMeshEntity = tnlTestGridEntity< ThisType, EntityDimension, Config >;
   typedef TestMeshEntity< meshDimension, GridEntityCrossStencilStorage< 1 > > TestCell;
   /////
   
   static constexpr int getMeshDimension() { return meshDimension; };

   Grid();

   static String getSerializationType();

   virtual String getSerializationTypeVirtual() const;

   void setDimensions( const Index xSize, const Index ySize );

   void setDimensions( const CoordinatesType& dimensions );

   __cuda_callable__
   inline const CoordinatesType& getDimensions() const;

   void setDomain( const PointType& origin,
                   const PointType& proportions );
   __cuda_callable__
   inline const PointType& getOrigin() const;

   __cuda_callable__
   inline const PointType& getProportions() const;

   template< typename EntityType >
   __cuda_callable__
   inline IndexType getEntitiesCount() const;
   
   template< typename EntityType >
   __cuda_callable__
   inline EntityType getEntity( const IndexType& entityIndex ) const;
   
   template< typename EntityType >
   __cuda_callable__
   inline Index getEntityIndex( const EntityType& entity ) const;

   template< typename EntityType >
   __cuda_callable__
   RealType getEntityMeasure( const EntityType& entity ) const;
      
   __cuda_callable__
   RealType getCellMeasure() const;
   
   __cuda_callable__
   inline PointType getSpaceSteps() const;

   template< int xPow, int yPow >
   __cuda_callable__
   inline const RealType& getSpaceStepsProducts() const;
   
   __cuda_callable__
   inline RealType getSmallestSpaceStep() const;

   
   template< typename GridFunction >
   typename GridFunction::RealType getAbsMax( const GridFunction& f ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getLpNorm( const GridFunction& f,
                                              const typename GridFunction::RealType& p ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceAbsMax( const GridFunction& f1,
                                                        const GridFunction& f2 ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceLpNorm( const GridFunction& f1,
                                                        const GridFunction& f2,
                                                        const typename GridFunction::RealType& p ) const;

   //! Method for saving the object to a file as a binary data
   bool save( File& file ) const;

   //! Method for restoring the object from a file
   bool load( File& file );

   bool save( const String& fileName ) const;

   bool load( const String& fileName );

   bool writeMesh( const String& fileName,
                   const String& format ) const;

   template< typename MeshFunction >
   bool write( const MeshFunction& function,
               const String& fileName,
               const String& format ) const;

   void writeProlog( tnlLogger& logger );

   protected:

   __cuda_callable__
   void computeSpaceSteps();

   CoordinatesType dimensions;
   
   IndexType numberOfCells, numberOfNxFaces, numberOfNyFaces, numberOfFaces, numberOfVertices;

   PointType origin, proportions;
   
   PointType spaceSteps;
   
   RealType spaceStepsProducts[ 5 ][ 5 ];
  
   template< typename, typename, int > 
   friend class GridEntityGetter;
};


#include <fstream>
#include <iomanip>
#include <core/tnlTNL_ASSERT.h>
#include <mesh/GnuplotWriter.h>
#include <mesh/grids/GridEntityGetter_impl.h>
#include <mesh/grids/NeighborGridEntityGetter2D_impl.h>
#include <mesh/grids/GridEntityMeasureGetter.h>

using namespace std;

template< typename Real,
          typename Device,
          typename Index >
Meshes::Grid< 2, Real, Device, Index > :: Grid()
: numberOfCells( 0 ),
  numberOfNxFaces( 0 ),
  numberOfNyFaces( 0 ),
  numberOfFaces( 0 ),
  numberOfVertices( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index >
String Meshes::Grid< 2, Real, Device, Index > :: getSerializationType()
{
   return String( "Meshes::Grid< " ) +
          convertToString( getMeshDimension() ) + ", " +
          getType< RealType >() + ", " +
          getType< Devices::Host >() + ", " +
          getType< IndexType >() + " >";
};

template< typename Real,
          typename Device,
          typename Index >
String Meshes::Grid< 2, Real, Device, Index > :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
void Meshes::Grid< 2, Real, Device, Index > :: computeSpaceSteps()
{
   if( this->getDimensions().x() > 0 && this->getDimensions().y() > 0 )
   {
      this->spaceSteps.x() = this->proportions.x() / ( Real ) this->getDimensions().x();
      this->spaceSteps.y() = this->proportions.y() / ( Real ) this->getDimensions().y();
      const RealType& hx = this->spaceSteps.x(); 
      const RealType& hy = this->spaceSteps.y();
      
      Real auxX, auxY;
      for( int i = 0; i < 5; i++ )
      {
         switch( i )
         {
            case 0:
               auxX = 1.0 / ( hx * hx );
               break;
            case 1:
               auxX = 1.0 / hx;
               break;
            case 2:
               auxX = 1.0;
               break;
            case 3:
               auxX = hx;
               break;
            case 4:
               auxX = hx * hx;
               break;
         }
         for( int j = 0; j < 5; j++ )
         {
            switch( j )
            {
               case 0:
                  auxY = 1.0 / ( hy * hy );
                  break;
               case 1:
                  auxY = 1.0 / hy;
                  break;
               case 2:
                  auxY = 1.0;
                  break;
               case 3:
                  auxY = hy;
                  break;
               case 4:
                  auxY = hy * hy;
                  break;
            }
            this->spaceStepsProducts[ i ][ j ] = auxX * auxY;         
         }
      }
   }
}

template< typename Real,
          typename Device,
          typename Index >
void Meshes::Grid< 2, Real, Device, Index > :: setDimensions( const Index xSize, const Index ySize )
{
   tnlTNL_ASSERT( xSize > 0,std::cerr << "xSize = " << xSize );
   tnlTNL_ASSERT( ySize > 0,std::cerr << "ySize = " << ySize );

   this->dimensions.x() = xSize;
   this->dimensions.y() = ySize;
   this->numberOfCells = xSize * ySize;
   this->numberOfNxFaces = ySize * ( xSize + 1 );
   this->numberOfNyFaces = xSize * ( ySize + 1 );
   this->numberOfFaces = this->numberOfNxFaces + this->numberOfNyFaces;
   this->numberOfVertices = ( xSize + 1 ) * ( ySize + 1 );
   computeSpaceSteps();
}

template< typename Real,
          typename Device,
          typename Index >
void Meshes::Grid< 2, Real, Device, Index > :: setDimensions( const CoordinatesType& dimensions )
{
   return this->setDimensions( dimensions. x(), dimensions. y() );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const typename Meshes::Grid< 2, Real, Device, Index >::CoordinatesType&
Meshes::Grid< 2, Real, Device, Index > :: getDimensions() const
{
   return this->dimensions;
}

template< typename Real,
          typename Device,
          typename Index >
void Meshes::Grid< 2, Real, Device, Index > :: setDomain( const PointType& origin,
                                                     const PointType& proportions )
{
   this->origin = origin;
   this->proportions = proportions;
   computeSpaceSteps();
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline 
const typename Meshes::Grid< 2, Real, Device, Index >::PointType&
Meshes::Grid< 2, Real, Device, Index >::getOrigin() const
{
   return this->origin;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline 
const typename Meshes::Grid< 2, Real, Device, Index > :: PointType&
   Meshes::Grid< 2, Real, Device, Index > :: getProportions() const
{
   return this->proportions;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityType >
__cuda_callable__ inline 
Index
Meshes::Grid< 2, Real, Device, Index >:: 
getEntitiesCount() const
{
   static_assert( EntityType::entityDimension <= 2 &&
                  EntityType::entityDimension >= 0, "Wrong grid entity dimension." );
   
   switch( EntityType::entityDimension )
   {
      case 2:
         return this->numberOfCells;
      case 1:
         return this->numberOfFaces;         
      case 0:
         return this->numberOfVertices;
   }            
   return -1;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityType >
__cuda_callable__ inline 
EntityType
Meshes::Grid< 2, Real, Device, Index >::
getEntity( const IndexType& entityIndex ) const
{
   static_assert( EntityType::entityDimension <= 2 &&
                  EntityType::entityDimension >= 0, "Wrong grid entity dimension." );
   
   return GridEntityGetter< ThisType, EntityType >::getEntity( *this, entityIndex );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityType >
__cuda_callable__ inline 
Index
Meshes::Grid< 2, Real, Device, Index >::
getEntityIndex( const EntityType& entity ) const
{
   static_assert( EntityType::entityDimension <= 2 &&
                  EntityType::entityDimension >= 0, "Wrong grid entity dimension." );
   
   return GridEntityGetter< ThisType, EntityType >::getEntityIndex( *this, entity );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityType >
__cuda_callable__
Real
Meshes::Grid< 2, Real, Device, Index >::
getEntityMeasure( const EntityType& entity ) const
{
   return GridEntityMeasureGetter< ThisType, EntityType::getDimensions() >::getMeasure( *this, entity );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Real
Meshes::Grid< 2, Real, Device, Index >::
getCellMeasure() const
{
   return this->template getSpaceStepsProducts< 1, 1 >();
}


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
typename Meshes::Grid< 2, Real, Device, Index >::PointType
Meshes::Grid< 2, Real, Device, Index >::
getSpaceSteps() const
{
   return this->spaceSteps;
}

template< typename Real,
          typename Device,
          typename Index >
   template< int xPow, int yPow  >
__cuda_callable__ inline 
const Real& 
Meshes::Grid< 2, Real, Device, Index >::
getSpaceStepsProducts() const
{
   tnlTNL_ASSERT( xPow >= -2 && xPow <= 2, 
             std::cerr << " xPow = " << xPow );
   tnlTNL_ASSERT( yPow >= -2 && yPow <= 2, 
             std::cerr << " yPow = " << yPow );

   return this->spaceStepsProducts[ yPow + 2 ][ xPow + 2 ];
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
Real Meshes::Grid< 2, Real, Device, Index > :: getSmallestSpaceStep() const
{
   return Min( this->spaceSteps.x(), this->spaceSteps.y() );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
      typename GridFunction::RealType
         Meshes::Grid< 2, Real, Device, Index >::getAbsMax( const GridFunction& f ) const
{
   return f.absMax();
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
      typename GridFunction::RealType
         Meshes::Grid< 2, Real, Device, Index >::getLpNorm( const GridFunction& f1,
                                                                 const typename GridFunction::RealType& p ) const
{
   typename GridFunction::RealType lpNorm( 0.0 );
   Cell cell( *this );
   for( cell.getCoordinates().y() = 0;
        cell.getCoordinates().y() < getDimensions().y();
        cell.getCoordinates().y()++ )
      for( cell.getCoordinates().x() = 0;
           cell.getCoordinates().x() < getDimensions().x();
           cell.getCoordinates().x()++ )
      {
         IndexType c = this->getEntityIndex( cell );
         lpNorm += pow( tnlAbs( f1[ c ] ), p );
      }
   lpNorm *= this->getSpaceSteps().x() * this->getSpaceSteps().y();
   return pow( lpNorm, 1.0/p );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
      typename GridFunction::RealType
         Meshes::Grid< 2, Real, Device, Index >::getDifferenceAbsMax( const GridFunction& f1,
                                                                 const GridFunction& f2 ) const
{
   return f1.differenceAbsMax( f2 );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
      typename GridFunction::RealType
         Meshes::Grid< 2, Real, Device, Index >::getDifferenceLpNorm( const GridFunction& f1,
                                                                 const GridFunction& f2,
                                                                 const typename GridFunction::RealType& p ) const
{
   typename GridFunction::RealType lpNorm( 0.0 );
   Cell cell( *this );
   for( cell.getCoordinates().y() = 0;
        cell.getCoordinates().y() < getDimensions().y();
        cell.getCoordinates().y()++ )
      for( cell.getCoordinates().x() = 0;
           cell.getCoordinates().x() < getDimensions().x();
           cell.getCoordinates().x()++ )
      {
         IndexType c = this->getEntityIndex( cell );
         lpNorm += pow( tnlAbs( f1[ c ] - f2[ c ] ), p );
      }
   lpNorm *= this->getSpaceSteps().x() * this->getSpaceSteps().y();
   return pow( lpNorm, 1.0 / p );
}

template< typename Real,
          typename Device,
          typename Index >
bool Meshes::Grid< 2, Real, Device, Index > :: save( File& file ) const
{
   if( ! tnlObject::save( file ) )
      return false;
   if( ! this->origin.save( file ) ||
       ! this->proportions.save( file ) ||
       ! this->dimensions.save( file ) )
   {
     std::cerr << "I was not able to save the domain description of a Grid." <<std::endl;
      return false;
   }
   return true;
};

template< typename Real,
          typename Device,
          typename Index >
bool Meshes::Grid< 2, Real, Device, Index > :: load( File& file )
{
   if( ! tnlObject::load( file ) )
      return false;
   CoordinatesType dimensions;
   if( ! this->origin.load( file ) ||
       ! this->proportions.load( file ) ||
       ! dimensions.load( file ) )
   {
     std::cerr << "I was not able to load the domain description of a Grid." <<std::endl;
      return false;
   }
   this->setDimensions( dimensions );
   return true;
};

template< typename Real,
          typename Device,
          typename Index >
bool Meshes::Grid< 2, Real, Device, Index > :: save( const String& fileName ) const
{
   return tnlObject :: save( fileName );
};

template< typename Real,
           typename Device,
           typename Index >
bool Meshes::Grid< 2, Real, Device, Index > :: load( const String& fileName )
{
   return tnlObject :: load( fileName );
};

template< typename Real,
          typename Device,
          typename Index >
bool Meshes::Grid< 2, Real, Device, Index > :: writeMesh( const String& fileName,
                                                     const String& format ) const
{
   fstream file;
   file. open( fileName. getString(), ios :: out );
   if( ! file )
   {
     std::cerr << "I am not able to open the file " << fileName << "." <<std::endl;
      return false;
   }
   if( format == "asymptote" )
   {
      file << "size( "
           << this->getProportions(). x() << "cm , "
           << this->getProportions(). y() << "cm );"
           <<std::endl <<std::endl;
      MeshEntity< 0 > vertex( *this );
      CoordinatesType& vertexCoordinates = vertex.getCoordinates();
      PointType v;
      for( Index j = 0; j < this->dimensions. y(); j ++ )
      {
         file << "draw( ";
         vertexCoordinates.x() = 0;
         vertexCoordinates.y() = j;
         v = vertex.getCenter();
         file << "( " << v. x() << ", " << v. y() << " )";
         for( Index i = 0; i < this->dimensions. x(); i ++ )
         {
            vertexCoordinates.x() = i + 1;
            vertexCoordinates.y() = j;
            v = vertex.getCenter();
            file << "--( " << v. x() << ", " << v. y() << " )";
         }
         file << " );" <<std::endl;
      }
      file <<std::endl;
      for( Index i = 0; i < this->dimensions. x(); i ++ )
      {
         file << "draw( ";
         vertexCoordinates.x() = i;
         vertexCoordinates.y() = 0;
         v = vertex.getCenter();
         file << "( " << v. x() << ", " << v. y() << " )";
         for( Index j = 0; j < this->dimensions. y(); j ++ )
         {
            vertexCoordinates.x() = i;
            vertexCoordinates.y() = j + 1;
            v = vertex.getCenter();
            file << "--( " << v. x() << ", " << v. y() << " )";
         }
         file << " );" <<std::endl;
      }
      file <<std::endl;

      MeshEntity< 2 > cell( *this );
      CoordinatesType& cellCoordinates = cell.getCoordinates();
      const RealType cellMeasure = this->getSpaceSteps().x() * this->getSpaceSteps().y();
      for( Index i = 0; i < this->dimensions. x(); i ++ )
         for( Index j = 0; j < this->dimensions. y(); j ++ )
         {
            cellCoordinates.x() = i;
            cellCoordinates.y() = j;
            v = vertex.getCenter();
            file << "label( scale(0.33) * Label( \"$" << setprecision( 3 ) << cellMeasure << setprecision( 8 )
                 << "$\" ), ( " << v. x() << ", " << v. y() << " ), S );" <<std::endl;
         }

      for( Index i = 0; i < this->dimensions. x(); i ++ )
         for( Index j = 0; j < this->dimensions. y(); j ++ )
         {
            PointType v1, v2, c;

            /****
             * East edge normal
             */
            /*v1 = this->getPoint( CoordinatesType( i + 1, j ), v1 );
            v2 = this->getPoint( CoordinatesType( i + 1, j + 1 ), v2 );
            c = ( ( Real ) 0.5 ) * ( v1 + v2 );
            this->getEdgeNormal< 1, 0 >( CoordinatesType( i, j ), v );
            v *= 0.5;
            file << "draw( ( " << c. x() << ", " << c. y() << " )--( "
                 << c. x() + v. x() << ", " << c.y() + v. y() << " ), Arrow(size=1mm),p=green);" <<std::endl;
            */
            /****
             * West edge normal
             */
            /*this->getPoint< -1, -1 >( CoordinatesType( i, j ), v1 );
            this->getPoint< -1, 1 >( CoordinatesType( i, j ), v2 );
            c = ( ( Real ) 0.5 ) * ( v1 + v2 );
            this->getEdgeNormal< -1, 0 >( CoordinatesType( i, j ), v );
            v *= 0.5;
            file << "draw( ( " << c. x() << ", " << c. y() << " )--( "
                 << c. x() + v. x() << ", " << c.y() + v. y() << " ), Arrow(size=1mm),p=blue);" <<std::endl;
            */
            /****
             * North edge normal
             */
            /*this->getPoint< 1, 1 >( CoordinatesType( i, j ), v1 );
            this->getPoint< -1, 1 >( CoordinatesType( i, j ), v2 );
            c = ( ( Real ) 0.5 ) * ( v1 + v2 );
            this->getEdgeNormal< 0, 1 >( CoordinatesType( i, j ), v );
            v *= 0.5;
            file << "draw( ( " << c. x() << ", " << c. y() << " )--( "
                 << c. x() + v. x() << ", " << c.y() + v. y() << " ), Arrow(size=1mm),p=green);" <<std::endl;
            */
            /****
             * South edge normal
             */
            /*this->getPoint< 1, -1 >( CoordinatesType( i, j ), v1 );
            this->getPoint< -1, -1 >( CoordinatesType( i, j ), v2 );
            c = ( ( Real ) 0.5 ) * ( v1 + v2 );
            this->getEdgeNormal< 0, -1 >( CoordinatesType( i, j ), v );
            v *= 0.5;
            file << "draw( ( " << c. x() << ", " << c. y() << " )--( "
                 << c. x() + v. x() << ", " << c.y() + v. y() << " ), Arrow(size=1mm),p=blue);" <<std::endl;
            */
         }
      return true;
   }
   return false;
}

template< typename Real,
           typename Device,
           typename Index >
   template< typename MeshFunction >
bool Meshes::Grid< 2, Real, Device, Index > :: write( const MeshFunction& function,
                                                 const String& fileName,
                                                 const String& format ) const
{
   if( this->template getEntitiesCount< Cell >() != function. getSize() )
   {
     std::cerr << "The size ( " << function. getSize() 
           << " ) of a mesh function does not agree with the DOFs ( " 
           << this->template getEntitiesCount< Cell >() << " ) of a mesh." <<std::endl;
      return false;
   }
   fstream file;
   file. open( fileName. getString(), ios :: out );
   if( ! file )
   {
     std::cerr << "I am not able to open the file " << fileName << "." <<std::endl;
      return false;
   }
   file << setprecision( 12 );
   if( format == "gnuplot" )
   {
      Cell cell( *this );
      CoordinatesType& cellCoordinates = cell.getCoordinates();
      for( cellCoordinates.y() = 0; cellCoordinates.y() < getDimensions(). y(); cellCoordinates.y() ++ )
      {
         for( cellCoordinates.x() = 0; cellCoordinates.x() < getDimensions(). x(); cellCoordinates.x() ++ )
         {
            PointType v = cell.getCenter();
            GnuplotWriter::write( file,  v );
            GnuplotWriter::write( file,  function[ this->getEntityIndex( cell ) ] );
            file <<std::endl;
         }
         file <<std::endl;
      }
   }
   file. close();
   return true;
}

template< typename Real,
           typename Device,
           typename Index >
void
Meshes::Grid< 2, Real, Device, Index >::
writeProlog( tnlLogger& logger )
{
   logger.writeParameter( "Dimension:", getMeshDimension() );
   logger.writeParameter( "Domain origin:", this->origin );
   logger.writeParameter( "Domain proportions:", this->proportions );
   logger.writeParameter( "Domain dimensions:", this->dimensions );
   logger.writeParameter( "Space steps:", this->getSpaceSteps() );
}









#ifdef UNDEF

template< int Dimension,
          typename Real = double,
          typename Device = Devices::Host,
          typename Index = int >
class Grid : public tnlObject
{
};


#include <mesh/grids/GridEntityGetter.h>
#include <mesh/grids/GridEntityMeasureGetter.h>
#include <core/tnlLogger.h>
#include <mesh/GnuplotWriter.h>
#include <mesh/grids/GridEntityConfig.h>

template< typename Real,
          typename Device,
          typename Index >
class Meshes::Grid< 2, Real, Device, Index > : public tnlObject
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef Containers::StaticVector< 2, Real > PointType;
   typedef Containers::StaticVector< 2, Index > CoordinatesType;
   
   static const int meshDimension = 2;

   /*template< int EntityDimension, 
             typename Config = GridEntityNoStencilStorage >//CrossStencilStorage< 1 > >
   using MeshEntity = GridEntity< ThisType, EntityDimension, Config >;
   
   typedef MeshEntity< meshDimension, GridEntityCrossStencilStorage< 1 > > Cell;
   typedef MeshEntity< meshDimension - 1, GridEntityNoStencilStorage > Face;
   typedef MeshEntity< 0 > Point;*/
   

   // TODO: remove this
   template< int EntityDimension, 
             typename Config = GridEntityNoStencilStorage >//CrossStencilStorage< 1 > >
   using TestMeshEntity = tnlTestGridEntity< ThisType, EntityDimension, Config >;
   typedef TestMeshEntity< meshDimension, GridEntityCrossStencilStorage< 1 > > Cell;
   /////
   
   static constexpr int getMeshDimension() { return meshDimension; };

   Grid();

   static String getSerializationType();

   virtual String getSerializationTypeVirtual() const;

   void setDimensions( const Index xSize, const Index ySize );

   void setDimensions( const CoordinatesType& dimensions );

   __cuda_callable__
   inline const CoordinatesType& getDimensions() const;

   void setDomain( const PointType& origin,
                   const PointType& proportions );
   __cuda_callable__
   inline const PointType& getOrigin() const;

   __cuda_callable__
   inline const PointType& getProportions() const;

   template< typename EntityType >
   __cuda_callable__
   inline IndexType getEntitiesCount() const;
   
   template< typename EntityType >
   __cuda_callable__
   inline EntityType getEntity( const IndexType& entityIndex ) const;
   
   template< typename EntityType >
   __cuda_callable__
   inline Index getEntityIndex( const EntityType& entity ) const;

   template< typename EntityType >
   __cuda_callable__
   RealType getEntityMeasure( const EntityType& entity ) const;
      
   __cuda_callable__
   RealType getCellMeasure() const;
   
   __cuda_callable__
   inline PointType getSpaceSteps() const;

   template< int xPow, int yPow >
   __cuda_callable__
   inline const RealType& getSpaceStepsProducts() const;
   
   __cuda_callable__
   inline RealType getSmallestSpaceStep() const;

   
   template< typename GridFunction >
   typename GridFunction::RealType getAbsMax( const GridFunction& f ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getLpNorm( const GridFunction& f,
                                              const typename GridFunction::RealType& p ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceAbsMax( const GridFunction& f1,
                                                        const GridFunction& f2 ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceLpNorm( const GridFunction& f1,
                                                        const GridFunction& f2,
                                                        const typename GridFunction::RealType& p ) const;

   //! Method for saving the object to a file as a binary data
   bool save( File& file ) const;

   //! Method for restoring the object from a file
   bool load( File& file );

   bool save( const String& fileName ) const;

   bool load( const String& fileName );

   bool writeMesh( const String& fileName,
                   const String& format ) const;

   template< typename MeshFunction >
   bool write( const MeshFunction& function,
               const String& fileName,
               const String& format ) const;

   void writeProlog( tnlLogger& logger );

   protected:

   __cuda_callable__
   void computeSpaceSteps();

   CoordinatesType dimensions;
   
   IndexType numberOfCells, numberOfNxFaces, numberOfNyFaces, numberOfFaces, numberOfVertices;

   PointType origin, proportions;
   
   PointType spaceSteps;
   
   RealType spaceStepsProducts[ 5 ][ 5 ];
  
   template< typename, typename, int > 
   friend class GridEntityGetter;
};

template< typename Real,
          typename Device,
          typename Index >
Meshes::Grid< 2, Real, Device, Index > :: Grid()
: numberOfCells( 0 ),
  numberOfNxFaces( 0 ),
  numberOfNyFaces( 0 ),
  numberOfFaces( 0 ),
  numberOfVertices( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index >
String Meshes::Grid< 2, Real, Device, Index > :: getSerializationType()
{
   return String( "Meshes::Grid< " ) +
          convertToString( getMeshDimension() ) + ", " +
          getType< RealType >() + ", " +
          getType< Devices::Host >() + ", " +
          getType< IndexType >() + " >";
};

template< typename Real,
          typename Device,
          typename Index >
String Meshes::Grid< 2, Real, Device, Index > :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
void Meshes::Grid< 2, Real, Device, Index > :: computeSpaceSteps()
{
   if( this->getDimensions().x() > 0 && this->getDimensions().y() > 0 )
   {
      this->spaceSteps.x() = this->proportions.x() / ( Real ) this->getDimensions().x();
      this->spaceSteps.y() = this->proportions.y() / ( Real ) this->getDimensions().y();
      const RealType& hx = this->spaceSteps.x(); 
      const RealType& hy = this->spaceSteps.y();
      
      Real auxX, auxY;
      for( int i = 0; i < 5; i++ )
      {
         switch( i )
         {
            case 0:
               auxX = 1.0 / ( hx * hx );
               break;
            case 1:
               auxX = 1.0 / hx;
               break;
            case 2:
               auxX = 1.0;
               break;
            case 3:
               auxX = hx;
               break;
            case 4:
               auxX = hx * hx;
               break;
         }
         for( int j = 0; j < 5; j++ )
         {
            switch( j )
            {
               case 0:
                  auxY = 1.0 / ( hy * hy );
                  break;
               case 1:
                  auxY = 1.0 / hy;
                  break;
               case 2:
                  auxY = 1.0;
                  break;
               case 3:
                  auxY = hy;
                  break;
               case 4:
                  auxY = hy * hy;
                  break;
            }
            this->spaceStepsProducts[ i ][ j ] = auxX * auxY;         
         }
      }
   }
}

template< typename Real,
          typename Device,
          typename Index >
void Meshes::Grid< 2, Real, Device, Index > :: setDimensions( const Index xSize, const Index ySize )
{
   tnlTNL_ASSERT( xSize > 0,std::cerr << "xSize = " << xSize );
   tnlTNL_ASSERT( ySize > 0,std::cerr << "ySize = " << ySize );

   this->dimensions.x() = xSize;
   this->dimensions.y() = ySize;
   this->numberOfCells = xSize * ySize;
   this->numberOfNxFaces = ySize * ( xSize + 1 );
   this->numberOfNyFaces = xSize * ( ySize + 1 );
   this->numberOfFaces = this->numberOfNxFaces + this->numberOfNyFaces;
   this->numberOfVertices = ( xSize + 1 ) * ( ySize + 1 );
   computeSpaceSteps();
}

template< typename Real,
          typename Device,
          typename Index >
void Meshes::Grid< 2, Real, Device, Index > :: setDimensions( const CoordinatesType& dimensions )
{
   return this->setDimensions( dimensions. x(), dimensions. y() );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const typename Meshes::Grid< 2, Real, Device, Index >::CoordinatesType&
Meshes::Grid< 2, Real, Device, Index > :: getDimensions() const
{
   return this->dimensions;
}

template< typename Real,
          typename Device,
          typename Index >
void Meshes::Grid< 2, Real, Device, Index > :: setDomain( const PointType& origin,
                                                     const PointType& proportions )
{
   this->origin = origin;
   this->proportions = proportions;
   computeSpaceSteps();
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline 
const typename Meshes::Grid< 2, Real, Device, Index >::PointType&
Meshes::Grid< 2, Real, Device, Index >::getOrigin() const
{
   return this->origin;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline 
const typename Meshes::Grid< 2, Real, Device, Index > :: PointType&
   Meshes::Grid< 2, Real, Device, Index > :: getProportions() const
{
   return this->proportions;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityType >
__cuda_callable__ inline 
Index
Meshes::Grid< 2, Real, Device, Index >:: 
getEntitiesCount() const
{
   static_assert( EntityType::entityDimension <= 2 &&
                  EntityType::entityDimension >= 0, "Wrong grid entity dimension." );
   
   switch( EntityType::entityDimension )
   {
      case 2:
         return this->numberOfCells;
      case 1:
         return this->numberOfFaces;         
      case 0:
         return this->numberOfVertices;
   }            
   return -1;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityType >
__cuda_callable__ inline 
EntityType
Meshes::Grid< 2, Real, Device, Index >::
getEntity( const IndexType& entityIndex ) const
{
   static_assert( EntityType::entityDimension <= 2 &&
                  EntityType::entityDimension >= 0, "Wrong grid entity dimension." );
   
   return GridEntityGetter< ThisType, EntityType >::getEntity( *this, entityIndex );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityType >
__cuda_callable__ inline 
Index
Meshes::Grid< 2, Real, Device, Index >::
getEntityIndex( const EntityType& entity ) const
{
   static_assert( EntityType::entityDimension <= 2 &&
                  EntityType::entityDimension >= 0, "Wrong grid entity dimension." );
   
   return GridEntityGetter< ThisType, EntityType >::getEntityIndex( *this, entity );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityType >
__cuda_callable__
Real
Meshes::Grid< 2, Real, Device, Index >::
getEntityMeasure( const EntityType& entity ) const
{
   return GridEntityMeasureGetter< ThisType, EntityType::getDimensions() >::getMeasure( *this, entity );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Real
Meshes::Grid< 2, Real, Device, Index >::
getCellMeasure() const
{
   return this->template getSpaceStepsProducts< 1, 1 >();
}


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
typename Meshes::Grid< 2, Real, Device, Index >::PointType
Meshes::Grid< 2, Real, Device, Index >::
getSpaceSteps() const
{
   return this->spaceSteps;
}

template< typename Real,
          typename Device,
          typename Index >
   template< int xPow, int yPow  >
__cuda_callable__ inline 
const Real& 
Meshes::Grid< 2, Real, Device, Index >::
getSpaceStepsProducts() const
{
   tnlTNL_ASSERT( xPow >= -2 && xPow <= 2, 
             std::cerr << " xPow = " << xPow );
   tnlTNL_ASSERT( yPow >= -2 && yPow <= 2, 
             std::cerr << " yPow = " << yPow );

   return this->spaceStepsProducts[ yPow + 2 ][ xPow + 2 ];
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
Real Meshes::Grid< 2, Real, Device, Index > :: getSmallestSpaceStep() const
{
   return Min( this->spaceSteps.x(), this->spaceSteps.y() );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
      typename GridFunction::RealType
         Meshes::Grid< 2, Real, Device, Index >::getAbsMax( const GridFunction& f ) const
{
   return f.absMax();
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
      typename GridFunction::RealType
         Meshes::Grid< 2, Real, Device, Index >::getLpNorm( const GridFunction& f1,
                                                                 const typename GridFunction::RealType& p ) const
{
   typename GridFunction::RealType lpNorm( 0.0 );
   Cell cell( *this );
   for( cell.getCoordinates().y() = 0;
        cell.getCoordinates().y() < getDimensions().y();
        cell.getCoordinates().y()++ )
      for( cell.getCoordinates().x() = 0;
           cell.getCoordinates().x() < getDimensions().x();
           cell.getCoordinates().x()++ )
      {
         IndexType c = this->getEntityIndex( cell );
         lpNorm += pow( tnlAbs( f1[ c ] ), p );
      }
   lpNorm *= this->getSpaceSteps().x() * this->getSpaceSteps().y();
   return pow( lpNorm, 1.0/p );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
      typename GridFunction::RealType
         Meshes::Grid< 2, Real, Device, Index >::getDifferenceAbsMax( const GridFunction& f1,
                                                                 const GridFunction& f2 ) const
{
   return f1.differenceAbsMax( f2 );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
      typename GridFunction::RealType
         Meshes::Grid< 2, Real, Device, Index >::getDifferenceLpNorm( const GridFunction& f1,
                                                                 const GridFunction& f2,
                                                                 const typename GridFunction::RealType& p ) const
{
   typename GridFunction::RealType lpNorm( 0.0 );
   Cell cell( *this );
   for( cell.getCoordinates().y() = 0;
        cell.getCoordinates().y() < getDimensions().y();
        cell.getCoordinates().y()++ )
      for( cell.getCoordinates().x() = 0;
           cell.getCoordinates().x() < getDimensions().x();
           cell.getCoordinates().x()++ )
      {
         IndexType c = this->getEntityIndex( cell );
         lpNorm += pow( tnlAbs( f1[ c ] - f2[ c ] ), p );
      }
   lpNorm *= this->getSpaceSteps().x() * this->getSpaceSteps().y();
   return pow( lpNorm, 1.0 / p );
}

template< typename Real,
          typename Device,
          typename Index >
bool Meshes::Grid< 2, Real, Device, Index > :: save( File& file ) const
{
   if( ! tnlObject::save( file ) )
      return false;
   if( ! this->origin.save( file ) ||
       ! this->proportions.save( file ) ||
       ! this->dimensions.save( file ) )
   {
     std::cerr << "I was not able to save the domain description of a Grid." <<std::endl;
      return false;
   }
   return true;
};

template< typename Real,
          typename Device,
          typename Index >
bool Meshes::Grid< 2, Real, Device, Index > :: load( File& file )
{
   if( ! tnlObject::load( file ) )
      return false;
   CoordinatesType dimensions;
   if( ! this->origin.load( file ) ||
       ! this->proportions.load( file ) ||
       ! dimensions.load( file ) )
   {
     std::cerr << "I was not able to load the domain description of a Grid." <<std::endl;
      return false;
   }
   this->setDimensions( dimensions );
   return true;
};

template< typename Real,
          typename Device,
          typename Index >
bool Meshes::Grid< 2, Real, Device, Index > :: save( const String& fileName ) const
{
   return tnlObject :: save( fileName );
};

template< typename Real,
           typename Device,
           typename Index >
bool Meshes::Grid< 2, Real, Device, Index > :: load( const String& fileName )
{
   return tnlObject :: load( fileName );
};

template< typename Real,
          typename Device,
          typename Index >
bool Meshes::Grid< 2, Real, Device, Index > :: writeMesh( const String& fileName,
                                                     const String& format ) const
{
   fstream file;
   file. open( fileName. getString(), ios :: out );
   if( ! file )
   {
     std::cerr << "I am not able to open the file " << fileName << "." <<std::endl;
      return false;
   }
   if( format == "asymptote" )
   {
      file << "size( "
           << this->getProportions(). x() << "cm , "
           << this->getProportions(). y() << "cm );"
           <<std::endl <<std::endl;
      TestMeshEntity< 0 > vertex( *this );
      CoordinatesType& vertexCoordinates = vertex.getCoordinates();
      PointType v;
      for( Index j = 0; j < this->dimensions. y(); j ++ )
      {
         file << "draw( ";
         vertexCoordinates.x() = 0;
         vertexCoordinates.y() = j;
         v = vertex.getCenter();
         file << "( " << v. x() << ", " << v. y() << " )";
         for( Index i = 0; i < this->dimensions. x(); i ++ )
         {
            vertexCoordinates.x() = i + 1;
            vertexCoordinates.y() = j;
            v = vertex.getCenter();
            file << "--( " << v. x() << ", " << v. y() << " )";
         }
         file << " );" <<std::endl;
      }
      file <<std::endl;
      for( Index i = 0; i < this->dimensions. x(); i ++ )
      {
         file << "draw( ";
         vertexCoordinates.x() = i;
         vertexCoordinates.y() = 0;
         v = vertex.getCenter();
         file << "( " << v. x() << ", " << v. y() << " )";
         for( Index j = 0; j < this->dimensions. y(); j ++ )
         {
            vertexCoordinates.x() = i;
            vertexCoordinates.y() = j + 1;
            v = vertex.getCenter();
            file << "--( " << v. x() << ", " << v. y() << " )";
         }
         file << " );" <<std::endl;
      }
      file <<std::endl;

      TestMeshEntity< 2 > cell( *this );
      CoordinatesType& cellCoordinates = cell.getCoordinates();
      const RealType cellMeasure = this->getSpaceSteps().x() * this->getSpaceSteps().y();
      for( Index i = 0; i < this->dimensions. x(); i ++ )
         for( Index j = 0; j < this->dimensions. y(); j ++ )
         {
            cellCoordinates.x() = i;
            cellCoordinates.y() = j;
            v = vertex.getCenter();
            file << "label( scale(0.33) * Label( \"$" << setprecision( 3 ) << cellMeasure << setprecision( 8 )
                 << "$\" ), ( " << v. x() << ", " << v. y() << " ), S );" <<std::endl;
         }

      for( Index i = 0; i < this->dimensions. x(); i ++ )
         for( Index j = 0; j < this->dimensions. y(); j ++ )
         {
            PointType v1, v2, c;

            /****
             * East edge normal
             */
            /*v1 = this->getPoint( CoordinatesType( i + 1, j ), v1 );
            v2 = this->getPoint( CoordinatesType( i + 1, j + 1 ), v2 );
            c = ( ( Real ) 0.5 ) * ( v1 + v2 );
            this->getEdgeNormal< 1, 0 >( CoordinatesType( i, j ), v );
            v *= 0.5;
            file << "draw( ( " << c. x() << ", " << c. y() << " )--( "
                 << c. x() + v. x() << ", " << c.y() + v. y() << " ), Arrow(size=1mm),p=green);" <<std::endl;
            */
            /****
             * West edge normal
             */
            /*this->getPoint< -1, -1 >( CoordinatesType( i, j ), v1 );
            this->getPoint< -1, 1 >( CoordinatesType( i, j ), v2 );
            c = ( ( Real ) 0.5 ) * ( v1 + v2 );
            this->getEdgeNormal< -1, 0 >( CoordinatesType( i, j ), v );
            v *= 0.5;
            file << "draw( ( " << c. x() << ", " << c. y() << " )--( "
                 << c. x() + v. x() << ", " << c.y() + v. y() << " ), Arrow(size=1mm),p=blue);" <<std::endl;
            */
            /****
             * North edge normal
             */
            /*this->getPoint< 1, 1 >( CoordinatesType( i, j ), v1 );
            this->getPoint< -1, 1 >( CoordinatesType( i, j ), v2 );
            c = ( ( Real ) 0.5 ) * ( v1 + v2 );
            this->getEdgeNormal< 0, 1 >( CoordinatesType( i, j ), v );
            v *= 0.5;
            file << "draw( ( " << c. x() << ", " << c. y() << " )--( "
                 << c. x() + v. x() << ", " << c.y() + v. y() << " ), Arrow(size=1mm),p=green);" <<std::endl;
            */
            /****
             * South edge normal
             */
            /*this->getPoint< 1, -1 >( CoordinatesType( i, j ), v1 );
            this->getPoint< -1, -1 >( CoordinatesType( i, j ), v2 );
            c = ( ( Real ) 0.5 ) * ( v1 + v2 );
            this->getEdgeNormal< 0, -1 >( CoordinatesType( i, j ), v );
            v *= 0.5;
            file << "draw( ( " << c. x() << ", " << c. y() << " )--( "
                 << c. x() + v. x() << ", " << c.y() + v. y() << " ), Arrow(size=1mm),p=blue);" <<std::endl;
            */
         }
      return true;
   }
   return false;
}

template< typename Real,
           typename Device,
           typename Index >
   template< typename MeshFunction >
bool Meshes::Grid< 2, Real, Device, Index > :: write( const MeshFunction& function,
                                                 const String& fileName,
                                                 const String& format ) const
{
   if( this->template getEntitiesCount< Cell >() != function. getSize() )
   {
      std::cerr << "The size ( " << function. getSize() 
           << " ) of a mesh function does not agree with the DOFs ( " 
           << this->template getEntitiesCount< Cell >() << " ) of a mesh." << std::endl;
      return false;
   }
   fstream file;
   file. open( fileName. getString(), ios :: out );
   if( ! file )
   {
     std::cerr << "I am not able to open the file " << fileName << "." <<std::endl;
      return false;
   }
   file << setprecision( 12 );
   if( format == "gnuplot" )
   {
      Cell cell( *this );
      CoordinatesType& cellCoordinates = cell.getCoordinates();
      for( cellCoordinates.y() = 0; cellCoordinates.y() < getDimensions(). y(); cellCoordinates.y() ++ )
      {
         for( cellCoordinates.x() = 0; cellCoordinates.x() < getDimensions(). x(); cellCoordinates.x() ++ )
         {
            PointType v = cell.getCenter();
            GnuplotWriter::write( file,  v );
            GnuplotWriter::write( file,  function[ this->getEntityIndex( cell ) ] );
            file <<std::endl;
         }
         file <<std::endl;
      }
   }
   file. close();
   return true;
}

template< typename Real,
           typename Device,
           typename Index >
void
Meshes::Grid< 2, Real, Device, Index >::
writeProlog( tnlLogger& logger )
{
   logger.writeParameter( "Dimension:", getMeshDimension() );
   logger.writeParameter( "Domain origin:", this->origin );
   logger.writeParameter( "Domain proportions:", this->proportions );
   logger.writeParameter( "Domain dimensions:", this->dimensions );
   logger.writeParameter( "Space steps:", this->getSpaceSteps() );
}
#endif




