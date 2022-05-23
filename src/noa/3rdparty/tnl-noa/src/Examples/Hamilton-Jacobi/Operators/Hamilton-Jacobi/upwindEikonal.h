#pragma once 

#include <mesh/tnlGrid.h>
#include <functions/tnlFunctions.h>

template< typename Mesh,
		    typename Real,
		    typename Index >
class upwindEikonalScheme
{
};

/****
 * 1D scheme
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class upwindEikonalScheme< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef tnlGrid< 1, Real, Device, Index > MeshType;
      typedef TNL::Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
      typedef typename MeshType::CoordinatesType CoordinatesType;


      static String getType();

      template< typename PreimageFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const PreimageFunction& u,
                       const MeshEntity& entity,
                       const RealType& f ) const
      {
         const RealType& hx_inv = entity.getMesh().template getSpaceStepsProducts< -1 >();
         const typename MeshEntity::template NeighborEntities< 1 >& neighborEntities = entity.getNeighborEntities();

         const RealType& u_c = u[ entity.getIndex() ];
         const RealType& u_e = u[ neighborEntities.template getEntityIndex< 1 >() ];
         const RealType& u_w = u[ neighborEntities.template getEntityIndex< -1 >() ];

         if( f > 0.0 )
         {
            const RealType& xf = negativePart( ( u_e - u_c ) * hx_inv );
            const RealType& xb = positivePart( ( u_c - u_w ) * hx_inv );
            return sqrt( xf * xf + xb * xb );
         }
         else if( f < 0.0 )
         {
            const RealType& xf = negativePart( ( u_c - u_w ) * hx_inv );
            const RealType& xb = positivePart( ( u_e - u_c ) * hx_inv );
            return sqrt( xf * xf + xb * xb );
         }
         else
         {
            return 0.0;
         }
      };
};

/****
 * 2D scheme
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class upwindEikonalScheme< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef tnlGrid< 2, Real, Device, Index > MeshType;
      typedef TNL::Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
      typedef typename MeshType::CoordinatesType CoordinatesType;

      static String getType();

      template< typename PreimageFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const PreimageFunction& u,
                       const MeshEntity& entity,
                       const RealType& f ) const
      {   
         const RealType& hx_inv = entity.getMesh().template getSpaceStepsProducts< -1,  0 >();
         const RealType& hy_inv = entity.getMesh().template getSpaceStepsProducts<  0, -1 >();

         const typename MeshEntity::template NeighborEntities< 2 >& neighborEntities = entity.getNeighborEntities();   
         const RealType& u_c = u[ entity.getIndex() ];
         const RealType& u_e = u[ neighborEntities.template getEntityIndex<  1,  0 >() ];
         const RealType& u_w = u[ neighborEntities.template getEntityIndex< -1,  0 >() ];
         const RealType& u_n = u[ neighborEntities.template getEntityIndex<  0,  1 >() ];
         const RealType& u_s = u[ neighborEntities.template getEntityIndex<  0, -1 >() ];
         if( f > 0.0 )
         {
            const RealType xf = negativePart( ( u_e - u_c ) * hx_inv );
            const RealType xb = positivePart( ( u_c - u_w ) * hx_inv );
            const RealType yf = negativePart( ( u_n - u_c ) * hy_inv );
            const RealType yb = positivePart( ( u_c - u_s ) * hy_inv );
            return sqrt( xf * xf + xb * xb + yf * yf + yb * yb );
         }
         else if( f < 0.0 )
         {
            const RealType xf = positivePart( ( u_e - u_c ) * hx_inv );
            const RealType xb = negativePart( ( u_c - u_w ) * hx_inv );
            const RealType yf = positivePart( ( u_n - u_c ) * hy_inv );
            const RealType yb = negativePart( ( u_c - u_s ) * hy_inv );
            return sqrt( xf * xf + xb * xb + yf * yf + yb * yb );
         }
         else
            return 0.0;
      }
};

/****
 * 3D scheme
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class upwindEikonalScheme< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index >
{

   public:
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef tnlGrid< 3, Real, Device, Index > MeshType;
      typedef TNL::Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
      typedef typename MeshType::CoordinatesType CoordinatesType;


      static String getType();

      template< typename PreimageFunction, typename MeshEntity >
      __cuda_callable__
       Real operator()( const PreimageFunction& u,
                        const MeshEntity& entity,
                        const RealType& f ) const
      {
         const RealType& hx_inv = entity.getMesh().template getSpaceStepsProducts< -1,  0,  0 >();
         const RealType& hy_inv = entity.getMesh().template getSpaceStepsProducts<  0, -1,  0 >();
         const RealType& hz_inv = entity.getMesh().template getSpaceStepsProducts<  0,  0, -1 >();

         const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
         const RealType& u_c = u[ entity.getIndex() ];
         const RealType& u_e = u[ neighborEntities.template getEntityIndex<  1,  0,  0 >() ];
         const RealType& u_w = u[ neighborEntities.template getEntityIndex< -1,  0,  0 >() ];
         const RealType& u_n = u[ neighborEntities.template getEntityIndex<  0,  1,  0 >() ];
         const RealType& u_s = u[ neighborEntities.template getEntityIndex<  0, -1,  0 >() ];
         const RealType& u_t = u[ neighborEntities.template getEntityIndex<  0,  0,  1 >() ];
         const RealType& u_b = u[ neighborEntities.template getEntityIndex<  0,  0, -1 >() ];

         if( f > 0.0 )
         {
            const RealType xf = negativePart( ( u_e - u_c ) * hx_inv );
            const RealType xb = positivePart( ( u_c - u_w ) * hx_inv );
            const RealType yf = negativePart( ( u_n - u_c ) * hy_inv );
            const RealType yb = positivePart( ( u_c - u_s ) * hy_inv );
            const RealType zf = negativePart( ( u_t - u_c ) * hz_inv );
            const RealType zb = positivePart( ( u_c - u_b ) * hz_inv );
            return sqrt( xf * xf + xb * xb + yf * yf + yb * yb + zf * zf + zb * zb );
         }
         else if( f < 0.0 )
         {
            const RealType xf = positivePart( ( u_e - u_c ) * hx_inv );
            const RealType xb = negativePart( ( u_c - u_w ) * hx_inv );
            const RealType yf = positivePart( ( u_n - u_c ) * hy_inv );
            const RealType yb = negativePart( ( u_c - u_s ) * hy_inv );
            const RealType zf = positivePart( ( u_t - u_c ) * hz_inv );
            const RealType zb = negativePart( ( u_c - u_b ) * hz_inv );
            return sqrt( xf * xf + xb * xb + yf * yf + yb * yb + zf * zf + zb * zb );
         }
         else
            return 0.0;
      };
};