// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Images/Image.h>

namespace TNL {
namespace Images {   

template< typename Index >
RegionOfInterest< Index >::
RegionOfInterest()
: top( -1 ), bottom( -1 ), left( -1 ), right( -1 )
{
}
 
template< typename Index >
bool
RegionOfInterest< Index >::
setup( const Config::ParameterContainer& parameters,
       const Image< Index >* image )
{
   const int roiTop    = parameters.getParameter< int >( "roi-top" );
   const int roiBottom = parameters.getParameter< int >( "roi-bottom" );
   const int roiRight  = parameters.getParameter< int >( "roi-right" );
   const int roiLeft   = parameters.getParameter< int >( "roi-left" );
 
   if( roiBottom < roiTop )
   {
      std::cerr << "Error: roi-bottom (" << roiBottom << ") is smaller than roi-top (" << roiTop << ")." << std::endl;
      return false;
   }
   if( roiRight < roiLeft )
   {
      std::cerr << "Error: roi-right (" << roiRight << ") is smaller than roi-left (" << roiLeft << ")." << std::endl;
      return false;
   }

   if( roiLeft == -1 )
        this->left = 0;
   else
   {
      if( roiLeft >= image->getWidth() )
      {
         std::cerr << "ROI left column is larger than image width ( " << image->getWidth() << ")." << std::endl;
         return false;
      }
      this->left = roiLeft;
   }
 
   if( roiRight == -1 )
      this->right = image->getWidth();
   else
   {
      if( roiRight >= image->getWidth() )
      {
         std::cerr << "ROI right column is larger than image width ( " << image->getWidth() << ")." << std::endl;
         return false;
      }
      this->right = roiRight;
   }
 
   if( roiTop == -1 )
      this->top = 0;
   else
   {
      if( roiTop >= image->getHeight() )
      {
         std::cerr << "ROI top line is larger than image height ( " << image->getHeight() << ")." << std::endl;
         return false;
      }
      this->top = roiTop;
   }
 
   if( roiBottom == -1 )
      this->bottom = image->getHeight();
   else
   {
      if( roiBottom >= image->getHeight() )
      {
         std::cerr << "ROI bottom line is larger than image height ( " << image->getHeight() << ")." << std::endl;
         return false;
      }
      this->bottom = roiBottom;
   }
   return true;
}

template< typename Index >
bool
RegionOfInterest< Index >::
check( const Image< Index >* image ) const
{
   if( top >= image->getHeight() ||
       bottom >= image->getHeight() ||
       left >= image->getWidth() ||
       right >= image->getWidth() )
      return false;
   return true;
}

template< typename Index >
Index
RegionOfInterest< Index >::
getTop() const
{
   return this->top;
}

template< typename Index >
Index
RegionOfInterest< Index >::
getBottom() const
{
   return this->bottom;
}

template< typename Index >
Index
RegionOfInterest< Index >::
getLeft() const
{
   return this->left;
}

template< typename Index >
Index
RegionOfInterest< Index >::
getRight() const
{
   return this->right;
}

template< typename Index >
Index
RegionOfInterest< Index >::
getWidth() const
{
   return this->right - this->left;
}

template< typename Index >
Index
RegionOfInterest< Index >::
getHeight() const
{
   return this->bottom - this->top;
}

template< typename Index >
   template< typename Grid >
bool
RegionOfInterest< Index >::
setGrid( Grid& grid,
         bool verbose )
{
    grid.setDimensions( this->getWidth(), this->getHeight() );
    typename Grid::PointType origin, proportions;
    origin.x() = 0.0;
    origin.y() = 0.0;
    proportions.x() = 1.0;
    proportions.y() = ( double ) grid.getDimensions().y() / ( double ) grid.getDimensions().x();
    grid.setDomain( origin, proportions );
    if( verbose )
    {
       std::cout << "Setting grid to dimensions " << grid.getDimensions() <<
                " and proportions " << grid.getProportions() << std::endl;
    }
    return true;
}


template< typename Index >
bool
RegionOfInterest< Index >::
isIn( const Index row, const Index column ) const
{
   if( row >= top && row < bottom &&
       column >= left && column < right )
      return true;
   return false;
}

} // namespace Images
} // namespace TNL