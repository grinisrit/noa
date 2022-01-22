// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/String.h>

namespace noaTNL {

/**
 * \brief Helper class for the construction of file names based on name, index and extension.
 *
 * Optionally, the file name can also handle node ID for distributed systems.
 * 
 * The following example demonstrates the use of FileName.
 * 
 * \par Example
 * \include FileNameExample.cpp
 * \par Output
 * \include FileNameExample.out
 */
class FileName
{
   public:

      /**
       * \brief Basic constructor.
       *
       * Sets no file name base, index to zero and index digits count to five;
       */
      FileName();

      /**
       * \brief Constructor with file name base parameter.
       * 
       * The index is set to zero and index digits count to five.
       * 
       * @param fileNameBase File name base.
       */
      FileName( const String& fileNameBase );

      /**
       * \brief Constructor with file name base and file name extension.
       * 
       * The index is set to zero and index digits count to five.
       * 
       * @param fileNameBase File name base.
       * @param extension File name extension.
       */
      FileName( const String& fileNameBase,
                const String& extension );

      /**
       * \brief Sets the file name base.
       *
       * @param fileNameBase String that specifies the new file name base.
       */
      void setFileNameBase( const String& fileNameBase );

      /**
       *  \brief Sets the file name extension.
       *
       * @param extension A String that specifies the new extension of file without dot.
       */
      void setExtension( const String& extension );

      /**
       * \brief Sets index of the file name.
       *
       * @param index Index of the file name.
       */
      void setIndex( const size_t index );

      /**
       * \brief Sets number of digits for index of the file name.
       *
       * @param digitsCount Number of digits. It is 5 by default.
       */
      void setDigitsCount( const size_t digitsCount );

      /**
       * \brief Sets the distributed system node ID as integer, for example MPI process ID.
       * 
       * @param nodeId Node ID.
       * 
       * See the following example:
       * 
       * \par Example
       * \include FileNameExampleDistributedSystemNodeId.cpp
       * \par Output
       * \include FileNameExampleDistributedSystemNodeId.out
       */
      void setDistributedSystemNodeId( size_t nodeId );

      /**
       * \brief Sets the distributed system node ID in a form of Cartesian coordinates.
       * 
       * @tparam Coordinates Type of Cartesian coordinates. It is Containers::StaticVector usually.
       * @param nodeId Node ID in a form of Cartesian coordinates.
       * 
       * See the following example:
       * 
       * \par Example
       * \include FileNameExampleDistributedSystemNodeCoordinates.cpp
       * \par Output
       * \include FileNameExampleDistributedSystemNodeCoordinates.out
       * 
       */
      template< typename Coordinates >
      void setDistributedSystemNodeCoordinates( const Coordinates& nodeId );

      /**
       * \brief Resets the distributed system node ID.
       */
      void resetDistributedSystemNodeId();

      /**
       * \brief Returns complete file name.
       *
       * @return String with the complete file name.
       */
      String getFileName();

   protected:

      String fileNameBase, extension, distributedSystemNodeId;

      size_t index, digitsCount;
};

/**
 * \brief Returns extension of given file name, i.e. part after the last dot.
 * 
 * @param fileName Input file name.
 * 
 * @return Extension of the given file name.
 */
String getFileExtension( const String fileName );

/**
 * \brief Cuts off the file extension.
 * 
 * @param file_name Input file name.
 * @return String with the file name without extension.
 */
String removeFileNameExtension( String fileName );

} // namespace noaTNL

#include <noa/3rdparty/TNL/FileName.hpp>
