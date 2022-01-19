// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <fstream>
#include <type_traits>

#include <TNL/String.h>
#include <TNL/Allocators/Host.h>
#include <TNL/Allocators/Cuda.h>

namespace TNL {

/**
 * \brief This class serves for binary IO. It allows to do IO even for data allocated on GPU together with on-the-fly data type conversion.
 *
 * \par Example
 * \include FileExample.cpp
 * \par Output
 * \include FileExample.out
 */
class File
{
   public:

      /**
       * \brief Basic constructor.
       */
      File() = default;

      File( const File& ) = delete;

      File( File&& ) = default;

      File& operator=( const File& ) = delete;

      File& operator=( File&& ) = default;

      /**
       * \brief Constructor which opens given file.
       *
       * All parameters are passed to the \ref open method.
       */
      File( const String& fileName,
            std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out );

      /**
       * \brief Open given file.
       *
       * Opens file with given \e fileName in some \e mode from \ref std::ios_base::openmode.
       * Note that the file is always opened in binary mode, i.e. \ref std::ios_base::binary
       * is always added to \e mode.
       *
       * Throws \ref std::ios_base::failure on failure.
       *
       * \param fileName String which indicates file name.
       * \param mode Indicates in what mode the file will be opened - see \ref std::ios_base::openmode.
       */
      void open( const String& fileName,
                 std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out );

      /**
       * \brief Closes the file.
       *
       * Throws \ref std::ios_base::failure on failure.
       */
      void close();

      /**
       * \brief Returns name of the file.
       */
      const String& getFileName() const
      {
         return this->fileName;
      }

      /**
       * \brief Method for loading data from the file.
       *
       * The data will be stored in \e buffer which was allocated using the
       * allocator of type \e Allocator. The data type of the buffer is given
       * by the template parameter \e Type. The second template parameter
       * \e SourceType defines the type of data in the source file. If both
       * types are different, on-the-fly conversion takes place during the
       * data loading.
       *
       * Throws \ref std::ios_base::failure on failure.
       *
       * \tparam Type type of data to be loaded to the \e buffer.
       * \tparam SourceType type of data stored on the file,
       * \tparam Allocator type of the allocator which was used to allocate \e buffer.
       * \param buffer Pointer in memory where the elements are loaded and stored after reading.
       * \param elements number of elements to be loaded from the file.
       *
       * The following example shows how to load data directly to GPU.
       *
       * \par Example
       * \include FileExampleCuda.cpp
       * \par Output
       * \include FileExampleCuda.out
       * The following example shows how to do on-the-fly data conversion.
       *
       * \par Example
       * \include FileExampleSaveAndLoad.cpp
       * \par Output
       * \include FileExampleSaveAndLoad.out
       */
      template< typename Type, typename SourceType = Type, typename Allocator = Allocators::Host< Type > >
      void load( Type* buffer, std::streamsize elements = 1 );

      /**
       * \brief Method for saving data to the file.
       *
       * The data from the \e buffer (with type \e Type) which was allocated
       * using an allocator of type \e Allocator. \e TargetType defines as what
       * data type the buffer shall be saved. If the type is different from the
       * data type, on-the-fly data type conversion takes place during the data
       * saving.
       *
       * Throws \ref std::ios_base::failure on failure.
       *
       * \tparam Type type of data in the \e buffer.
       * \tparam TargetType tells as what type data the buffer shall be saved.
       * \tparam Allocator type of the allocator which was used to allocate \e buffer.
       * \tparam Index type of index by which the elements are indexed.
       * \param buffer buffer that is going to be saved to the file.
       * \param elements number of elements saved to the file.
       *
       * See \ref File::load for examples.
       */
      template< typename Type, typename TargetType = Type, typename Allocator = Allocators::Host< Type > >
      void save( const Type* buffer, std::streamsize elements = 1 );

   protected:
      // implementation for all allocators which allocate data accessible from host
      template< typename Type,
                typename SourceType,
                typename Allocator,
                typename = std::enable_if_t< ! std::is_same< Allocator, Allocators::Cuda< Type > >::value > >
      void load_impl( Type* buffer, std::streamsize elements );

      // implementation for \ref Allocators::Cuda
      template< typename Type,
                typename SourceType,
                typename Allocator,
                typename = std::enable_if_t< std::is_same< Allocator, Allocators::Cuda< Type > >::value >,
                typename = void >
      void load_impl( Type* buffer, std::streamsize elements );

      // implementation for all allocators which allocate data accessible from host
      template< typename Type,
                typename TargetType,
                typename Allocator,
                typename = std::enable_if_t< ! std::is_same< Allocator, Allocators::Cuda< Type > >::value > >
      void save_impl( const Type* buffer, std::streamsize elements );

      // implementation for \ref Allocators::Cuda
      template< typename Type,
                typename TargetType,
                typename Allocator,
                typename = std::enable_if_t< std::is_same< Allocator, Allocators::Cuda< Type > >::value >,
                typename = void >
      void save_impl( const Type* buffer, std::streamsize elements );

      std::fstream file;
      String fileName;
};

/**
 * \brief Returns true if the file exists and false otherwise.
 *
 * Finds out if the file \e fileName exists.
 * \param fileName Name of the file to check.
 * \return returns true if the file exists and false othervise
 */
bool fileExists( const String& fileName );

/**
 * \brief Serialization of strings
 */
File& operator<<( File& file, const std::string& str );

/**
 * \brief Deserialization of strings.
 */
File& operator>>( File& file, std::string& str );

} // namespace TNL

#include <TNL/File.hpp>
