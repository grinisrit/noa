// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include <TNL/String.h>
#include <TNL/File.h>

/**
 * \brief The main TNL namespace.
 */
namespace TNL {

/**
 * \brief Basic class for majority of TNL objects like matrices, meshes, grids, solvers, etc..
 *
 * Objects like numerical meshes, matrices large vectors etc. are inherited by
 * this class.
 *
 * Since the virtual destructor is not defined as \ref __cuda_callable__,
 * objects inherited from Object should not be created in CUDA kernels.
 *
 * In addition to methods of this class, see the following related functions:
 *
 * \ref getObjectType
 *
 * \ref parseObjectType
 *
 */
class Object
{
   public:

      /**
       * \brief Static serialization type getter.
       *
       * Objects in TNL are saved as in a device independent manner. This method
       * is supposed to return the object type but with the device type replaced
       * by Devices::Host. For example \c Array< double, Devices::Cuda > is
       * saved as \c Array< double, Devices::Host >.
       */
      static String getSerializationType();

      /***
       * \brief Virtual serialization type getter.
       *
       * Objects in TNL are saved as in a device independent manner. This method
       * is supposed to return the object type but with the device type replaced
       * by Devices::Host. For example \c Array< double, Devices::Cuda > is
       * saved as \c Array< double, Devices::Host >.
       */
      virtual String getSerializationTypeVirtual() const;

      /**
       * \brief Method for saving the object to a file as a binary data.
       *
       * Throws \ref Exceptions::FileSerializationError if the object cannot be saved.
       *
       * \param file Name of file object.
       */
      virtual void save( File& file ) const;

      /**
       * \brief Method for restoring the object from a file.
       *
       * Throws \ref Exceptions::FileDeserializationError if the object cannot be loaded.
       *
       * \param file Name of file object.
       */
      virtual void load( File& file );

      /**
       * \brief Method for saving the object to a file as a binary data.
       *
       * Throws \ref Exceptions::FileSerializationError if the object cannot be saved.
       *
       * \param fileName String defining the name of a file.
       */
      void save( const String& fileName ) const;

      /**
       * \brief Method for restoring the object from a file.
       *
       * Throws \ref Exceptions::FileDeserializationError if the object cannot be loaded.
       *
       * \param fileName String defining the name of a file.
       */
      void load( const String& fileName );

      /**
       * \brief Destructor.
       *
       * Since it is not defined as \ref __cuda_callable__, objects inherited
       * from Object should not be created in CUDA kernels.
       */
      virtual ~Object() = default;
};

/**
 * \brief Extracts object type from a binary file.
 *
 * Throws \ref Exceptions::FileDeserializationError if the object type cannot be detected.
 *
 * @param file is file where the object is stored
 * @return string with the object type
 */
String getObjectType( File& file );

/**
 * \brief Does the same as \ref getObjectType but with a \e fileName parameter instead of file.
 *
 * Throws \ref Exceptions::FileDeserializationError if the object type cannot be detected.
 *
 * @param fileName name of a file where the object is stored
 * @return string with the object type
 */
String getObjectType( const String& fileName );

/**
 * \brief Parses the object type
 *
 * @param objectType is a string with the object type to be parsed.
 * @return vector of strings where the first one is the object type and the next
 * strings are the template parameters.
 *
 * \par Example
 * \include ParseObjectTypeExample.cpp
 * \par Output
 * \include ParseObjectTypeExample.out
 */
std::vector< String >
parseObjectType( const String& objectType );

/**
 * \brief Saves object type into a binary file.
 *
 * Throws \ref Exceptions::FileDeserializationError if the object type cannot be detected.
 *
 * @param file is the file where the object will be saved
 * @param type is the object type to be saved
 */
void saveObjectType( File& file, const String& type );

} // namespace TNL

#include <TNL/Object.hpp>
