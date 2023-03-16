// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Images/DicomHeader.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Images/DicomImageInfo.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Images/DicomPatientInfo.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Images/DicomSeriesInfo.h>

namespace noa::TNL {
namespace Images {

inline DicomHeader::DicomHeader()
{
#ifdef HAVE_DCMTK_H
   fileFormat = new DcmFileFormat();
#endif
   isLoaded = false;
   imageInfoObj = new DicomImageInfo( *this );
   patientInfoObj = new DicomPatientInfo( *this );
   seriesInfoObj = new DicomSeriesInfo( *this );
}

inline DicomHeader::~DicomHeader()
{
   delete imageInfoObj;
   delete patientInfoObj;
   delete seriesInfoObj;
#ifdef HAVE_DCMTK_H
   delete fileFormat;
#endif
}

inline bool
DicomHeader::loadFromFile( const String& fileName )
{
#ifdef HAVE_DCMTK_H
   OFCondition status = fileFormat->loadFile( fileName.getString() );
   if( status.good() ) {
      isLoaded = true;
      return true;
   }
#endif
   isLoaded = false;
   return false;
}

#ifdef HAVE_DCMTK_H
inline DcmFileFormat&
DicomHeader::getFileFormat()
{
   return *fileFormat;
}
#endif

inline DicomImageInfo&
DicomHeader::getImageInfo()
{
   return *imageInfoObj;
}

inline DicomPatientInfo&
DicomHeader::getPatientInfo()
{
   return *patientInfoObj;
}

inline DicomSeriesInfo&
DicomHeader::getSeriesInfo()
{
   return *seriesInfoObj;
}

}  // namespace Images
}  // namespace noa::TNL
