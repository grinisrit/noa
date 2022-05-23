// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HAVE_DCMTK_H
   #define HAVE_CONFIG_H
   #include <dcmtk/dcmdata/dcfilefo.h>
   #include <dcmtk/dcmdata/dcdeftag.h>
#endif

namespace noa::TNL {
namespace Images {

class DicomSeriesInfo;
class DicomPatientInfo;
class DicomImageInfo;

/***
 * Class provides acces to the DICOM file header (contains complete
 *   information about DICOM file) and stores the information objects
 *   focused on essential data about image, patient and serie.
 */
class DicomHeader
{
public:
   inline DicomHeader();

   inline virtual ~DicomHeader();

#ifdef HAVE_DCMTK_H
   inline DcmFileFormat&
   getFileFormat();
#endif

   inline DicomImageInfo&
   getImageInfo();

   inline DicomPatientInfo&
   getPatientInfo();

   inline DicomSeriesInfo&
   getSeriesInfo();

   inline bool
   loadFromFile( const String& fileName );

protected:
   DicomImageInfo* imageInfoObj;

   DicomPatientInfo* patientInfoObj;

   DicomSeriesInfo* seriesInfoObj;

#ifdef HAVE_DCMTK_H
   DcmFileFormat* fileFormat;
#endif

   bool isLoaded;
};

}  // namespace Images
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Images/DicomHeader_impl.h>
