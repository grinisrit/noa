// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/String.h>

#ifdef HAVE_DCMTK_H
#define HAVE_CONFIG_H
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#define HAVE_STD_STRING
#include <dcmtk/ofstd/ofstring.h>
#endif

namespace noaTNL {
namespace Images {   

class DicomHeader;

/***
 * SeriesInfoObj class stores selected informations about DICOM series.
 * (accesses information via DicomHeader class)
 */
class DicomSeriesInfo
{
   public:
 
       inline DicomSeriesInfo( DicomHeader &dicomHeader );
 
       inline virtual ~DicomSeriesInfo();

       inline const String& getModality();
 
       inline const String& getStudyInstanceUID();
 
       inline const String& getSeriesInstanceUID();
 
       inline const String& getSeriesDescription();
 
       inline const String& getSeriesNumber();
 
       inline const String& getSeriesDate();
 
       inline const String& getSeriesTime();
 
       inline const String& getPerformingPhysiciansName();
 
       inline const String& getPerformingPhysicianIdentificationSequence();
 
       inline const String& getOperatorsName();
 
       inline const String& getOperatorIdentificationSequence();
 
       inline const String& getAcquisitionTime();
 
   private:
 
       DicomHeader &dicomHeader;
 
       bool retrieveInfo();
 
       bool isObjectRetrieved;

       String modality;

       String studyInstanceUID;

       String seriesInstanceUID;

       String seriesNumber;

       String seriesDescription;

       String seriesDate;

       String seriesTime;

       String performingPhysiciansName;

       String performingPhysicianIdentificationSequence;

       String operatorsName;

       String operatorIdentificationSequence;

       String frameTime;

       String faDateTime;

       String faRefTime;

       String AFD;

       String acquisitionTime;
};

} // namespace Images
} // namespace noaTNL

#include <noa/3rdparty/TNL/Images//DicomSeriesInfo_impl.h>

