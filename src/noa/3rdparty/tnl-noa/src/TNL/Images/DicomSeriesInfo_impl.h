// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdio>

#include <noa/3rdparty/tnl-noa/src/TNL/Images/DicomHeader.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Images/DicomSeriesInfo.h>

namespace noa::TNL {
namespace Images {

inline DicomSeriesInfo::DicomSeriesInfo( DicomHeader& dicomHeader ) : dicomHeader( dicomHeader )
{
   isObjectRetrieved = false;
}

inline DicomSeriesInfo::~DicomSeriesInfo() = default;

inline bool
DicomSeriesInfo::retrieveInfo()
{
#ifdef HAVE_DCMTK_H
   OFString str;
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_Modality, str );
   this->modality = str.data();

   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_StudyInstanceUID, str );
   this->studyInstanceUID = str.data();

   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_SeriesInstanceUID, str );
   this->seriesInstanceUID = str.data();

   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_SeriesNumber, str );
   this->seriesNumber = str.data();

   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_SeriesDescription, str );
   this->seriesDescription = str.data();

   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_SeriesDate, str );
   this->seriesDate = str.data();

   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_SeriesTime, str );
   this->seriesTime = str.data();

   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_PerformingPhysicianName, str );
   this->performingPhysiciansName = str.data();

   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_PerformingPhysicianIdentificationSequence, str );
   this->performingPhysicianIdentificationSequence = str.data();

   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_OperatorsName, str );
   this->operatorsName = str.data();

   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_OperatorIdentificationSequence, str );
   this->operatorIdentificationSequence = str.data();

   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_FrameAcquisitionDuration, str );
   this->frameTime = str.data();

   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_FrameAcquisitionDateTime, str );
   this->faDateTime = str.data();

   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_FrameReferenceTime, str );
   this->faRefTime = str.data();

   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_ActualFrameDuration, str );
   this->AFD = str.data();

   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_AcquisitionTime, str );
   this->acquisitionTime = str.data();

   // prostudovat delay time
   // OFString delayTime = "";
   // dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_DelayTime, delayTime);

   // std::cout << faDateTime << " " << faRefTime << " "<< AFD << " " << AT << std::endl;

   isObjectRetrieved = true;
   return true;
#else
   std::cerr << "DICOM format is not supported in this build of TNL." << std::endl;
   return false;
#endif
}

inline const String&
DicomSeriesInfo::getModality()
{
   if( ! isObjectRetrieved )
      retrieveInfo();
   return this->modality;
}

inline const String&
DicomSeriesInfo::getStudyInstanceUID()
{
   if( ! isObjectRetrieved )
      retrieveInfo();
   return this->studyInstanceUID;
}

inline const String&
DicomSeriesInfo::getSeriesInstanceUID()
{
   if( ! isObjectRetrieved )
      retrieveInfo();
   return this->seriesInstanceUID;
}

inline const String&
DicomSeriesInfo::getSeriesNumber()
{
   if( ! isObjectRetrieved )
      retrieveInfo();
   return this->seriesNumber;
}

inline const String&
DicomSeriesInfo::getSeriesDescription()
{
   if( ! isObjectRetrieved )
      retrieveInfo();
   return this->seriesDescription;
}

inline const String&
DicomSeriesInfo::getSeriesDate()
{
   if( ! isObjectRetrieved )
      retrieveInfo();
   return this->seriesDate;
}

inline const String&
DicomSeriesInfo::getSeriesTime()
{
   if( ! isObjectRetrieved )
      retrieveInfo();
   return this->seriesTime;
}

inline const String&
DicomSeriesInfo::getPerformingPhysiciansName()
{
   if( ! isObjectRetrieved )
      retrieveInfo();
   return this->performingPhysiciansName;
}

inline const String&
DicomSeriesInfo::getPerformingPhysicianIdentificationSequence()
{
   if( ! isObjectRetrieved )
      retrieveInfo();
   return this->performingPhysicianIdentificationSequence;
}

inline const String&
DicomSeriesInfo::getOperatorsName()
{
   if( ! isObjectRetrieved )
      retrieveInfo();
   return this->operatorsName;
}

inline const String&
DicomSeriesInfo::getOperatorIdentificationSequence()
{
   if( ! isObjectRetrieved )
      retrieveInfo();
   return this->operatorIdentificationSequence;
}

inline const String&
DicomSeriesInfo::getAcquisitionTime()
{
   if( ! isObjectRetrieved )
      retrieveInfo();
   return this->acquisitionTime;
}

}  // namespace Images
}  // namespace noa::TNL
