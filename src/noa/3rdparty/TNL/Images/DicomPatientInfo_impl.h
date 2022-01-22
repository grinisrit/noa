// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Images/DicomPatientInfo.h>
#include <noa/3rdparty/TNL/Images/DicomHeader.h>

#ifdef HAVE_DCMTK_H
#define HAVE_CONFIG_H
#define HAVE_STD_STRING
#include <dcmtk/ofstd/ofstring.h>
#endif

namespace noaTNL {
namespace Images {   

inline DicomPatientInfo::DicomPatientInfo( DicomHeader &dicomHeader )
: dicomHeader( dicomHeader )
{
    isObjectRetrieved = false;
}

inline DicomPatientInfo::~DicomPatientInfo()
{
}

inline bool DicomPatientInfo::retrieveInfo()
{
#ifdef HAVE_DCMTK_H
   OFString str;
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_PatientName, str );
   this->name = str.data();
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_PatientSex, str );
   this->sex = str.data();
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_PatientID, str );
   this->ID = str.data();
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_PatientWeight, str );
   this->weight = str.data();
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_PatientPosition, str );
   this->patientPosition = str.data();
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_PatientOrientation, str );
   this->patientOrientation = str.data();

   isObjectRetrieved = true;
   return true;
#else
   std::cerr << "DICOM format is not supported in this build of TNL." << std::endl;
   return false;
#endif
}

inline const String& DicomPatientInfo::getName()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return name;
}

inline const String& DicomPatientInfo::getSex()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return sex;
}

inline const String& DicomPatientInfo::getID()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return ID;
}

inline const String& DicomPatientInfo::getWeight()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return weight;
}

inline const String& DicomPatientInfo::getPosition()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return patientPosition;
}

inline const String& DicomPatientInfo::getOrientation()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return patientOrientation;
}

} // namespace Images
} // namespace noaTNL
