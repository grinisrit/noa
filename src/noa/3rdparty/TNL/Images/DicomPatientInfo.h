// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/String.h>

#ifdef HAVE_DCMTK_H
#define HAVE_CONFIG_H
#define HAVE_STD_STRING
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include <dcmtk/ofstd/ofstring.h>
#endif

namespace TNL {
namespace Images {   

class DicomHeader;

/***
 * PatientInfoObj class stores selected informations about patient.
 * (accesses information via DicomHeader class)
 */
class DicomPatientInfo
{
   public:
 
      inline DicomPatientInfo(DicomHeader &aDicomHeader);
 
      inline virtual ~DicomPatientInfo();

      inline const String& getName();
 
      inline const String& getSex();
 
      inline const String& getID();
 
      inline const String& getWeight();
 
      inline const String& getPosition();
 
      inline const String& getOrientation();

   private:

       DicomHeader &dicomHeader;
       bool retrieveInfo();
       bool isObjectRetrieved;

       String name;

       String sex;

       String ID;

       String weight;

       String patientPosition;

       String patientOrientation;
};

} // namespace Images
} // namespace TNL

#include <TNL/Images//DicomPatientInfo_impl.h>

