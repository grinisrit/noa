// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Images//DicomHeader.h>
#include <TNL/Images//DicomSeriesInfo.h>
#include <TNL/Images//DicomPatientInfo.h>
#include <TNL/Images//DicomImageInfo.h>

namespace TNL {
namespace Images {

inline DicomHeader::DicomHeader()
{
#ifdef HAVE_DCMTK_H
    fileFormat = new DcmFileFormat();
#endif
    isLoaded = false;
    imageInfoObj = new DicomImageInfo(*this);
    patientInfoObj = new DicomPatientInfo(*this);
    seriesInfoObj = new DicomSeriesInfo(*this);
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

inline bool DicomHeader::loadFromFile( const String& fileName )
{
#ifdef HAVE_DCMTK_H
    OFCondition status = fileFormat->loadFile( fileName.getString() );
    if(status.good())
    {
        isLoaded = true;
        return true;
    }
#endif
    isLoaded = false;
    return false;
}

#ifdef HAVE_DCMTK_H
inline DcmFileFormat &DicomHeader::getFileFormat()
{
    return *fileFormat;
}
#endif

inline DicomImageInfo &DicomHeader::getImageInfo()
{
    return *imageInfoObj;
}

inline DicomPatientInfo &DicomHeader::getPatientInfo()
{
    return *patientInfoObj;
}

inline DicomSeriesInfo &DicomHeader::getSeriesInfo()
{
    return *seriesInfoObj;
}

} // namespace Images
} // namespace TNL

