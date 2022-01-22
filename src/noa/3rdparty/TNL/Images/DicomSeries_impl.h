// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Images//DicomSeries.h>
#include <noa/3rdparty/TNL/Images//DicomSeriesInfo.h>
#include <dirent.h>

namespace noaTNL {
namespace Images {   

int findLastIndexOf(String &str, const char* c)
{
    for (int i = str.getLength(); i > -1; i--)
    {
        char *a = &(str.operator [](i-1));
        if(*a == *c)
            return i;
    }
    return -1;
}

int filter(const struct dirent *dire)
{
    //check it is not DIR or unknowm d_type
    if(dire->d_type == DT_UNKNOWN && dire->d_type == DT_DIR)
        return 0;

    return 1;
}

inline DicomSeries::DicomSeries( const String& filePath)
{
#ifdef HAVE_DCMTK_H
    dicomImage = 0;
    pixelData = 0;
#endif
    imagesInfo.imagesCount = 0;
    imagesInfo.maxColorValue = 0;
    imagesInfo.minColorValue = 128000;

    if( !loadDicomSeries( filePath ) )
        isLoaded = false;
    else
        isLoaded = true;
}

inline DicomSeries::~DicomSeries()
{
    int length = dicomSeriesHeaders.getSize();
    for(int i = 0; i < length; i++)
    {
        DicomHeader *header = dicomSeriesHeaders[i];
        delete header;
        header = 0;
    }

#ifdef HAVE_DCMTK_H
    if(dicomImage)
        delete dicomImage;

    if(pixelData)
        delete pixelData;
#endif
}

template< typename Real,
          typename Device,
          typename Index,
          typename Vector >
bool
DicomSeries::
getImage( const int imageIdx,
          const Meshes::Grid< 2, Real, Device, Index >& grid,
          const RegionOfInterest< int > roi,
          Vector& vector )
{
#ifdef HAVE_DCMTK_H
   const Uint16* imageData = this->getData( imageIdx );
   typedef Meshes::Grid< 2, Real, Device, Index > GridType;
   typename GridType::Cell cell( grid );
 
   Index i, j;
   int position( 0 );
   for( i = 0; i < this->height; i ++ )
   {
      for( j = 0; j < this->width; j ++ )
      {
         if( roi.isIn( i, j ) )
         {
            cell.getCoordinates().x() = j - roi.getLeft();
            cell.getCoordinates().y() = roi.getBottom() - 1 - i;
            cell.refresh();
            //Index cellIndex = grid.getCellIndex( CoordinatesType( j - roi.getLeft(),
            //                                                      roi.getBottom() - 1 - i ) );
            Uint16 col = imageData[ position ];
            vector.setElement( cell.getIndex(), ( Real ) col / ( Real ) 65535 );
            //cout << vector.getElement( cellIndex ) << " ";
         }
         position++;
      }
      //cout << std::endl;
   }
   return true;
#else
   std::cerr << "DICOM format is not supported in this build of TNL." << std::endl;
   return false;
#endif
}

inline bool DicomSeries::retrieveFileList( const String& filePath)
{
    String filePathString(filePath);
    String suffix(filePath.getString(), filePathString.getLength() - 3);

    /***
     * Check DICOM files
     */
   if( suffix != "ima" && suffix != "dcm" )
   {
       std::cerr << "The given file is not a DICOM file." << std::endl;
      return false;
   }

   int fileNamePosition = findLastIndexOf( filePathString, "/" );

   /***
    * Parse file path
    */
   String fileName(filePath.getString(), fileNamePosition);
   String directoryPath(filePath.getString(), 0, filePathString.getLength() - fileNamePosition);

   int separatorPosition = findLastIndexOf(fileName, "_");
   if (separatorPosition == -1)
   {
      //try another separator
      separatorPosition = findLastIndexOf(fileName, "-");
   }
   if( separatorPosition == -1 )
      return false;
   else
   {
      //numbered files
      String fileNamePrefix(fileName.getString(), 0, fileName.getLength() - separatorPosition);

      struct dirent **dirp;
      std::list< String > files;

      //scan and sort directory
      int ndirs = scandir(directoryPath.getString(), &dirp, filter, alphasort);
      for(int i = 0 ; i < ndirs; ++i)
      {
         files.push_back( String((char *)dirp[i]->d_name) );
         delete dirp[i];
      }

      for (auto& file : files)
      {
         //check if file prefix contained
         if (strstr(file.getString(), fileNamePrefix.getString()))
         {
            fileList.push_back( directoryPath + file );
         }
      }
   }
   return true;
}

inline bool DicomSeries::loadImage( const String& filePath, int number)
{
#ifdef HAVE_DCMTK_H
   //load header
   DicomHeader *header = new DicomHeader();
   dicomSeriesHeaders.setSize( fileList.size() );
   dicomSeriesHeaders.setElement( number, header );
   if( !header->loadFromFile( filePath ) )
      return false;

   //check series UID
   const String& seriesUID = dicomSeriesHeaders[ 0 ]->getSeriesInfo().getSeriesInstanceUID();
   if( seriesUID != header->getSeriesInfo().getSeriesInstanceUID() )
      return false;

   //load image
   if( dicomImage ) delete dicomImage;
   dicomImage = NULL;

   dicomImage = new DicomImage( filePath.getString() );

   if(dicomImage->getFrameCount() > 1)
   {
     std::cout << filePath <<" not supported format-Dicom Image has more than one frame";
      return false;
   }

   if(!dicomImage->isMonochrome())
   {
     std::cout << filePath <<" not supported format--Dicom Image is not monochrome";
      return false;
   }

    if (dicomImage != NULL)
    {
        EI_Status imageStatus = dicomImage->getStatus();
        if (imageStatus == EIS_Normal)
        {
            //ok - image loaded
        }
        else if (imageStatus == EIS_MissingAttribute)
        {
            //bitmap is propably old ARC/NEMA format
            std::cerr << "Error: cannot load DICOM image(ACR/NEMA) (" << DicomImage::getString (dicomImage->getStatus()) << ")" << std::endl;

            delete dicomImage;
            dicomImage = NULL;
            return false;
        }
        else
        {
            delete dicomImage;
            dicomImage = NULL;
            std::cerr << "Error: cannot load DICOM image (" << DicomImage::getString (dicomImage->getStatus()) << ")" << std::endl;
            return false;
        }
    }

    if(number == 0)
    {
        this->height = dicomImage->getHeight();
    }
    else if( ( IndexType ) dicomImage->getHeight() != this->height)
    {
        std::cerr << filePath <<" image has bad height value\n";
    }

    if(number == 0)
    {
        this->width = dicomImage->getWidth ();
    }
    else if( ( IndexType ) dicomImage->getWidth() != this->width)
    {
        std::cerr << filePath <<" image has bad width value\n";
    }

    if(number == 0)
    {
        imagesInfo.bps = dicomImage->getDepth ();
    }
    else if( ( IndexType ) dicomImage->getDepth() != imagesInfo.bps )
    {
        std::cerr << filePath <<" image has bad bps value\n";
    }

    //update vales
    double min, max;
    dicomImage->getMinMaxValues( min, max );
    if(imagesInfo.minColorValue > min)
    {
        imagesInfo.minColorValue = min;
    }

    if(imagesInfo.maxColorValue < max)
    {
        imagesInfo.maxColorValue = max;
    }

    const unsigned long size = dicomImage->getOutputDataSize(16);
    //number of unsigned ints to allocate
    imagesInfo.frameUintsCount = size / sizeof(Uint16);
    if (number == 0)
    {//perform allocation only once
        imagesInfo.frameSize = size;
        if (pixelData)
            delete pixelData;
        pixelData = new Uint16[imagesInfo.frameUintsCount * fileList.size()];
    }
    else
    {//check image size for compatibility
        if( ( unsigned long ) imagesInfo.frameSize != size )
        {
            std::cerr << filePath << " image has bad frame size value\n";
            return false;
        }
    }

    dicomImage->setMinMaxWindow();
    double center, width;
    dicomImage->getWindow(center,width);
    imagesInfo.window.center = center;
    imagesInfo.window.width = width ;
    dicomImage->setWindow(imagesInfo.window.center, imagesInfo.window.width);

    void *target = pixelData + (imagesInfo.frameUintsCount * imagesInfo.imagesCount);
    dicomImage->getOutputData(target,size,16);
    imagesInfo.imagesCount++;

    //delete image object - data are stored separately
    delete dicomImage;
    dicomImage = NULL;
    return true;
#else
    std::cerr << "DICOM format is not supported in this build of TNL." << std::endl;
    return false;
#endif
}


inline bool DicomSeries::loadDicomSeries( const String& filePath )
{
   /***
    * Load list of files
    */
   if( ! retrieveFileList( filePath ) )
   {
      std::cerr << "I am not able to retrieve the files of the DICOM series in " << filePath << "." << std::endl;
      return false;
   }

   //load images
   int counter = 0;
   for( auto& file : fileList )
   {
      if( !loadImage( file.getString(), counter ) )
      {
         std::cerr << file << " skipped";
      }
      counter++;
   }
   return true;
}

inline int DicomSeries::getImagesCount()
{
    return imagesInfo.imagesCount;
}

#ifdef HAVE_DCMTK_H
inline const Uint16 *DicomSeries::getData( int imageNumber )
{
    return &pixelData[ imageNumber * imagesInfo.frameUintsCount ];
}
#endif

inline int DicomSeries::getColorCount()
{
    return imagesInfo.colorsCount;
}

inline int DicomSeries::getBitsPerSampleCount()
{
    return imagesInfo.bps;
}

inline int DicomSeries::getMinColorValue()
{
    return imagesInfo.minColorValue;
}

inline WindowCenterWidth DicomSeries::getWindowDefaults()
{
    return imagesInfo.window;
}

inline int DicomSeries::getMaxColorValue()
{
    return imagesInfo.maxColorValue;
}

inline void DicomSeries::freeData()
{
#ifdef HAVE_DCMTK_H
    if (pixelData)
        delete pixelData;
    pixelData = NULL;
#endif
}

inline DicomHeader &DicomSeries::getHeader(int image)
{
    //check user argument
    if((image > 0) | (image <= dicomSeriesHeaders.getSize()))
        return *dicomSeriesHeaders.getElement(image);
    throw std::out_of_range("image index out of range");
}

inline bool DicomSeries::isDicomSeriesLoaded()
{
    return isLoaded;
}

} // namespace Images
} // namespace noaTNL
