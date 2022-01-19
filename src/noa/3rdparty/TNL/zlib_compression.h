// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <cstdint>  // std::uint64_t

#include <zlib.h>

#include <TNL/base64.h>

namespace TNL {

/**
 * \brief Use zlib to compress the data, then encode it with base64 and write
 * the result to the given stream.
 *
 * Options for compression_level:
 *   - Z_NO_COMPRESSION
 *   - Z_BEST_SPEED
 *   - Z_BEST_COMPRESSION
 *   - Z_DEFAULT_COMPRESSION
 */
template< typename HeaderType = std::uint64_t, typename T >
void
write_compressed_block( const T* data,
                        const std::size_t data_size,
                        std::ostream& output_stream,
                        const int compression_level = Z_DEFAULT_COMPRESSION )
{
   if (data_size == 0)
      return;

   // allocate a buffer for compressing data and do so
   uLongf compressed_data_length = compressBound(data_size * sizeof(T));
   std::unique_ptr<std::uint8_t[]> compressed_data{new std::uint8_t[compressed_data_length]};

   // compress the data
   const int status = compress2(
            reinterpret_cast<Bytef*>(compressed_data.get()),
            &compressed_data_length,
            reinterpret_cast<const Bytef*>(data),
            data_size * sizeof(T),
            compression_level
         );
   if (status != Z_OK)
      throw std::runtime_error("zlib compression failed");

   // create compression header
   const HeaderType compression_header[4] = {
      (HeaderType) 1,                       // number of blocks
      (HeaderType) (data_size * sizeof(T)), // size of block
      (HeaderType) (data_size * sizeof(T)), // size of last block
      (HeaderType) compressed_data_length   // list of compressed sizes of blocks
   };

   // base64-encode the compression header
   std::unique_ptr<char[]> encoded_header(
            base64::encode(reinterpret_cast<const std::uint8_t*>(&compression_header[0]),
                           4 * sizeof(HeaderType))
         );
   output_stream << encoded_header.get();

   // base64-encode the compressed data
   std::unique_ptr<char[]> encoded_data(
            base64::encode(compressed_data.get(), compressed_data_length)
         );
   output_stream << encoded_data.get();
}

/**
 * \brief Decompress data in given byte array and return an array of elements of
 * type \e T and length \e data_size.
 */
template< typename T >
std::unique_ptr<T[]>
decompress_data( const std::uint8_t* decoded_data, const std::size_t decoded_data_length, const std::size_t data_size )
{
   // decompress the data
   std::unique_ptr<T[]> data{new T[data_size]};
   uLongf uncompressed_length_data = data_size * sizeof(T);
   const int status = uncompress(
            reinterpret_cast<Bytef*>(data.get()),
            &uncompressed_length_data,
            reinterpret_cast<const Bytef*>(decoded_data),
            decoded_data_length
         );
   if (status != Z_OK)
      throw std::runtime_error("zlib decompression failed");

   if (uncompressed_length_data != data_size * sizeof(T))
      throw std::length_error("uncompressed data length does not match the size stored in the compression header: "
                              + std::to_string(uncompressed_length_data) + " vs "
                              + std::to_string(data_size));

   return data;
}

/**
 * \brief Decode and decompress data from a stream. The data must be in the
 * format as produced by the \ref write_compressed_block function.
 *
 * The function returns a pair of the resulting data size and a \ref std::unique_ptr
 * to the data.
 */
template< typename HeaderType = std::uint64_t, typename T >
std::pair<HeaderType, std::unique_ptr<T[]>>
decompress_block( const char* data )
{
   // decode the header
   const int encoded_header_length = base64::get_encoded_length(4 * sizeof(HeaderType));
   std::pair<std::size_t, std::unique_ptr<std::uint8_t[]>> decoded_header = base64::decode(data, encoded_header_length);
   const HeaderType* compression_header = reinterpret_cast<const HeaderType*>(decoded_header.second.get());

   if (compression_header[0] != 1)
      throw std::length_error("unexpected number of compressed blocks: "
                              + std::to_string(compression_header[0]));
   // Note: for short arrays, paraview 5.9 creates files with:
   //       num_blocks == 1, block_size == 32768, last_block_size == [the actual data size]
   //if (compression_header[1] != compression_header[2])
   //    throw std::logic_error("inconsistent block sizes in the compression header: "
   //                           + std::to_string(compression_header[1]) + " vs "
   //                           + std::to_string(compression_header[2]));
   const HeaderType data_size = compression_header[2] / sizeof(T);
   const HeaderType compressed_data_length = base64::get_encoded_length(compression_header[3]);

   // decode the data
   std::pair<std::size_t, std::unique_ptr<std::uint8_t[]>> decoded_data = base64::decode(data + encoded_header_length, compressed_data_length);

   // decompress the data and return
   return {data_size, decompress_data<T>(decoded_data.second.get(), decoded_data.first, data_size)};
}

/**
 * \brief Decode and decompress data from a stream. The data must be in the
 * format as produced by the \ref write_compressed_block function.
 *
 * The function returns a pair of the resulting data size and a \ref std::unique_ptr
 * to the data.
 */
template< typename HeaderType = std::uint64_t, typename T >
std::pair< HeaderType, std::unique_ptr<T[]> >
decompress_block( std::istream& input_stream )
{
   // read the header
   const int encoded_header_length = base64::get_encoded_length(4 * sizeof(HeaderType));
   std::unique_ptr<char[]> encoded_header{new char[encoded_header_length]};
   input_stream.read(encoded_header.get(), encoded_header_length);
   if (!input_stream.good())
      throw std::length_error("input is not long enough to contain a compression header");

   // decode the header
   std::pair<std::size_t, std::unique_ptr<std::uint8_t[]>> decoded_header = base64::decode(encoded_header.get(), encoded_header_length);
   const HeaderType* compression_header = reinterpret_cast<const HeaderType*>(decoded_header.second.get());

   if (compression_header[0] != 1)
      throw std::length_error("unexpected number of compressed blocks: "
                              + std::to_string(compression_header[0]));
   // Note: for short arrays, paraview 5.9 creates files with:
   //       num_blocks == 1, block_size == 32768, last_block_size == [the actual data size]
   //if (compression_header[1] != compression_header[2])
   //    throw std::logic_error("inconsistent block sizes in the compression header: "
   //                           + std::to_string(compression_header[1]) + " vs "
   //                           + std::to_string(compression_header[2]));
   const HeaderType data_size = compression_header[2] / sizeof(T);
   const HeaderType compressed_data_length = base64::get_encoded_length(compression_header[3]);

   // read the compressed data
   std::unique_ptr<char[]> encoded_data{new char[compressed_data_length]};
   input_stream.read(encoded_data.get(), compressed_data_length);
   if (!input_stream.good())
      throw std::length_error("failed to read the compressed data");

   // decode the data
   std::pair<std::size_t, std::unique_ptr<std::uint8_t[]>> decoded_data = base64::decode(encoded_data.get(), compressed_data_length);

   // decompress the data and return
   return {data_size, decompress_data<T>(decoded_data.second.get(), decoded_data.first, data_size)};
}

} // namespace TNL
