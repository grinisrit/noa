// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <stdexcept>
#include <cmath>    // std::ceil

namespace noaTNL {
/**
 * \brief Namespace for base64 encoding and decoding functions.
 *
 * The actual algorithms are based on these sources:
 *
 * - http://web.mit.edu/freebsd/head/contrib/wpa/src/utils/base64.c
 * - https://stackoverflow.com/questions/180947/base64-decode-snippet-in-c/
 * - https://stackoverflow.com/questions/342409/how-do-i-base64-encode-decode-in-c
 */
namespace base64 {

/**
 * \brief Get the length of base64-encoded block for given data byte length.
 */
inline std::size_t
get_encoded_length( std::size_t byte_length )
{
   std::size_t encoded = std::ceil(byte_length * (4.0 / 3.0));
   // base64 uses padding to a multiple of 4
   if( encoded % 4 == 0 )
      return encoded;
   return encoded + 4 - (encoded % 4);
}

/**
 * \brief Static table for base64 encoding.
 */
static constexpr unsigned char encoding_table[65] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/**
 * \brief Static table for base64 decoding.
 *
 * Can be built with the following code:
 *
 * \code
 * std::uint8_t decoding_table[256];
 * for( int i = 0; i < 256; i++ )
 *    decoding_table[i] = 128;
 * for( std::uint8_t i = 0; i < sizeof(encoding_table) - 1; i++ )
 *    decoding_table[encoding_table[i]] = i;
 * decoding_table[(int) '='] = 0;
 * \endcode
 */
static constexpr std::uint8_t decoding_table[256] = {
   128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
   128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
   128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,  62, 128, 128, 128,  63,
    52,  53,  54,  55,  56,  57,  58,  59,  60,  61, 128, 128, 128,   0, 128, 128,
   128,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
    15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25, 128, 128, 128, 128, 128,
   128,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,
    41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51, 128, 128, 128, 128, 128,
   128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
   128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
   128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
   128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
   128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
   128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
   128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
   128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128
};

/**
 * \brief Do a base64 encoding of the given data.
 *
 * \param data Pointer to the data to be encoded.
 * \param data_size Length of the input data (in bytes).
 * \return A \ref std::unique_ptr to the encoded data.
 */
inline std::unique_ptr<char[]>
encode( const std::uint8_t* data, std::size_t data_size )
{
   const std::size_t output_length = get_encoded_length( data_size );
   std::unique_ptr<char[]> encoded_data{new char[output_length + 1]};

   const std::uint8_t* end = data + data_size;
   const std::uint8_t* in = data;
   char* out = encoded_data.get();
   char* pos = out;

   while( end - in >= 3 ) {
      *pos++ = encoding_table[in[0] >> 2];
      *pos++ = encoding_table[((in[0] & 0x03) << 4) | (in[1] >> 4)];
      *pos++ = encoding_table[((in[1] & 0x0f) << 2) | (in[2] >> 6)];
      *pos++ = encoding_table[in[2] & 0x3f];
      in += 3;
   }

   if( end - in ) {
      *pos++ = encoding_table[in[0] >> 2];
      if( end - in == 1 ) {
         *pos++ = encoding_table[(in[0] & 0x03) << 4];
         *pos++ = '=';
      }
      else {
         *pos++ = encoding_table[((in[0] & 0x03) << 4) | (in[1] >> 4)];
         *pos++ = encoding_table[(in[1] & 0x0f) << 2];
      }
      *pos++ = '=';
   }

   *pos++ = '\0';
   return encoded_data;
}

/**
 * \brief Internal base64 decoding function.
 *
 * \param input Pointer to the encoded data (C string).
 * \param input_length Length of the input string.
 * \param output Pointer to a pre-allocated output buffer.
 * \param output_length Length of the output buffer.
 * \return Size of the decoded data (in bytes).
 */
inline std::ptrdiff_t
decode_block( const char* input, std::size_t input_length, std::uint8_t* output, std::size_t output_length )
{
   const std::size_t min_buffer_size = std::ceil(input_length * (3.0 / 4.0));
   if( output_length < min_buffer_size )
      throw std::logic_error( "base64: insufficient output buffer size " + std::to_string(output_length)
                              + " (needed at least " + std::to_string(min_buffer_size) + " bytes)" );

   std::size_t count = 0;
   int pad = 0;
   std::uint8_t block[4];
   std::uint8_t* pos = output;

   for (std::size_t i = 0; i < input_length; i++) {
      const std::uint8_t tmp = decoding_table[(int) input[i]];
      if( tmp == 128 )
         continue;

      if( input[i] == '=' )
         pad++;
      block[count] = tmp;
      count++;

      if( count == 4 ) {
         *pos++ = (block[0] << 2) | (block[1] >> 4);
         *pos++ = (block[1] << 4) | (block[2] >> 2);
         *pos++ = (block[2] << 6) | block[3];
         count = 0;
         if( pad > 2 )
            // invalid padding
            throw std::invalid_argument( "base64: decoding error: input has invalid padding" );
         if( pad > 0 ) {
            pos -= pad;
            break;
         }
      }
   }

   // check left-over chars
   if( count )
      throw std::invalid_argument( "base64: decoding error: invalid input (length not padded to a multiple of 4)" );

   return pos - output;
}

/**
 * \brief Do a base64 decoding of the given data.
 *
 * \param data Pointer to the encoded data (C string).
 * \param data_size Length of the input string.
 * \return A pair of the decoded data length and a \ref std::unique_ptr to the
 *         decoded data.
 */
inline std::pair< std::size_t, std::unique_ptr<std::uint8_t[]> >
decode( const char* data, const std::size_t data_size )
{
   const std::size_t buffer_size = std::ceil(data_size * (3.0 / 4.0));
   std::unique_ptr<std::uint8_t[]> decoded_data{new std::uint8_t[buffer_size + 1]};

   const std::size_t decoded_length_data = decode_block( data, data_size, decoded_data.get(), buffer_size );
   return {decoded_length_data, std::move(decoded_data)};
}

/**
 * \brief Write a base64-encoded block of data into the given stream.
 *
 * The encoded data is prepended with a short header, which is the base64-encoded
 * byte length of the data. The type of the byte length value is `HeaderType`.
 */
template< typename HeaderType = std::uint64_t, typename T >
void
write_encoded_block( const T* data, const std::size_t data_length, std::ostream& output_stream )
{
   const HeaderType size = data_length * sizeof(T);
   std::unique_ptr<char[]> encoded_size = base64::encode( reinterpret_cast<const std::uint8_t*>(&size), sizeof(HeaderType) );
   output_stream << encoded_size.get();
   std::unique_ptr<char[]> encoded_data = base64::encode( reinterpret_cast<const std::uint8_t*>(data), size );
   output_stream << encoded_data.get();
}

} // namespace base64
} // namespace noaTNL
