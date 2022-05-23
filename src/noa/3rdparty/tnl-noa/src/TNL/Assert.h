// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

/****
 * The purpose of this file is to define the TNL_ASSERT_* debugging macros as
 * shown below.
 *
 * If the 'NDEBUG' macro is defined, the build is considered to be optimized
 * and all assert macros are empty. Otherwise, the conditions are checked and
 * failures lead to the diagnostics message being printed to std::cerr and
 * program abortion (via 'throw EXIT_FAILURE' statement).
 *
 * For the purpose of providing Python bindings it is possible to change the
 * reporting behaviour by defining the TNL_THROW_ASSERTION_ERROR macro, which
 * leads to throwing the ::noa::TNL::Assert::AssertionError holding the error
 * message (which is not printed in this case). The AssertionError class does
 * not inherit from std::exception to avoid being caught by normal exception
 * handlers, but the code for Python bindings can use it to translate it to the
 * Python's AssertionError exception.
 *
 * Implemented by: Jakub Klinkovsky
 */

// wrapper for nvcc pragma which disables warnings about __host__ __device__
// functions: https://stackoverflow.com/q/55481202
#ifdef __NVCC__
   #define TNL_NVCC_HD_WARNING_DISABLE #pragma hd_warning_disable
#else
   #define TNL_NVCC_HD_WARNING_DISABLE
#endif

#ifdef NDEBUG

   // empty macros for optimized build
   /**
    * \brief Asserts that the expression \e val evaluates to \e true.
    *
    * The assertion succeeds if, and only if, \e val evaluates to equal to \e true.
    * On success the test continues without any side effects.
    * On failure the test is terminated with the error message \e msg.
    */
   #define TNL_ASSERT_TRUE( val, msg )
   /**
    * \brief Asserts that the expression \e val evaluates to \e false.
    *
    * The assertion succeeds if, and only if, \e val evaluates to equal to \e false.
    * On success the test continues without any side effects.
    * On failure the test is terminated with the error message \e msg.
    */
   #define TNL_ASSERT_FALSE( val, msg )
   /**
    * \brief Asserts that the expression \e val1 is equal to \e val2.
    *
    * The assertion succeeds if, and only if, \e val1 and \e val2 are equal.
    * On success the test continues without any side effects.
    * On failure the test is terminated with the error message \e msg.
    */
   #define TNL_ASSERT_EQ( val1, val2, msg )
   /**
    * \brief Asserts that the expression \e val1 is not equal to \e val2.
    *
    * The assertion succeeds if, and only if, \e val1 and \e val2 are not equal.
    * On success the test continues without any side effects.
    * On failure the test is terminated with the error message \e msg.
    */
   #define TNL_ASSERT_NE( val1, val2, msg )
   /**
    * \brief Asserts that the expression \e val1 is less than or equal to \e val2.
    *
    * The assertion succeeds if, and only if, \e val1 is less than or equal to \e val2.
    * On success the test continues without any side effects.
    * On failure the test is terminated with the error message \e msg.
    */
   #define TNL_ASSERT_LE( val1, val2, msg )
   /**
    * \brief Asserts that the expression \e val1 is less than \e val2.
    *
    * The assertion succeeds if, and only if, \e val1 is less than \e val2.
    * On success the test continues without any side effects.
    * On failure the test is terminated with the error message \e msg.
    */
   #define TNL_ASSERT_LT( val1, val2, msg )
   /**
    * \brief Asserts that the expression \e val1 is greater than or equal to \e val2.
    *
    * The assertion succeeds if, and only if, \e val1 is greater than or equal to \e val2.
    * On success the test continues without any side effects.
    * On failure the test is terminated with the error message \e msg.
    */
   #define TNL_ASSERT_GE( val1, val2, msg )
   /**
    * \brief Asserts that the expression \e val1 is greater than \e val2.
    *
    * The assertion succeeds if, and only if, \e val1 is greater than \e val2.
    * On success the test continues without any side effects.
    * On failure the test is terminated with the error message \e msg.
    */
   #define TNL_ASSERT_GT( val1, val2, msg )
   /**
    * \brief Asserts that the specified \e ___tnl__assert_condition is valid.
    *
    * The assertion succeeds if, and only if, ___tnl__assert_condition is valid.
    * On success the test continues without any side effects.
    * On failure the test is terminated with the error message \e ___tnl__assert_command.
    */
   #define TNL_ASSERT( ___tnl__assert_condition, ___tnl__assert_command )

#else /* #ifdef NDEBUG */

   #include <sstream>
   #include <iostream>
   #include <cstdio>

   #include <noa/3rdparty/tnl-noa/src/TNL/Cuda/CudaCallable.h>

namespace noa::TNL {
/**
 * \brief Internal namespace for helper classes used in the TNL_ASSERT_* macros.
 */
namespace Assert {

   #ifdef TNL_THROW_ASSERTION_ERROR
// This will be used by the code for Python bindings to translate assertion
// failures to the Python's AssertionError exception.
class AssertionError
{
public:
   AssertionError( const std::string& msg ) : msg( msg ) {}

   const char*
   what() const
   {
      return msg.c_str();
   }

private:
   std::string msg;
};

inline void
printDiagnosticsHost( const char* assertion,
                      const char* message,
                      const char* file,
                      const char* function,
                      int line,
                      const char* diagnostics )
{
   std::stringstream str;
   str << "Assertion '" << assertion << "' failed !!!\n"
       << "Message: " << message << "\n"
       << "File: " << file << "\n"
       << "Function: " << function << "\n"
       << "Line: " << line << "\n"
       << "Diagnostics:\n"
       << diagnostics << std::endl;

   throw AssertionError( str.str() );
}

   #else   // TNL_THROW_ASSERTION_ERROR

// This will be used in regular C++ code
inline void
printDiagnosticsHost( const char* assertion,
                      const char* message,
                      const char* file,
                      const char* function,
                      int line,
                      const char* diagnostics )
{
   std::cerr << "Assertion '" << assertion << "' failed !!!\n"
             << "Message: " << message << "\n"
             << "File: " << file << "\n"
             << "Function: " << function << "\n"
             << "Line: " << line << "\n"
             << "Diagnostics:\n"
             << diagnostics << std::endl;
}
   #endif  // TNL_THROW_ASSERTION_ERROR

__cuda_callable__
inline void
printDiagnosticsCuda( const char* assertion,
                      const char* message,
                      const char* file,
                      const char* function,
                      int line,
                      const char* diagnostics )
{
   std::printf( "Assertion '%s' failed !!!\n"
                "Message: %s\n"
                "File: %s\n"
                "Function: %s\n"
                "Line: %d\n"
                "Diagnostics: %s\n",
                assertion,
                message,
                file,
                function,
                line,
                diagnostics );
}

__cuda_callable__
inline void
fatalFailure()
{
   #ifdef __CUDA_ARCH__
   // https://devtalk.nvidia.com/default/topic/509584/how-to-cancel-a-running-cuda-kernel-/
   // TODO: it is reported as "illegal instruction", but that leads to an abort as well...
   asm( "trap;" );
   #else
   throw EXIT_FAILURE;
   #endif
}

template< typename T >
struct Formatter
{
   static std::string
   printToString( const T& value )
   {
      ::std::stringstream ss;
      ss << value;
      return ss.str();
   }
};

template<>
struct Formatter< bool >
{
   static std::string
   printToString( const bool& value )
   {
      if( value )
         return "true";
      else
         return "false";
   }
};

template< typename T, typename U >
struct Formatter< std::pair< T, U > >
{
   static std::string
   printToString( const std::pair< T, U >& pair )
   {
      ::std::stringstream ss;
      ss << '(' << pair.first << ',' << pair.second << ')';
      return ss.str();
   }
};

template< typename T1, typename T2 >
__cuda_callable__
void
cmpHelperOpFailure( const char* assertion,
                    const char* message,
                    const char* file,
                    const char* function,
                    int line,
                    const char* lhs_expression,
                    const char* rhs_expression,
                    const T1& lhs_value,
                    const T2& rhs_value,
                    const char* op )
{
   #ifdef __CUDA_ARCH__
   // diagnostics is not supported - we don't have the machinery
   // to construct the dynamic error message
   printDiagnosticsCuda( assertion, message, file, function, line, "Not supported in CUDA kernels." );
   #else
   const std::string formatted_lhs_value = Formatter< T1 >::printToString( lhs_value );
   const std::string formatted_rhs_value = Formatter< T2 >::printToString( rhs_value );
   std::stringstream str;
   if( std::string( op ) == "==" ) {
      str << "      Expected: " << lhs_expression;
      if( formatted_lhs_value != lhs_expression ) {
         str << "\n      Which is: " << formatted_lhs_value;
      }
      str << "\nTo be equal to: " << rhs_expression;
      if( formatted_rhs_value != rhs_expression ) {
         str << "\n      Which is: " << formatted_rhs_value;
      }
      str << std::endl;
   }
   else {
      str << "Expected: (" << lhs_expression << ") " << op << " (" << rhs_expression << "), "
          << "actual: " << formatted_lhs_value << " vs " << formatted_rhs_value << std::endl;
   }
   printDiagnosticsHost( assertion, message, file, function, line, str.str().c_str() );
   #endif
   fatalFailure();
}

TNL_NVCC_HD_WARNING_DISABLE
template< typename T1, typename T2 >
__cuda_callable__
void
cmpHelperTrue( const char* assertion,
               const char* message,
               const char* file,
               const char* function,
               int line,
               const char* expr1,
               const char* expr2,
               const T1& val1,
               const T2& val2 )
{
   // explicit cast is necessary, because T1::operator! might not be defined
   if( ! (bool) val1 )
      ::noa::TNL::Assert::cmpHelperOpFailure( assertion, message, file, function, line, expr1, "true", val1, true, "==" );
}

TNL_NVCC_HD_WARNING_DISABLE
template< typename T1, typename T2 >
__cuda_callable__
void
cmpHelperFalse( const char* assertion,
                const char* message,
                const char* file,
                const char* function,
                int line,
                const char* expr1,
                const char* expr2,
                const T1& val1,
                const T2& val2 )
{
   if( val1 )
      ::noa::TNL::Assert::cmpHelperOpFailure( assertion, message, file, function, line, expr1, "false", val1, false, "==" );
}

   // A macro for implementing the helper functions needed to implement
   // TNL_ASSERT_??. It is here just to avoid copy-and-paste of similar code.
   #define TNL_IMPL_CMP_HELPER_( op_name, op )                                                                            \
      template< typename T1, typename T2 >                                                                                \
      __cuda_callable__                                                                                                   \
      void cmpHelper##op_name( const char* assertion,                                                                     \
                               const char* message,                                                                       \
                               const char* file,                                                                          \
                               const char* function,                                                                      \
                               int line,                                                                                  \
                               const char* expr1,                                                                         \
                               const char* expr2,                                                                         \
                               const T1& val1,                                                                            \
                               const T2& val2 )                                                                           \
      {                                                                                                                   \
         if( ! ( (val1) op( val2 ) ) )                                                                                    \
            ::noa::TNL::Assert::cmpHelperOpFailure( assertion, message, file, function, line, expr1, expr2, val1, val2, #op ); \
      }

// Implements the helper function for TNL_ASSERT_EQ
TNL_NVCC_HD_WARNING_DISABLE
TNL_IMPL_CMP_HELPER_( EQ, == );
// Implements the helper function for TNL_ASSERT_NE
TNL_NVCC_HD_WARNING_DISABLE
TNL_IMPL_CMP_HELPER_( NE, != );
// Implements the helper function for TNL_ASSERT_LE
TNL_NVCC_HD_WARNING_DISABLE
TNL_IMPL_CMP_HELPER_( LE, <= );
// Implements the helper function for TNL_ASSERT_LT
TNL_NVCC_HD_WARNING_DISABLE
TNL_IMPL_CMP_HELPER_( LT, < );
// Implements the helper function for TNL_ASSERT_GE
TNL_NVCC_HD_WARNING_DISABLE
TNL_IMPL_CMP_HELPER_( GE, >= );
// Implements the helper function for TNL_ASSERT_GT
TNL_NVCC_HD_WARNING_DISABLE
TNL_IMPL_CMP_HELPER_( GT, > );

   #undef TNL_IMPL_CMP_HELPER_

}  // namespace Assert
}  // namespace noa::TNL

   // Internal macro wrapping the __PRETTY_FUNCTION__ "magic".
   #if defined( __NVCC__ ) && ( __CUDACC_VER_MAJOR__ < 8 )
      #define __TNL_PRETTY_FUNCTION "(not known in CUDA 7.5 or older)"
   #else
      #define __TNL_PRETTY_FUNCTION __PRETTY_FUNCTION__
   #endif

   // On Linux, __STRING is defined in glibc's sys/cdefs.h, but there is no such
   // header on Windows and possibly other platforms.
   #ifndef __STRING
      #define __STRING( arg ) #arg
   #endif

   // Internal macro to compose the string representing the assertion.
   // We can't do it easily at runtime, because we have to support assertions
   // in CUDA kernels, which can't use std::string objects. Instead, we do it
   // at compile time - adjacent strings are joined at the language level.
   #define __TNL_JOIN_STRINGS( val1, op, val2 ) __STRING( val1 ) " " __STRING( op ) " " __STRING( val2 )

   // Internal macro to pass all the arguments to the specified cmpHelperOP
   #define __TNL_ASSERT_PRED2( pred, op, val1, val2, msg ) \
      pred( __TNL_JOIN_STRINGS( val1, op, val2 ), msg, __FILE__, __TNL_PRETTY_FUNCTION, __LINE__, #val1, #val2, val1, val2 )

   // Main definitions of the TNL_ASSERT_* macros
   // unary
   #define TNL_ASSERT_TRUE( val, msg ) __TNL_ASSERT_PRED2( ::noa::TNL::Assert::cmpHelperTrue, ==, val, true, msg )
   #define TNL_ASSERT_FALSE( val, msg ) __TNL_ASSERT_PRED2( ::noa::TNL::Assert::cmpHelperFalse, ==, val, false, msg )
   // binary
   #define TNL_ASSERT_EQ( val1, val2, msg ) __TNL_ASSERT_PRED2( ::noa::TNL::Assert::cmpHelperEQ, ==, val1, val2, msg )
   #define TNL_ASSERT_NE( val1, val2, msg ) __TNL_ASSERT_PRED2( ::noa::TNL::Assert::cmpHelperNE, !=, val1, val2, msg )
   #define TNL_ASSERT_LE( val1, val2, msg ) __TNL_ASSERT_PRED2( ::noa::TNL::Assert::cmpHelperLE, <=, val1, val2, msg )
   #define TNL_ASSERT_LT( val1, val2, msg ) __TNL_ASSERT_PRED2( ::noa::TNL::Assert::cmpHelperLT, <, val1, val2, msg )
   #define TNL_ASSERT_GE( val1, val2, msg ) __TNL_ASSERT_PRED2( ::noa::TNL::Assert::cmpHelperGE, >=, val1, val2, msg )
   #define TNL_ASSERT_GT( val1, val2, msg ) __TNL_ASSERT_PRED2( ::noa::TNL::Assert::cmpHelperGT, >, val1, val2, msg )

   /****
    * Original assert macro with custom command for diagnostics.
    */

   // __CUDA_ARCH__ is defined by the compiler only for code executed on GPU
   #ifdef __CUDA_ARCH__
      #define TNL_ASSERT( ___tnl__assert_condition, ___tnl__assert_command )                                             \
         if( ! ( ___tnl__assert_condition ) ) {                                                                          \
            std::printf( "Assertion '%s' failed !!! \n File: %s \n Line: %d \n Diagnostics: Not supported with CUDA.\n", \
                         __STRING( ___tnl__assert_condition ),                                                           \
                         __FILE__,                                                                                       \
                         __LINE__ );                                                                                     \
            asm( "trap;" );                                                                                              \
         }

   #else  // #ifdef __CUDA_ARCH__
      #ifdef TNL_THROW_ASSERTION_ERROR

         // This will be used by the code for Python bindings to translate assertion
         // failures to the Python's AssertionError exception.
         #define TNL_ASSERT( ___tnl__assert_condition, ___tnl__assert_command )                                  \
            if( ! ( ___tnl__assert_condition ) ) {                                                               \
               std::stringstream buffer;                                                                         \
               auto old = std::cerr.rdbuf( buffer.rdbuf() );                                                     \
                                                                                                                 \
               std::cerr << "Assertion '" << __STRING( ___tnl__assert_condition ) << "' failed !!!" << std::endl \
                         << "File: " << __FILE__ << std::endl                                                    \
                         << "Function: " << __PRETTY_FUNCTION__ << std::endl                                     \
                         << "Line: " << __LINE__ << std::endl                                                    \
                         << "Diagnostics: ";                                                                     \
               ___tnl__assert_command;                                                                           \
                                                                                                                 \
               std::string msg = buffer.str();                                                                   \
               std::cerr.rdbuf( old );                                                                           \
               throw ::noa::TNL::Assert::AssertionError( msg );                                                       \
            }

      #else  // #ifdef TNL_THROW_ASSERTION_ERROR

         // This will be used in regular C++ code
         #define TNL_ASSERT( ___tnl__assert_condition, ___tnl__assert_command )                                  \
            if( ! ( ___tnl__assert_condition ) ) {                                                               \
               std::cerr << "Assertion '" << __STRING( ___tnl__assert_condition ) << "' failed !!!" << std::endl \
                         << "File: " << __FILE__ << std::endl                                                    \
                         << "Function: " << __TNL_PRETTY_FUNCTION << std::endl                                   \
                         << "Line: " << __LINE__ << std::endl                                                    \
                         << "Diagnostics: ";                                                                     \
               ___tnl__assert_command;                                                                           \
               throw EXIT_FAILURE;                                                                               \
            }

      #endif  // #ifdef TNL_THROW_ASSERTION_ERROR
   #endif     // #ifdef __CUDA_ARCH__

#endif  // #ifdef NDEBUG
