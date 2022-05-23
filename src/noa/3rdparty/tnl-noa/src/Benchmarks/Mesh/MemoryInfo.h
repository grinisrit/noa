#pragma once

// References:
// - https://stackoverflow.com/a/64166/4180822
// - https://lemire.me/blog/2020/03/03/calling-free-or-delete/
// - https://stackoverflow.com/questions/15529643/what-does-malloc-trim0-really-mean

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <sys/types.h>
#include <sys/sysinfo.h>
#include <malloc.h>

inline long
getTotalVirtualMemory()
{
    struct sysinfo memInfo;
    sysinfo (&memInfo);

    long totalVirtualMem = memInfo.totalram;
    // Add other values in next statement to avoid int overflow on right hand side...
    totalVirtualMem += memInfo.totalswap;
    totalVirtualMem *= memInfo.mem_unit;

    return totalVirtualMem;
}

inline long
getUsedVirtualMemory()
{
    struct sysinfo memInfo;
    sysinfo (&memInfo);

    long virtualMemUsed = memInfo.totalram - memInfo.freeram;
    // Add other values in next statement to avoid int overflow on right hand side...
    virtualMemUsed += memInfo.totalswap - memInfo.freeswap;
    virtualMemUsed *= memInfo.mem_unit;

    return virtualMemUsed;
}

inline long
parseLine(char* line)
{
    // This assumes that a digit will be found and the line ends in " kB".
    int i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++;
    line[i-3] = '\0';
    return atol(p);
}

// virtual memory currently used by the calling process
inline long
getSelfVirtualMemory()
{
    // explicitly release unused memory
    malloc_trim(0);

    FILE* file = fopen("/proc/self/status", "r");
    long result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmSize:", 7) == 0){
            // convert from kB to B
            result = parseLine(line) * 1024;
            break;
        }
    }
    fclose(file);
    return result;
}

inline long
getTotalPhysicalMemory()
{
    struct sysinfo memInfo;
    sysinfo (&memInfo);

    long totalPhysMem = memInfo.totalram;
    //Multiply in next statement to avoid int overflow on right hand side...
    totalPhysMem *= memInfo.mem_unit;

    return totalPhysMem;
}

inline long
getUsedPhysicalMemory()
{
    struct sysinfo memInfo;
    sysinfo (&memInfo);

    long physMemUsed = memInfo.totalram - memInfo.freeram;
    //Multiply in next statement to avoid int overflow on right hand side...
    physMemUsed *= memInfo.mem_unit;

    return physMemUsed;
}

inline long
getSelfPhysicalMemory()
{
    // explicitly release unused memory
    malloc_trim(0);

    FILE* file = fopen("/proc/self/status", "r");
    long result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmRSS:", 6) == 0){
            // convert from kB to B
            result = parseLine(line) * 1024;
            break;
        }
    }
    fclose(file);
    return result;
}


#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/StaticVector.h>

struct MemoryBenchmarkResult
: public TNL::Benchmarks::BenchmarkResult
{
   using HeaderElements = TNL::Benchmarks::Logging::HeaderElements;
   using RowElements = TNL::Benchmarks::Logging::RowElements;

   double memory = std::numeric_limits<double>::quiet_NaN();
   double memstddev = std::numeric_limits<double>::quiet_NaN();

   virtual HeaderElements getTableHeader() const override
   {
      return HeaderElements({ "time", "stddev", "stddev/time", "bandwidth", "speedup", "memory", "memstddev", "memstddev/memory" });
   }

   virtual RowElements getRowElements() const override
   {
      RowElements elements;
      elements << time << stddev << stddev / time << bandwidth;
      if( speedup != 0 )
         elements << speedup;
      else
         elements << "N/A";
      elements << memory << memstddev << memstddev / memory;
      return elements;
   }
};

template< long MAX_COPIES = 10, typename Mesh >
MemoryBenchmarkResult
testMemoryUsage( const TNL::Config::ParameterContainer& parameters,
                 const Mesh& mesh )
{
    const size_t memoryLimit = parameters.getParameter< size_t >( "mem-limit" ) * 1024 * 1024;
    TNL::Containers::StaticVector< MAX_COPIES, Mesh > meshes;
    TNL::Containers::StaticVector< MAX_COPIES, double > data;
    data.setValue( 0 );

    long prevCheck = getSelfPhysicalMemory();
    meshes[0] = mesh;
    long check = getSelfPhysicalMemory();
    data[0] = check - prevCheck;
    prevCheck = check;
    const int copies = TNL::min( memoryLimit / data[0], MAX_COPIES - 1 ) + 1;

    for( int i = 1; i < copies; i++ ) {
        meshes[i] = mesh;
        check = getSelfPhysicalMemory();
        data[i] = check - prevCheck;
        prevCheck = check;
    }

    MemoryBenchmarkResult result;

    const double mean = TNL::sum( data ) / (double) copies;
    result.memory = mean / 1024.0 / 1024.0;  // MiB

    if( copies > 1 ) {
        for( int i = copies; i < MAX_COPIES; i++ ) {
            data[i] = mean;
        }

        const double stddev = 1.0 / std::sqrt( copies ) * TNL::l2Norm( data - mean );
        result.memstddev = stddev / 1024.0 / 1024.0;  // MiB
    }

    return result;
}
