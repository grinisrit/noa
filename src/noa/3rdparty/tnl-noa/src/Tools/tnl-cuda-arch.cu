#include <cstdio>
#include <cstring>

int main( int argc, char** argv )
{
    int num_devices = 0;
    cudaError_t error_id = cudaGetDeviceCount( &num_devices );

    if( error_id != cudaSuccess ) {
        fprintf(stderr, "cudaGetDeviceCount returned error %d (%s)\n",
                (int) error_id, cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }

    for( int i = 0; i < num_devices; i++ )
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties( &prop, i );
            int compute_minor = prop.minor;
            // sm_21 is the only 'real' architecture that does not have 'virtual' counterpart
            if( prop.major == 2 )
                compute_minor = 0;

            if( i > 0 )
                printf(" ");

            if( argc == 2 && strcmp( argv[1], "--clang" ) == 0 ) {
                printf( "--cuda-gpu-arch=sm_%d%d",
                        prop.major, prop.minor );
            }
            else {
                printf( "-gencode arch=compute_%d%d,code=sm_%d%d",
                        prop.major, compute_minor, prop.major, prop.minor );
            }
    }
    printf("\n");
}
