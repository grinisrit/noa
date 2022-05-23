#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/Array.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

using namespace TNL;
using namespace std;


template< typename Device >
void VectorExample()
{
    Containers::Vector< int, Device > vector1( 5 );
    vector1 = 0;

    Containers::Vector< int, Device > vector2( 3 );
    vector2 = 1;
    vector2.swap( vector1 );
    vector2.setElement( 2, 4 );

    cout << "First vector:" << vector1.getData() << endl;
    cout << "Second vector:" << vector2.getData() << endl;

    vector2.reset();
    cout << "Second vector after reset:" << vector2.getData() << endl;

    Containers::Vector< int, Device > vect = { 1, 2, -3, 3 };
    cout << "The smallest element is:" << min( vect ) << endl;
    cout << "The absolute biggest element is:" << max( abs( vect ) ) << endl;
    cout << "Sum of all vector elements:" << sum( vect ) << endl;
    vect *= 2.0;
    cout << "Vector multiplied by 2:" << vect << endl;
}

int main()
{
    std::cout << "Running vector example on the host system: " << std::endl;
    VectorExample< Devices::Host >();

#ifdef HAVE_CUDA
    std::cout << "Running vector example on the CUDA device: " << std::endl;
    VectorExample< Devices::Cuda >();
#endif
}

