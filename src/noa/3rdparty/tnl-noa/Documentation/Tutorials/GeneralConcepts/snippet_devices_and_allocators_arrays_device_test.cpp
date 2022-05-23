template< typename Array >
void testDevice
{
    using Device = typename Array::DeviceType;
    if( std::is_same< Device, TNL::Device::Host >::value )
        std::cout << "Device is host CPU." << std::endl;
    if( std::is_same< Device, TNL::Device::Cuda >::value )
        std::cout << "Device is CUDA GPU." << std::endl;
}