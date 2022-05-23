template< typename Array >
void deduceDevice
{
    using Device = typename Array::DeviceType;
    TNL::Container::Array< int, Device > array;
}