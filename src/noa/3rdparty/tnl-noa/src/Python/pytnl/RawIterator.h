#pragma once

#include <iterator>

template< typename DataType >
class RawIterator : public std::iterator<std::random_access_iterator_tag,
                                           DataType,
                                           ptrdiff_t,
                                           DataType*,
                                           DataType&>
{
protected:
    DataType*               m_ptr;

public:
    RawIterator( DataType* ptr = nullptr ) { m_ptr = ptr; }
    RawIterator( const RawIterator<DataType> & rawIterator ) = default;
    ~RawIterator(){}

    RawIterator<DataType>&  operator=( const RawIterator<DataType> & rawIterator ) = default;
    RawIterator<DataType>&  operator=( DataType* ptr ) { m_ptr = ptr; return (*this); }

    operator                bool() const
    {
        if(m_ptr)
            return true;
        else
            return false;
    }

    bool                    operator==( const RawIterator<DataType> & rawIterator ) const { return ( m_ptr == rawIterator.getConstPtr() ); }
    bool                    operator!=( const RawIterator<DataType> & rawIterator ) const { return ( m_ptr != rawIterator.getConstPtr() ); }

    RawIterator<DataType>&  operator+=( const ptrdiff_t & movement ){ m_ptr += movement; return (*this); }
    RawIterator<DataType>&  operator-=( const ptrdiff_t & movement ){ m_ptr -= movement; return (*this); }
    RawIterator<DataType>&  operator++() { ++m_ptr; return (*this); }
    RawIterator<DataType>&  operator--() { --m_ptr; return (*this); }
    RawIterator<DataType>   operator++( int ) { auto temp(*this); ++m_ptr; return temp; }
    RawIterator<DataType>   operator--( int ) { auto temp(*this); --m_ptr; return temp; }
    RawIterator<DataType>   operator+( const ptrdiff_t & movement ) { auto oldPtr = m_ptr; m_ptr+=movement; auto temp(*this); m_ptr = oldPtr; return temp; }
    RawIterator<DataType>   operator-( const ptrdiff_t & movement ) { auto oldPtr = m_ptr; m_ptr-=movement; auto temp(*this); m_ptr = oldPtr; return temp; }

    ptrdiff_t               operator-( const RawIterator<DataType>& rawIterator ) { return std::distance(rawIterator.getPtr(), this->getPtr()); }

    DataType&               operator*() { return *m_ptr; }
    const DataType&         operator*() const { return *m_ptr; }
    DataType*               operator->() { return m_ptr; }

    DataType*               getPtr() const { return m_ptr; }
    const DataType*         getConstPtr() const { return m_ptr; }
};
