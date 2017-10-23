#ifndef MMINUS_EQUAL_OPERATOR_HPP_INCLUDED_SDPOIJHASLKJSDLSFD9H4LHASF98Y4VKJBFDI
#define MMINUS_EQUAL_OPERATOR_HPP_INCLUDED_SDPOIJHASLKJSDLSFD9H4LHASF98Y4VKJBFDI

#include "./typedef.hpp"

namespace cunn
{
    template<typename Matrix, typename Type, typename Allocator>
    struct crtp_minus_equal_operator
    {
        typedef Matrix                                                          zen_type;
        typedef Type                                                            value_type;

        zen_type& operator -=( const value_type& rhs )
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            //std::transform( zen.begin(), zen.end(), zen.begin(), std::bind2nd( std::minus<value_type>(), rhs ) );
            std::transform( zen.begin(), zen.end(), zen.begin(), [rhs](auto x){ return x - rhs; } );
            return zen;
        }

        zen_type& operator -=( const zen_type& rhs )
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            //std::transform( zen.begin(), zen.end(), rhs.begin(), zen.begin(), std::minus<value_type>() );
            std::transform( zen.begin(), zen.end(), rhs.begin(), zen.begin(), []( auto x, auto y ){ return x-y; } );
            return zen;
        }

    };//struct crtp_minus_equal_operator

}

#endif//_MINUS_EQUAL_OPERATOR_HPP_INCLUDED_SDPOIJHASLKJSDLSFD9H4LHASF98Y4VKJBFDI

