#ifndef MBRACKET_OPERATOR_HPP_INCLUDED_SDOIJASF983AFLJKVCMNASKLJHA9H834AKLJSFFDE
#define MBRACKET_OPERATOR_HPP_INCLUDED_SDOIJASF983AFLJKVCMNASKLJHA9H834AKLJSFFDE

#include "./typedef.hpp"

namespace cunn
{
    template<typename Matrix, typename Type, typename Allocator>
    struct crtp_bracket_operator
    {
        typedef Matrix                                      zen_type;
        typedef unsigned long                               size_type;

        auto operator[]( const size_type index )
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            assert( index < zen.row() && "Row index outof boundary!" );
            return zen.row_begin( index );
        }

        auto operator[]( const size_type index ) const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            assert( index < zen.row() && "Row index outof boundary!" );
            return zen.row_begin( index );
        }
    };//struct crtp_typedef

}

#endif//_BRACKET_OPERATOR_HPP_INCLUDED_SDOIJASF983AFLJKVCMNASKLJHA9H834AKLJSFFDE

