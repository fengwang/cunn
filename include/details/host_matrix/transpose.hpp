#ifndef PVJYMCRAIHFNVEGKOCGGUFVJIGMCWWKKNBKSVFRDSBJCHEMPIQBQDVLEHINSGLIENEDTVCWXT
#define PVJYMCRAIHFNVEGKOCGGUFVJIGMCWWKKNBKSVFRDSBJCHEMPIQBQDVLEHINSGLIENEDTVCWXT

#include "./typedef.hpp"

namespace cunn
{
    template<typename Matrix, typename Type, typename Allocator>
    struct crtp_transpose
    {
        typedef Matrix                                                          zen_type;
        typedef Type                                                            value_type;

        zen_type const&  transpose() const noexcept
        {
            zen_type& zen = static_cast<zen_type const&>( *this );
            zen_type ans{ zen.col(), zen.row() };

            for ( auto r = 0UL; r != zen.row(); ++r )
                for ( auto c = 0UL; c != zen.col(); ++c )
                    ans[c][r] = zen[r][c];
            return  ans;
        }

    };//struct crtp_transpose

}

#endif//PVJYMCRAIHFNVEGKOCGGUFVJIGMCWWKKNBKSVFRDSBJCHEMPIQBQDVLEHINSGLIENEDTVCWXT

