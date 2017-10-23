#ifndef MATRIX_VIEW_HPP_INCLUDED_FDSAS3498USDKLJ498USFDALIJ49IJFASDLKJ48I9JFSKF
#define MATRIX_VIEW_HPP_INCLUDED_FDSAS3498USDKLJ498USFDALIJ49IJFASDLKJ48I9JFSKF

extern "C" void cuda_gemm_ab_c( unsigned long a_row, unsigned long a_col, unsigned long b_col, float const* A, float const* B, float const* C );
extern "C" void cuda_gemm_aTb_c(    float const* A, unsigned long a_row, unsigned long a_col,
                                    float const* B, unsigned long b_row, unsigned long b_col,
                                    float* C, unsigned long c_row, unsigned long c_col );
extern "C" void cuda_gemm_abT_c(    float const* A, unsigned long a_row, unsigned long a_col,
                                    float const* B, unsigned long b_row, unsigned long b_col,
                                    float* C, unsigned long c_row, unsigned long c_col );
#include "../cuda/cuda.hpp"
#include "../utility.hpp"
namespace cunn
{
    template< typename Type >
    struct device_matrix_view
    {
        typedef Type                                    value_type;
        typedef value_type*                             pointer;
        typedef const pointer                           const_pointer;
        typedef unsigned long                           size_type;

        size_type               row_;                   ///< row of the device_matrix_view
        size_type               col_;                   ///< column of the device_matrix_view
        pointer                 data_;                  ///< pointer to the allocated memory

        /// <summary>
        /// Default constructor.
        /// </summary>
        /// <param name="row">
        /// Row of the device_matrix_view.
        /// </param>
        /// <param name="col">
        /// Column of the device_matrix_view.
        /// </param>
        /// <param name="data">
        /// The memory to use for this device_matrix_view.
        /// </param>
        device_matrix_view( size_type const row = 0, size_type const col = 0, pointer data = nullptr ) : row_{row}, col_{col}, data_{data} { }

        pointer begin() noexcept
        {
            return data_;
        }

        pointer end() noexcept
        {
            return data_ + row_*col_;
        }

        pointer begin() const noexcept
        {
            return data_;
        }

        pointer end() const noexcept
        {
            return data_ + row_*col_;
        }

        /// <summary>
        /// Access allocated memory
        /// </summary>
        /// @return The direct pointer to the memory allocated, #data_.
        pointer data() noexcept
        {
            return data_;
        }

        /// <summary>
        /// Access allocated memory
        /// </summary>
        /// @return The direct constant pointer to the memory allocated, #data_.
        const_pointer data() const noexcept
        {
            return data_;
        }

        /// <summary>
        /// get the column of the device_matrix_view
        /// </summary>
        /// @return Returns the column of the device_matrix_view
        ///
        /// Example:
        /// @code{.cpp}
        /// device_matrix_view<int> m{3, 4};
        /// assert( m.col() == 4 );
        /// @endcode
        size_type col() const noexcept
        {
            return col_;
        }

        /// <summary>
        /// get the row of the device_matrix_view
        /// </summary>
        /// @return Returns the row of the device_matrix_view, #row_
        ///
        /// Example
        /// @code{.cpp}
        /// device_matrix_view<double> m{ 12, 34 };
        /// asser( m.row() == 12 );
        /// @endcode
        auto row() const noexcept
        {
            return row_;
        }

        /// <summary>
        /// get the size of the device_matrix_view
        /// </summary>
        /// @return Returns the size of the device_matrix_view
        /// @code{.cpp}
        /// device_matrix_view<float> mf{ 2, 4 };
        /// assert( mf.size() == 2*4 );
        /// @endcode
        auto size() const noexcept
        {
            return row_ * col_;
        }

        /// <summayr>
        /// resize the device_matrix_view
        /// </summary>
        void resize( size_type new_row, size_type new_col )
        {
            //log( "Matrix view mapped to memory address ", data_, " is resized from ", row_, " by ", col_, " to ", new_row, " by ", new_col );
            row_ = new_row;
            col_ = new_col;
        }

        /// <summary>
        /// Swap the contents of two matrices
        /// </summary>
        /// <param name="other">
        /// Anther device_matrix_view to be exchanged with the host device_matrix_view, if it is identical to the host device_matrix_view, the behaviour is undefined.
        /// </param>
        ///
        /// Example usage to swap two matrices
        /// @code{.cpp}
        /// //...
        /// device_matrix_view<double> m1{ 2, 3 }
        /// device_matrix_view<double> m2{ 3, 2 }
        /// //...
        /// m1.swap( m2 );
        /// //...
        /// @endcode
        void swap( device_matrix_view<Type>& other )
        {
            std::swap( row_, other.row_ );
            std::swap( col_, other.col_ );
            std::swap( data_, other.data_ );
        }

    };//struct

    template< typename Type >
    std::ostream& operator << ( std::ostream& os, device_matrix_view<Type> const& view )
    {
        std::vector<Type> vec;
        vec.resize( view.size() );
        device_to_host( view.begin(), view.end(), vec.data() );
        auto start = vec.begin();
        for ( auto r = 0UL; r != view.row(); ++r )
        {
            for ( auto c = 0UL; c != view.col(); ++c )
                os << vec[r*view.col()+c] << "\t";
            os << std::endl;
        }
        return os;
    }


    // c = a*b
    inline void gemm_ab_c( device_matrix_view<float> const& a_, device_matrix_view<float> const& b_, device_matrix_view<float>& c_ )
    {
        assert( a_.row() == c_.row() );
        assert( a_.col() == b_.row() );
        assert( b_.col() == c_.col() );

        cuda_gemm_ab_c( a_.row(), a_.col(), b_.col(), a_.data(), b_.data(), c_.data() );
    }

    // c = a^T b
    inline void gemm_aTb_c( device_matrix_view<float> const& a_, device_matrix_view<float> const& b_, device_matrix_view<float>& c_ )
    {
        assert( a_.col() == c_.row() );
        assert( a_.row() == b_.row() );
        assert( b_.col() == c_.col() );

        cuda_gemm_aTb_c( a_.data(), a_.row(), a_.col(), b_.data(), b_.row(), b_.col(), c_.data(), c_.row(), c_.col() );
    }

    // c = a b^T
    inline void gemm_abT_c( device_matrix_view<float> const& a_, device_matrix_view<float> const& b_, device_matrix_view<float>& c_ )
    {
        assert( a_.row() == c_.row() );
        assert( a_.col() == b_.col() );
        assert( b_.row() == c_.col() );

        cuda_gemm_abT_c( a_.data(), a_.row(), a_.col(), b_.data(), b_.row(), b_.col(), c_.data(), c_.row(), c_.col() );
    }

}//namespace

#endif
