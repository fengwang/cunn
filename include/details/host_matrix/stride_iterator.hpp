#ifndef MSTRIDE_ITERATOR_HPP_INCLUDED_SDPOIANSDWU9HASFKLJANSFKLJN439UISAHFLKJ49U
#define MSTRIDE_ITERATOR_HPP_INCLUDED_SDPOIANSDWU9HASFKLJANSFKLJN439UISAHFLKJ49U

#include "../utility.hpp"

namespace cunn
{
    // TODO: adding `noexcept` wherever possible

    /**
     *      <summary>
     *      Stride iterator is an iterator adaptor that takes a random access iterator range and provides
     *      a random access iterator on it goes through a stride (a sequence of iterator n steps apart),
     *      which is useful when striding a matrix column or diagonal.
     *      </summary>
     */
    template<typename Iterator_Type>
    struct stride_iterator
    {

        typedef Iterator_Type                                                           iterator_type;
        typedef typename std::iterator_traits<iterator_type>::value_type                value_type;
        typedef typename std::iterator_traits<iterator_type>::reference                 reference;
        typedef typename std::iterator_traits<iterator_type>::difference_type           difference_type;
        typedef typename std::iterator_traits<iterator_type>::pointer                   pointer;
        typedef typename std::iterator_traits<iterator_type>::iterator_category         iterator_category;
        typedef stride_iterator                                                         self_type;
        typedef typename std::make_unsigned<difference_type>                            unsigned_difference_type;

        static_assert( std::is_same<std::random_access_iterator_tag, iterator_category>::value, "Not a random access iterator" );

        iterator_type                       iterator_;      /**< iterator stored */
        difference_type                     step_;          /**< stride step */

        ///  default ctor.
        stride_iterator() noexcept : iterator_( 0 ), step_( 1 ) { }

        stride_iterator( const self_type& ) noexcept = default;

        stride_iterator( self_type&& ) noexcept = default;

        self_type& operator = ( const self_type& ) noexcept = default;

        self_type& operator = ( self_type&& ) noexcept = default;

        /// <summary>
        /// ctor from an iterator and a step
        /// </summary>
        ///
        /// <param name="it">
        /// a random access iterator
        /// </param>
        /// <param name="dt">
        /// a non-zero integer, for the stride step
        /// </param>
        stride_iterator( const iterator_type& it, const difference_type& dt ) noexcept : iterator_( it ), step_( dt ) { }

        /// <summary>
        /// prefix increasement,
        /// </summary>
        /// the iterator stored in this structure, #iterator_, increased by a step size #step_
        self_type& operator++() noexcept
        {
            iterator_ += step_;
            return *this;
        }

        /// <summary>
        /// postfix increasement
        /// </summary>
        /// @see operator++()
        const self_type operator ++( int ) noexcept
        {
            self_type ans( *this );
            operator++();
            return ans;
        }

        /// <summary>
        /// operator +=
        /// </summary>
        /// <param name="dt">
        /// a constant integer
        /// </param>
        /// the stored iterator, #iterator_, steps forward \p dt times. This is equivalent to execute #operator++ \p dt times
        self_type& operator+=( const difference_type dt ) noexcept
        {
            iterator_ += dt * step_;
            return *this;
        }

        /// <summary>
        /// operator +, taking a stride_iterator and an interger
        /// </summary>
        /// <param name="lhs">
        /// a constant stride_iterator instance
        /// </param>
        /// <param name="rhs">
        /// a constant unsigned interger, times the lhs stride_iterator increase
        /// </param>
        /// @returns A new stride_iterator representing the lhs stride_iterator increases rhs times
        /// @see operator+=
        friend const self_type operator + ( const self_type& lhs, const difference_type rhs ) noexcept
        {
            self_type ans( lhs );
            ans += rhs;
            return ans;
        }

        friend const self_type operator + ( const self_type& lhs, const unsigned_difference_type rhs ) noexcept // to remove clang++ warnings of -Wsign-conversion
        {
            return lhs + static_cast<difference_type>(rhs);
        }

        /// <summary>
        /// operator +, taking an integer and a stride iterator
        /// </summary>
        /// <param name="lhs">
        /// a constant unsigned interger, increase times of the rhs stride_itrator
        /// </param>
        /// <param name="rhs">
        /// a constant stride_iterator instance
        /// </param>
        /// @returns A new stride_iterator representing the rhs stride_iterator increases lhs times
        /// @see operator+=
        friend const self_type operator + ( const difference_type lhs, const self_type& rhs ) noexcept
        {
            return rhs + lhs;
        }

        friend const self_type operator + ( const unsigned_difference_type lhs, const self_type& rhs ) noexcept // to remove clang++ warnings
        {
            return static_cast<difference_type>(lhs) + rhs;
        }

        /// <summary>
        /// prefix decreasement
        /// </summary>
        /// the stored iterator, #iterator_ steps back, i.e., decreased by a step size #step_
        self_type& operator--() noexcept
        {
            iterator_ -= step_;
            return *this;
        }

        /// <summary>
        /// postfix decreasement
        /// </summary>
        /// @see operator--()
        const self_type operator -- ( int ) noexcept
        {
            self_type ans( *this );
            operator--();
            return ans;
        }

        /// <summary>
        /// operator -=
        /// <summary>
        /// <param name="dt">
        /// a constant integer
        /// </param>
        /// the stored iterator, #iterator_, steps backward \p dt times. This is equivalent to execute #operator-- \p dt times
        self_type& operator-=( const difference_type dt ) noexcept
        {
            iterator_ -= dt * step_;
            return *this;
        }

        /// <summary>
        /// operator -
        /// </summary>
        /// <param name="lhs">
        /// a constant stride_iterator instance
        /// </param>
        /// <param name="rhs">
        /// a constant unsigned interger, times the lhs stride_iterator decrease
        /// </param>
        /// @returns A new stride_iterator representing the lhs stride_iterator decreases \p rhs times
        /// @see operator-=
        friend const self_type operator - ( const self_type& lhs, const difference_type rhs ) noexcept
        {
            self_type ans( lhs );
            ans -= rhs;
            return ans;
        }

        /// <summary>
        /// lvalue accessment
        /// <summary>
        /// <param name="dt">
        /// a constant integer
        /// </param>
        reference operator[]( const difference_type dt ) noexcept
        {
            return iterator_[dt * step_];
        }

        /// <summary>
        /// const lvalue accessment
        /// </summary>
        const reference operator[]( const difference_type dt ) const noexcept
        {
            return iterator_[dt * step_];
        }

        /// <summary>
        /// deptr
        /// </summary>
        reference operator*() noexcept
        {
            return *iterator_;
        }

        /// <summary>
        /// const dptr
        /// </summary>
        const reference operator*() const noexcept
        {
            return *iterator_;
        }

        /// <summary>
        /// operator ==
        /// </summary>
        friend bool operator == ( const self_type& lhs, const self_type& rhs ) noexcept
        {
            assert( lhs.step_ == rhs.step_ );
            return lhs.iterator_ == rhs.iterator_;
        }

        /// <summary>
        /// operator !=
        /// </summary>
        friend bool operator != ( const self_type& lhs, const self_type& rhs ) noexcept
        {
            assert( lhs.step_ == rhs.step_ );
            return lhs.iterator_ != rhs.iterator_;
        }

        /// <summary>
        /// operator <
        /// </summary>
        friend bool operator < ( const self_type& lhs, const self_type& rhs ) noexcept
        {
            assert( lhs.step_ == rhs.step_ );
            return lhs.iterator_ < rhs.iterator_;
        }

        /// <summary>
        /// operator <=
        /// </summary>
        friend bool operator <= ( const self_type& lhs, const self_type& rhs ) noexcept
        {
            assert( lhs.step_ == rhs.step_ );
            return lhs.iterator_ <= rhs.iterator_;
        }

        /// <summary>
        /// operator >
        /// </summary>
        friend bool operator > ( const self_type& lhs, const self_type& rhs ) noexcept
        {
            assert( lhs.step_ == rhs.step_ );
            return lhs.iterator_ > rhs.iterator_;
        }

        /// <summary>
        /// operator >=
        /// </summary>
        friend bool operator >= ( const self_type& lhs, const self_type& rhs ) noexcept
        {
            assert( lhs.step_ == rhs.step_ );
            return lhs.iterator_ >= rhs.iterator_;
        }

        /// <summary>
        /// operator -
        /// </summary>
        friend difference_type operator - ( const self_type& lhs, const self_type& rhs ) noexcept
        {
            assert( lhs.step_ == rhs.step_ );
            return ( lhs.iterator_ - rhs.iterator_ ) / lhs.step_;
        }

        /// <summary>
        /// return the step
        /// </summary>
        difference_type step() const noexcept
        {
            return step_;
        }

    }; //stride_iterator

}//namespace NAMESPACE_MATRIX

#endif//_STRIDE_ITERATOR_HPP_INCLUDED_SDPOIANSDWU9HASFKLJANSFKLJN439UISAHFLKJ49U

