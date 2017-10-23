#ifndef MDIRECT_ITERATOR_HPP_INCLUDED_DSPON43PIOHAFSKLJH4098HASFDLKJNHSDKFJNADFD
#define MDIRECT_ITERATOR_HPP_INCLUDED_DSPON43PIOHAFSKLJH4098HASFDLKJNHSDKFJNADFD

#include "./typedef.hpp"

namespace cunn
{
    template<typename Matrix, typename Type, typename Allocator>
    struct crtp_iterator
    {
        typedef Matrix                                                          zen_type;
        typedef crtp_typedef<Type, Allocator>                                   type_proxy_type;
        typedef typename type_proxy_type::anti_diag_type                        anti_diag_type;
        typedef typename type_proxy_type::col_type                              col_type;
        typedef typename type_proxy_type::const_anti_diag_type                  const_anti_diag_type;
        typedef typename type_proxy_type::const_col_type                        const_col_type;
        typedef typename type_proxy_type::const_diag_type                       const_diag_type;
        typedef typename type_proxy_type::const_iterator                        const_iterator;
        typedef typename type_proxy_type::const_pointer                         const_pointer;
        typedef typename type_proxy_type::const_reverse_anti_diag_type          const_reverse_anti_diag_type;
        typedef typename type_proxy_type::const_reverse_col_type                const_reverse_col_type;
        typedef typename type_proxy_type::const_reverse_diag_type               const_reverse_diag_type;
        typedef typename type_proxy_type::const_reverse_iterator                const_reverse_iterator;
        typedef typename type_proxy_type::const_reverse_lower_diag_type         const_reverse_lower_diag_type;
        typedef typename type_proxy_type::const_reverse_row_type                const_reverse_row_type;
        typedef typename type_proxy_type::const_reverse_upper_diag_type         const_reverse_upper_diag_type;
        typedef typename type_proxy_type::const_row_type                        const_row_type;
        typedef typename type_proxy_type::diag_type                             diag_type;
        typedef typename type_proxy_type::difference_type                       difference_type;
        typedef typename type_proxy_type::iterator                              iterator;
        typedef typename type_proxy_type::pointer                               pointer;
        typedef typename type_proxy_type::reverse_anti_diag_type                reverse_anti_diag_type;
        typedef typename type_proxy_type::reverse_col_type                      reverse_col_type;
        typedef typename type_proxy_type::reverse_diag_type                     reverse_diag_type;
        typedef typename type_proxy_type::reverse_iterator                      reverse_iterator;
        typedef typename type_proxy_type::reverse_lower_diag_type               reverse_lower_diag_type;
        typedef typename type_proxy_type::reverse_row_type                      reverse_row_type;
        typedef typename type_proxy_type::reverse_upper_diag_type               reverse_upper_diag_type;
        typedef typename type_proxy_type::row_type                              row_type;
        typedef typename type_proxy_type::size_type                             size_type;

        /// <summary>
        /// Gets an iterator pointing to the first element in the allocated memory, note that the iterator is not necessarily valid
        /// </summary>
        /// @return An iterator to the beginning of the allocated memory
        iterator begin() noexcept
        {
            auto& zen = static_cast<zen_type&>(*this);
            return &(zen.data_[0]);
        }

        /// <summary>
        /// Gets a constant iterator pointing to the first element in the allocated memory, note that the iterator is not necessarily valid
        /// </summary>
        /// @return A constant iterator to the beginning of the allocated memory
        const_iterator begin() const noexcept
        {
            auto& zen = static_cast<zen_type const&>(*this);
            return &(zen.data_[0]);
        }

        /// <summary>
        /// Gets a constant iterator pointing to the first element in the allocated memory, note that the iterator is not necessarily valid
        /// </summary>
        /// @return A constant iterator to the beginning of the allocated memory
        const_iterator cbegin() const noexcept
        {
            return begin();
        }

        /// <summary>
        /// Returns an iterator referring to the past-the-end element in the allocated memory
        /// </summary>
        /// @return An iterator to the element past the end of the allocated memory
        iterator end() noexcept
        {
            auto& zen = static_cast<zen_type const&>(*this);
            return begin() + zen.size();
        }

        /// <summary>
        /// Returns a constant iterator referring to the past-the-end element in the allocated memory
        /// </summary>
        /// @return A constant iterator to the element past the end of the allocated memory
        const_iterator end() const noexcept
        {
            auto& zen = static_cast<zen_type const&>(*this);
            return begin() + zen.size();
        }

        /// <summary>
        /// Returns a constant iterator referring to the past-the-end element in the allocated memory
        /// </summary>
        /// @return A constant iterator to the element past the end of the allocated memory
        const_iterator cend() const noexcept
        {
            return end();
        }

        /// <summary>
        /// Returns a reverse iterator pointing to the last element in the allocated memory
        /// </summary>
        /// @return A reverse iterator to the reverse beginning of the allocated memory
        reverse_iterator rbegin() noexcept
        {
            return reverse_iterator( end() );
        }

        /// <summary>
        /// Returns a constant reverse iterator pointing to the last element in the allocated memory
        /// </summary>
        /// @return A reverse constant iterator to the reverse beginning of the allocated memory
        const_reverse_iterator rbegin() const noexcept
        {
            return const_reverse_iterator( end() );
        }

        /// <summary>
        /// Returns a constant reverse iterator pointing to the last element in the allocated memory
        /// </summary>
        /// @return A reverse constant iterator to the reverse beginning of the allocated memory
        const_reverse_iterator crbegin() const noexcept
        {
            return rbegin();
        }

        /// <summary>
        /// Returns a reverse iterator pointing to the pass-the-first element in the allocated memory
        /// </summary>
        /// @return A reverse iterator to the reverse end of the allocated memory
        reverse_iterator rend() noexcept
        {
            return reverse_iterator( begin() );
        }

        /// <summary>
        /// Returns a const reverse iterator pointing to the pass-the-first element in the allocated memory
        /// </summary>
        /// @return A const reverse iterator to the reverse end of the allocated memory
        const_reverse_iterator rend() const noexcept
        {
            return const_reverse_iterator( begin() );
        }

        /// <summary>
        /// Returns a const reverse iterator pointing to the pass-the-first element in the allocated memory
        /// </summary>
        /// @return A const reverse iterator to the reverse end of the allocated memory
        const_reverse_iterator crend() const noexcept
        {
            return rend();
        }


        /// <summary>
        /// Returns an \b iterator pointing to the first element in the selected upper diagonal.
        /// </summary>
        /// <param name="index">
        /// The index of the selected upper diagonal.
        /// </param>
        /// @see upper_diag_end
        ///
        /// Example to generate an upper-diagonal matrix of <br>
        /// [ 0, 1, 2, 3 ] <br>
        /// [ 0, 0, 1, 2 ] <br>
        /// [ 0, 0, 0, 1 ] <br>
        /// [ 0, 0, 0, 0 ] <br>
        ///
        /// @code{.cpp}
        /// matrix<unsigned long> mat{ 4, 4, };
        /// for ( unsigned long idx = 0; idx != mat.col(); ++idx )
        ///     std::fill( mat.upper_diag_begin(idx), mat.upper_diag_end(idx), idx );
        /// @endcode
        diag_type upper_diag_begin( const size_type index = 0 ) noexcept
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            return diag_type( zen.begin() + index, static_cast<difference_type>(zen.col() + 1) );
        }

        /// <summary>
        /// Returns an \b iterator referring to the past-the-end element in the last element of the selected upper diagonal.
        /// </summary>
        /// <param name="index">
        /// The index of the selected upper diagonal.
        /// </param>
        /// @see upper_diag_begin
        diag_type upper_diag_end( const size_type index = 0 ) noexcept
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            size_type depth = zen.col() - index;

            if ( zen.row() < depth ) { depth = zen.row(); }

            return diag_type( upper_diag_begin( index ) + static_cast<difference_type>(depth) );
        }

        /// <summary>
        /// Returns a \b constant \b iterator pointing to the first element in the selected upper diagonal.
        /// </summary>
        /// <param name="index">
        /// The index of the selected upper diagonal.
        /// </param>
        /// @see upper_diag_end
        /// Example code to print all the elements in the 2nd upper diagonal <br>
        /// @code{.cpp}
        /// matrix<double> mat{ 5, 5 };
        /// auto generator = [](}{ double val = 0.0; return [=]() mutable { return val += 1.0; return val; }; }();
        /// std::generate( mat.begin(), mat.end(), generator );
        /// std::copy( mat.upper_diag_begin(2), mat.upper_diag_end(2), std::ostream_iterator<double>( std::cout, "\t" ) );
        /// @endcode
        /// At output this will produce <br>
        /// 3	9	15 <br>
        const_diag_type upper_diag_begin( const size_type index = 0 ) const noexcept
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return const_diag_type( zen.begin() + index, zen.col() + 1 );
        }

        /// <summary>
        /// Returns a \b constant \b iterator referring to the past-the-end element in the last element of the selected upper diagonal.
        /// </summary>
        /// <param name="index">
        /// The index of the selected upper diagonal.
        /// </param>
        /// @see upper_diag_begin
        const_diag_type upper_diag_end( const size_type index = 0 ) const noexcept
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            size_type depth = zen.col() - index;

            if ( zen.row() < depth ) { depth = zen.row(); }

            return upper_diag_begin( index ) + depth;
        }


        /// <summary>
        /// Returns a \b constant \b iterator pointing to the first element in the selected upper diagonal.
        /// </summary>
        /// <param name="index">
        /// The index of the selected upper diagonal.
        /// </param>
        /// @see upper_diag_cend
        /// Example code to print all the elements in the 2nd upper diagonal <br>
        /// @code{.cpp}
        /// matrix<double> mat{ 5, 7 };
        /// auto generator = [](}{ double val = 0.0; return [=]() mutable { return val += 1.0; return val; }; }();
        /// std::generate( mat.begin(), mat.end(), generator );
        /// std::copy( mat.upper_diag_cbegin(2), mat.upper_diag_cend(2), std::ostream_iterator<double>( std::cout, "\t" ) );
        /// @endcode
        /// At output this will produce <br>
        /// @code{.cpp}
        /// 3	11	19	27	35
        /// @endcode
        /// where the generate matrix \c mat is <br>
        /// @code{.cpp}
        /// 1	2	3	4	5	6	7
        /// 8	9	10	11	12	13	14
        /// 15	16	17	18	19	20	21
        /// 22	23	24	25	26	27	28
        /// 29	30	31	32	33	34	35
        /// @endcode
        const_diag_type upper_diag_cbegin( const size_type index = 0 ) const noexcept
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return const_diag_type( zen.cbegin() + index, zen.col() + 1 );
        }

        /// <summary>
        /// Returns a \b constant \b iterator referring to the past-the-end element in the last element of the selected upper diagonal.
        /// </summary>
        /// <param name="index">
        /// The index of the selected upper diagonal.
        /// </param>
        /// @see upper_diag_cbegin
        const_diag_type upper_diag_cend( const size_type index = 0 ) const noexcept
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            size_type depth = zen.col() - index;

            if ( zen.row() < depth ) { depth = zen.row(); }

            return upper_diag_cbegin( index ) + depth;
        }

        /// <summary>
        /// Returns a \b reverse \b iterator to the last element in the selected upper diagoanl.
        /// </summary>
        /// <param name="index">
        /// The index of the selected upper diagonal.
        /// </param>
        /// Example to generate a matrix of
        /// @code{.cpp}
        /// 0	0	0	4	0	0	0
        /// 0	0	0	0	3	0	0
        /// 0	0	0	0	0	2	0
        /// 0	0	0	0	0	0	1
        /// 0	0	0	0	0	0	0
        /// @endcode
        ///
        /// @code{.cpp}
        /// auto generator = [](){ double val = 0.0; return [=]() mutable { val += 1.0; return val; }; }();
        /// matrix<double> mat{5,7};
        /// std::fill( mat.begin(), mat.end(), 0.0 );
        /// std::generate( mat.upper_diag_rbegin(3), mat.upper_diag_rend(3), generator );
        /// @endcode
        /// @see upper_diag_rend
        reverse_upper_diag_type upper_diag_rbegin( const size_type index = 0 ) noexcept
        {
            return reverse_upper_diag_type( upper_diag_end( index ) );
        }

        /// <summary>
        /// Returns a \b reverse \b iterator pointing to the pass-the-first element in the selected diagonal.
        /// </summary>
        /// <param name="index">
        /// The index of the selected upper diagonal.
        /// </param>
        /// @see upper_diag_rbegin
        reverse_upper_diag_type upper_diag_rend( const size_type index = 0 ) noexcept
        {
            return reverse_upper_diag_type( upper_diag_begin( index ) );
        }

        /// <summary>
        /// Returns a \b constant \b reverse \b iterator pointing to the pass-the-first element in the selected diagonal.
        /// </summary>
        /// <param name="index">
        /// The index of the selected upper diagonal.
        /// </param>
        /// @see upper_diag_crbegin
        const_reverse_upper_diag_type upper_diag_rbegin( const size_type index = 0 ) const noexcept
        {
            return const_reverse_upper_diag_type( upper_diag_end( index ) );
        }

        const_reverse_upper_diag_type upper_diag_rend( const size_type index = 0 ) const noexcept
        {
            return const_reverse_upper_diag_type( upper_diag_begin( index ) );
        }

        const_reverse_upper_diag_type upper_diag_crbegin( const size_type index = 0 ) const noexcept
        {
            return const_reverse_upper_diag_type( upper_diag_end( index ) );
        }

        const_reverse_upper_diag_type upper_diag_crend( const size_type index = 0 ) const noexcept
        {
            return const_reverse_upper_diag_type( upper_diag_begin( index ) );
        }


        diag_type lower_diag_begin( const size_type index = 0 ) noexcept
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            return diag_type( zen.begin() + index * zen.col(), static_cast<difference_type>(zen.col() + 1) );
        }

        diag_type lower_diag_end( const size_type index = 0 ) noexcept
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            size_type depth = zen.row() - index;

            if ( zen.col() < depth ) { depth = zen.col(); }

            return lower_diag_begin( index ) + static_cast<difference_type>(depth);
        }

        const_diag_type lower_diag_begin( const size_type index = 0 ) const noexcept
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return const_diag_type( zen.begin() + index * zen.col(), zen.col() + 1 );
        }

        const_diag_type lower_diag_end( const size_type index = 0 ) const noexcept
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            size_type depth = zen.row() - index;

            if ( zen.col() < depth ) { depth = zen.col(); }

            return lower_diag_begin( index ) + depth;
        }

        const_diag_type lower_diag_cbegin( const size_type index = 0 ) const noexcept
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return const_diag_type( zen.begin() + index * zen.col(), zen.col() + 1 );
        }

        const_diag_type lower_diag_cend( const size_type index = 0 ) const noexcept
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            size_type depth = zen.row() - index;

            if ( zen.col() < depth ) { depth = zen.col(); }

            return lower_diag_begin( index ) + depth;
        }

        reverse_lower_diag_type lower_diag_rbegin( const size_type index = 0 ) noexcept
        {
            return reverse_lower_diag_type( lower_diag_end( index ) );
        }

        reverse_lower_diag_type lower_diag_rend( const size_type index = 0 ) noexcept
        {
            return reverse_lower_diag_type( lower_diag_begin( index ) );
        }

        const_reverse_lower_diag_type lower_diag_rbegin( const size_type index = 0 ) const noexcept
        {
            return const_reverse_lower_diag_type( lower_diag_end( index ) );
        }

        const_reverse_lower_diag_type lower_diag_rend( const size_type index = 0 ) const noexcept
        {
            return const_reverse_lower_diag_type( lower_diag_begin( index ) );
        }

        const_reverse_lower_diag_type lower_diag_crbegin( const size_type index = 0 ) const noexcept
        {
            return const_reverse_lower_diag_type( lower_diag_end( index ) );
        }

        const_reverse_lower_diag_type lower_diag_crend( const size_type index = 0 ) const noexcept
        {
            return const_reverse_lower_diag_type( lower_diag_begin( index ) );
        }

        diag_type diag_begin( const difference_type index = 0 ) noexcept
        {
            if ( index > 0 ) { return upper_diag_begin( static_cast<size_type>(index) ); }

            return lower_diag_begin( static_cast<size_type>(-index) );
        }

        diag_type diag_end( const difference_type index = 0 ) noexcept
        {
            if ( index > 0 ) { return upper_diag_end( static_cast<size_type>(index) ); }

            return lower_diag_end( static_cast<size_type>(-index) );
        }

        const_diag_type diag_begin( const difference_type index = 0 ) const noexcept
        {
            if ( index > 0 ) { return upper_diag_begin( index ); }

            return lower_diag_begin( -index );
        }

        const_diag_type diag_end( const difference_type index = 0 ) const noexcept
        {
            if ( index > 0 ) { return upper_diag_end( index ); }

            return lower_diag_end( -index );
        }

        const_diag_type diag_cbegin( const difference_type index = 0 ) const noexcept
        {
            if ( index > 0 ) { return upper_diag_cbegin( index ); }

            return lower_diag_cbegin( -index );
        }

        const_diag_type diag_cend( const difference_type index = 0 ) const noexcept
        {
            if ( index > 0 ) { return upper_diag_cend( index ); }

            return lower_diag_cend( -index );
        }

        reverse_diag_type diag_rbegin( const difference_type index = 0 ) noexcept
        {
            if ( index > 0 ) { return upper_diag_rbegin( index ); }

            return lower_diag_rbegin( -index );
        }

        reverse_diag_type diag_rend( const difference_type index = 0 ) noexcept
        {
            if ( index > 0 ) { return upper_diag_rend( index ); }

            return lower_diag_rend( -index );
        }

        const_reverse_diag_type diag_rbegin( const difference_type index = 0 ) const noexcept
        {
            if ( index > 0 ) { return upper_diag_rbegin( index ); }

            return lower_diag_rbegin( -index );
        }

        const_reverse_diag_type diag_rend( const difference_type index = 0 ) const noexcept
        {
            if ( index > 0 ) { return upper_diag_rend( index ); }

            return lower_diag_rend( -index );
        }

        const_reverse_diag_type diag_crbegin( const difference_type index = 0 ) const noexcept
        {
            if ( index > 0 ) { return upper_diag_crbegin( index ); }

            return lower_diag_crbegin( -index );
        }

        const_reverse_diag_type diag_crend( const difference_type index = 0 ) const noexcept
        {
            if ( index > 0 ) { return upper_diag_crend( index ); }

            return lower_diag_crend( -index );
        }


        anti_diag_type upper_anti_diag_begin( const size_type index = 0 ) noexcept
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            return anti_diag_type( zen.begin() + zen.col() - index - 1, zen.col() - 1 );
        }

        anti_diag_type upper_anti_diag_end( const size_type index = 0 ) noexcept
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            size_type depth = zen.col() - index;

            if ( zen.row() < depth ) { depth = zen.row(); }

            return upper_anti_diag_begin( index ) + depth;
        }

        const_anti_diag_type upper_anti_diag_begin( const size_type index = 0 ) const noexcept
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return const_anti_diag_type( zen.begin() + zen.col() - index - 1, zen.col() - 1 );
        }

        const_anti_diag_type upper_anti_diag_end( const size_type index = 0 ) const noexcept
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            size_type depth = zen.col() - index;

            if ( zen.row() < depth ) { depth = zen.row(); }

            return upper_anti_diag_begin( index ) + depth;
        }

        const_anti_diag_type upper_anti_diag_cbegin( const size_type index = 0 ) const noexcept
        {
            return upper_anti_diag_begin( index );
        }

        const_anti_diag_type upper_anti_diag_cend( const size_type index = 0 ) const noexcept
        {
            return upper_anti_diag_end( index );
        }

        reverse_anti_diag_type upper_anti_diag_rbegin( const size_type index = 0 ) noexcept
        {
            return reverse_anti_diag_type( upper_anti_diag_end( index ) );
        }

        reverse_anti_diag_type upper_anti_diag_rend( const size_type index = 0 ) noexcept
        {
            return reverse_anti_diag_type( upper_anti_diag_begin( index ) );
        }

        const_reverse_anti_diag_type upper_anti_diag_rbegin( const size_type index = 0 ) const noexcept
        {
            return const_reverse_anti_diag_type( upper_anti_diag_end( index ) );
        }

        const_reverse_anti_diag_type upper_anti_diag_rend( const size_type index = 0 ) const noexcept
        {
            return const_reverse_anti_diag_type( upper_anti_diag_begin( index ) );
        }

        const_reverse_anti_diag_type upper_anti_diag_crbegin( const size_type index = 0 ) const noexcept
        {
            return upper_anti_diag_rbegin( index );
        }

        const_reverse_anti_diag_type upper_anti_diag_crend( const size_type index = 0 ) const noexcept
        {
            return upper_anti_diag_rend( index );
        }


        anti_diag_type lower_anti_diag_begin( const size_type index = 0 ) noexcept
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            return anti_diag_type( zen.begin() + ( zen.col() * ( index + 1 ) ) - 1, zen.col() - 1 );
        }

        anti_diag_type lower_anti_diag_end( const size_type index = 0 ) noexcept
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            size_type depth = zen.row() - index;

            if ( zen.col() < depth ) { depth = zen.col(); }

            return lower_anti_diag_begin( index ) + depth;
        }

        const_anti_diag_type lower_anti_diag_begin( const size_type index = 0 ) const noexcept
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return const_anti_diag_type( zen.begin() + ( zen.col() * ( index + 1 ) ) - 1, zen.col() - 1 );
        }

        const_anti_diag_type lower_anti_diag_end( const size_type index = 0 ) const noexcept
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            size_type depth = zen.row() - index;

            if ( zen.col() < depth ) { depth = zen.col(); }

            return lower_anti_diag_begin( index ) + depth;
        }

        const_anti_diag_type lower_anti_diag_cbegin( const size_type index = 0 ) const noexcept
        {
            return lower_anti_diag_begin( index );
        }

        const_anti_diag_type lower_anti_diag_cend( const size_type index = 0 ) const noexcept
        {
            return lower_anti_diag_end( index );
        }

        reverse_anti_diag_type lower_anti_diag_rbegin( const size_type index = 0 ) noexcept
        {
            return reverse_anti_diag_type( lower_anti_diag_end( index ) );
        }

        reverse_anti_diag_type lower_anti_diag_rend( const size_type index = 0 ) noexcept
        {
            return reverse_anti_diag_type( lower_anti_diag_begin( index ) );
        }

        const_reverse_anti_diag_type lower_anti_diag_rbegin( const size_type index = 0 ) const noexcept
        {
            return const_reverse_anti_diag_type( lower_anti_diag_end( index ) );
        }

        const_reverse_anti_diag_type lower_anti_diag_rend( const size_type index = 0 ) const noexcept
        {
            return const_reverse_anti_diag_type( lower_anti_diag_begin( index ) );
        }

        const_reverse_anti_diag_type lower_anti_diag_crbegin( const size_type index = 0 ) const noexcept
        {
            return lower_anti_diag_rbegin( index );
        }

        const_reverse_anti_diag_type lower_anti_diag_crend( const size_type index = 0 ) const noexcept
        {
            return lower_anti_diag_rend( index );
        }

        anti_diag_type anti_diag_begin( const difference_type index = 0 ) noexcept
        {
            if ( index > 0 ) { return upper_anti_diag_begin( index ); }

            return lower_anti_diag_begin( -index );
        }

        anti_diag_type anti_diag_end( const difference_type index = 0 ) noexcept
        {
            if ( index > 0 ) { return upper_anti_diag_end( index ); }

            return lower_anti_diag_end( -index );
        }

        const_anti_diag_type anti_diag_begin( const difference_type index = 0 ) const noexcept
        {
            if ( index > 0 ) { return upper_anti_diag_begin( index ); }

            return lower_anti_diag_begin( -index );
        }

        const_anti_diag_type anti_diag_end( const difference_type index = 0 ) const noexcept
        {
            if ( index > 0 ) { return upper_anti_diag_end( index ); }

            return lower_anti_diag_end( -index );
        }

        const_anti_diag_type anti_diag_cbegin( const difference_type index = 0 ) const noexcept
        {
            if ( index > 0 ) { return upper_anti_diag_cbegin( index ); }

            return lower_anti_diag_cbegin( -index );
        }

        const_anti_diag_type anti_diag_cend( const difference_type index = 0 ) const noexcept
        {
            if ( index > 0 ) { return upper_anti_diag_cend( index ); }

            return lower_anti_diag_cend( -index );
        }

        reverse_anti_diag_type anti_diag_rbegin( const difference_type index = 0 ) noexcept
        {
            if ( index > 0 ) { return upper_anti_diag_rbegin( index ); }

            return lower_anti_diag_rbegin( -index );
        }

        reverse_anti_diag_type anti_diag_rend( const difference_type index = 0 ) noexcept
        {
            if ( index > 0 ) { return upper_anti_diag_rend( index ); }

            return lower_anti_diag_rend( -index );
        }

        const_reverse_anti_diag_type anti_diag_rbegin( const difference_type index = 0 ) const noexcept
        {
            if ( index > 0 ) { return upper_anti_diag_rbegin( index ); }

            return lower_anti_diag_rbegin( -index );
        }

        const_reverse_anti_diag_type anti_diag_rend( const difference_type index = 0 ) const noexcept
        {
            if ( index > 0 ) { return upper_anti_diag_rend( index ); }

            return lower_anti_diag_rend( -index );
        }

        const_reverse_anti_diag_type anti_diag_crbegin( const difference_type index = 0 ) const noexcept
        {
            if ( index > 0 ) { return upper_anti_diag_crbegin( index ); }

            return lower_anti_diag_crbegin( -index );
        }

        const_reverse_anti_diag_type anti_diag_crend( const difference_type index = 0 ) const noexcept
        {
            if ( index > 0 ) { return upper_anti_diag_crend( index ); }

            return lower_anti_diag_crend( -index );
        }


        /// <summary>
        /// Gets an iterator pointing to the first element in the selected row;
        /// No boundary check is performed, thus this iterator is not necessarily valid if \p index is greate_or_equal than the \p row of this matrix.
        /// </summary>
        /// <param name="index">
        /// The index of the selected row, starting from 0
        /// </param>
        /// @returns An iterator pointing to the first element of the row
        /// @see row_end
        ///
        /// Example usage to generate a matrix of <br>
        /// [ 1.0 1.0 1.0 ] <br>
        /// [ 2.0 2.0 2.0 ] <br>
        /// [ 3.0 3.0 3.0 ] <br>
        /// @code{.cpp}
        ///
        /// matrix<double> mat{3, 3};
        /// for ( unsigned long index = 0; index != mat.row(); ++index )
        /// std::fill( mat.row_begin(index), mat.row_end(index), 1.0*(index+1) );
        ///
        /// @endcode
        row_type row_begin( const size_type index ) noexcept
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            return row_type( zen.begin() + index * zen.col() );
        }

        /// <summary>
        /// Gets an iterator referring to the past-the-end element in the last element of the selected row.
        /// No boundary check is performed, thus this iterator is not necessarily valid if \p index is greate_or_equal than the \p row of this matrix.
        /// </summary>
        /// <param name="index">
        /// The index of the selected row, starting from 0
        /// </param>
        /// @returns An iterator pointing to the past-the-end in the last element of the row.
        /// @see row_begin
        /// @see row_cend
        row_type row_end( const size_type index ) noexcept
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            return row_begin( index ) + zen.col();
        }

        /// <summary>
        /// Gets a constant iterator pointing to the first element in the selected row;
        /// No boundary check is performed, thus this iterator is not necessarily valid if \p index is greate_or_equal than the \p row of this matrix.
        /// </summary>
        /// <param name="index">
        /// The index of the selected row, starting from 0
        /// </param>
        /// @returns An constant iterator pointing to the first element of the row
        /// @see row_end
        /// @see row_cbegin
        ///
        /// Example usage to print a matrix
        /// @code{.cpp}
        ///
        /// matrix<double> mat{3, 3};
        /// //...
        /// for ( unsigned long index = 0; index != mat.row(); ++index )
        /// {
        ///     std::copy( mat.row_begin(index), mat.row_end(index), std::ostream_iterator<double>( std::cout, " " ) );
        ///     std::cout << "\n";
        /// }
        ///
        /// @endcode
        const_row_type row_begin( const size_type index ) const noexcept
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return const_row_type( zen.begin() + index * zen.col() );
        }

        /// <summary>
        /// Gets a constant iterator referring to the past-the-end element in the last element of the selected row.
        /// No boundary check is performed, thus this iterator is not necessarily valid if \p index is greate_or_equal than the \p row of this matrix.
        /// </summary>
        /// <param name="index">
        /// The index of the selected row, starting from 0
        /// </param>
        /// @returns A constant iterator pointing to the past-the-end in the last element of the row.
        /// @see row_begin
        const_row_type row_end( const size_type index ) const noexcept
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return row_begin( index ) + zen.col();
        }


        /// <summary>
        /// Gets a constant iterator pointing to the first element in the selected row;
        /// No boundary check is performed, thus this iterator is not necessarily valid if \p index is greate_or_equal than the \p row of this matrix.
        /// </summary>
        /// <param name="index">
        /// The index of the selected row, starting from 0
        /// </param>
        /// @returns An constant iterator pointing to the first element of the row
        /// @see row_begin
        ///
        /// Example usage to print a matrix
        /// @code{.cpp}
        ///
        /// matrix<double> mat{3, 3};
        /// //...
        /// for ( unsigned long index = 0; index != mat.row(); ++index )
        /// {
        ///     std::copy( mat.row_cbegin(index), mat.row_cend(index), std::ostream_iterator<double>( std::cout, " " ) );
        ///     std::cout << "\n";
        /// }
        ///
        /// @endcode
        const_row_type row_cbegin( const size_type index ) const noexcept
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return const_row_type( zen.begin() + index * zen.col());
        }


        /// <summary>
        /// Gets a constant iterator referring to the past-the-end element in the last element of the selected row.
        /// </summary>
        /// No boundary check is performed, thus this iterator is not necessarily valid if \p index is greate_or_equal than the \p row of this matrix.
        /// <param name="index">
        /// The index of the selected row, starting from 0
        /// </param>
        /// @returns A constant iterator pointing to the past-the-end in the last element of the row.
        /// @see row_end
        const_row_type row_cend( const size_type index ) const noexcept
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return row_begin( index ) + zen.col();
        }

        /// <summary>
        /// Returns a reverse iterator to the last element in the selected row.
        /// </summary>
        /// No boundary check is performed, thus this iterator is not necessarily valid if \p index is greate_or_equal than the \p row of this matrix.
        /// <param name="index">
        /// The index of the selected row, starting from 0
        /// </param>
        /// @see row_rend
        ///
        /// Example usage to generate a matrix of <br>
        /// [ 2.0 1.0 0.0 ] <br>
        /// [ 5.0 4.0 3.0 ] <br>
        /// [ 8.0 7.0 6.0 ] <br>
        /// @code{.cpp}
        ///
        /// matrix<double> mat{ 3, 3 };
        /// auto generator = [](){ double start = 0.0; return [start]() mutable { return start++; }; } ();
        /// for ( unsigned long index = 0; index != mat.row(); ++index )
        ///     std::generate( mat.row_rbegin(index), mat.row_rend(index), generator );
        ///
        /// @endcode
        reverse_row_type row_rbegin( const size_type index ) noexcept
        {
            return reverse_row_type( row_end( index ) );
        }

        /// <summary>
        /// Returns a reverse iterator pointing to the pass-the-first element in the selected row.
        /// </summary>
        /// No boundary check is performed, thus this iterator is not necessarily valid if \p index is greate_or_equal than the \p row of this matrix.
        /// <param name="index">
        /// The index of the selected row, starting from 0
        /// </param>
        /// @see row_rbegin
        reverse_row_type row_rend( const size_type index ) noexcept
        {
            return reverse_row_type( row_begin( index ) );
        }

        /// <summary>
        /// Returns a constant reverse iterator to the last element in the selected row.
        /// </summary>
        /// No boundary check is performed, thus this iterator is not necessarily valid if \p index is greate_or_equal than the \p row of this matrix.
        /// <param name="index">
        /// The index of the selected row, starting from 0
        /// </param>
        const_reverse_row_type row_rbegin( const size_type index ) const noexcept
        {
            return const_reverse_row_type( row_end( index ) );
        }

        /// <summary>
        /// Returns a constant reverse iterator pointing to the pass-the-first element in the selected row.
        /// </summary>
        /// No boundary check is performed, thus this iterator is not necessarily valid if \p index is greate_or_equal than the \p row of this matrix.
        /// <param name="index">
        /// The index of the selected row, starting from 0
        /// </param>
        /// @see row_rbegin
        const_reverse_row_type row_rend( const size_type index ) const noexcept
        {
            return const_reverse_row_type( row_begin( index ) );
        }

        /// <summary>
        /// Returns a constant reverse iterator to the last element in the selected row.
        /// </summary>
        /// No boundary check is performed, thus this iterator is not necessarily valid if \p index is greate_or_equal than the \p row of this matrix.
        /// <param name="index">
        /// The index of the selected row, starting from 0
        /// </param>
        /// @see row_rbegin
        const_reverse_row_type row_crbegin( const size_type index ) const noexcept
        {
            return const_reverse_row_type( row_end( index ) );
        }

        /// <summary>
        /// Returns a constant reverse iterator pointing to the pass-the-first element in the selected row.
        /// </summary>
        /// No boundary check is performed, thus this iterator is not necessarily valid if \p index is greate_or_equal than the \p row of this matrix.
        /// <param name="index">
        /// The index of the selected row, starting from 0
        /// </param>
        /// @see row_rend
        const_reverse_row_type row_crend( const size_type index ) const noexcept
        {
            return const_reverse_row_type( row_begin( index ) );
        }

        col_type col_begin( const size_type index ) noexcept
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            return col_type( zen.begin() + index, static_cast<difference_type>(zen.col()) );
        }

        col_type col_end( const size_type index ) noexcept
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            return col_begin( index ) + static_cast<difference_type>(zen.row());
        }

        const_col_type col_begin( const size_type index ) const noexcept
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return const_col_type( zen.begin() + index, static_cast<difference_type>(zen.col()) );
        }

        const_col_type col_end( const size_type index ) const noexcept
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return col_begin( index ) + zen.row();
        }

        const_col_type col_cbegin( const size_type index ) const noexcept
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return const_col_type( zen.begin() + index, zen.col() );
        }

        const_col_type col_cend( const size_type index ) const noexcept
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return col_begin( index ) + zen.row();
        }

        reverse_col_type col_rbegin( const size_type index ) noexcept
        {
            return reverse_col_type( col_end( index ) );
        }

        reverse_col_type col_rend( const size_type index ) noexcept
        {
            return reverse_col_type( col_begin( index ) );
        }

        const_reverse_col_type col_rbegin( const size_type index ) const noexcept
        {
            return const_reverse_col_type( col_end( index ) );
        }

        const_reverse_col_type col_rend( const size_type index ) const noexcept
        {
            return const_reverse_col_type( col_begin( index ) );
        }

        const_reverse_col_type col_crbegin( const size_type index ) const noexcept
        {
            return const_reverse_col_type( col_end( index ) );
        }

        const_reverse_col_type col_crend( const size_type index ) const noexcept
        {
            return const_reverse_col_type( col_begin( index ) );
        }

    };//struct

}//namespace

#endif

