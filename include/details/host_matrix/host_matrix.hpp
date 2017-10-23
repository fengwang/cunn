#ifndef SDALK4309IAFDOIJHVNMKASFKLJ948YALOASDJKL230AOIFHASDLFJKAH1FASDSAUI9H4VD
#define SDALK4309IAFDOIJHVNMKASFKLJ948YALOASDJKL230AOIFHASDLFJKAH1FASDSAUI9H4VD

#include "./typedef.hpp"
#include "./io.hpp"
#include "./iterator.hpp"
#include "./minus_equal_operator.hpp"
#include "./host_allocator.hpp"
#include "./bracket_operator.hpp"
#include "./transpose.hpp"

namespace cunn
{

    template< typename Type, typename Allocator = host_allocator<Type> >
    struct host_matrix :
        crtp_io<host_matrix<Type, Allocator>, Type, Allocator >,
        crtp_iterator<host_matrix<Type, Allocator>, Type, Allocator >,
        crtp_bracket_operator<host_matrix<Type, Allocator>, Type, Allocator >,
        crtp_transpose<host_matrix<Type, Allocator>, Type, Allocator >,
        crtp_minus_equal_operator<host_matrix<Type, Allocator>, Type, Allocator >
    {
        typedef host_matrix                                                         self_type;
        typedef crtp_typedef<Type, Allocator>                                       type_proxy_type;
        typedef typename type_proxy_type::size_type                                 size_type;
        typedef typename type_proxy_type::value_type                                value_type;
        typedef typename type_proxy_type::pointer                                   pointer;
        typedef typename type_proxy_type::const_pointer                             const_pointer;
        typedef typename type_proxy_type::allocator_type                            allocator_type;

        size_type               row_;                   ///< row of the host_matrix
        size_type               col_;                   ///< column of the host_matrix
        allocator_type          alloc_;                 ///< explicit allocator responsible for the memory allocated, may holding states
        pointer                 data_;                  ///< pointer to the allocated memory

        /// <summary>
        /// Default constructor.
        /// </summary>
        /// <param name="row">
        /// Row of the host_matrix.
        /// </param>
        /// <param name="col">
        /// Column of the host_matrix.
        /// </param>
        /// <param name="alloc">
        /// The allocator to use for all memory constructions of this host_matrix.
        /// </param>
        host_matrix( size_type const row = 0, size_type const col = 0, allocator_type const& alloc = allocator_type{} )
            : row_{row}, col_{col}, alloc_{alloc}, data_{nullptr}
        {
            if ( size() )
                data_ = alloc_.allocate( size() );
        }

        /// <summary>
        /// constructor
        /// </summary>
        /// <param name="other_alloc">
        /// The allocator to use for all memory constructions of this host_matrix.
        /// </param>
        host_matrix( allocator_type const& other_alloc ) : row_{0}, col_{0}, alloc_{other_alloc}, data_{nullptr} {}

        /// <summary>
        /// copy constructor
        /// </summary>
        /// <param name="other">
        /// Another host_matrix to construct from.
        /// </param>
        /// Example
        /// @code{.cpp}
        /// host_matrix<double> ma;
        /// //...
        /// host_matrix<double> mb{ma};
        /// @endcode
        host_matrix( self_type const& other ) : alloc_{other.alloc_}, data_{nullptr}
        {
            copy( other );
        }

        /// <summary>
        /// copy constructor with customer allocator
        /// </summary>
        /// <param name="other">
        /// Another host_matrix to construct from.
        /// </param>
        /// <param name="other_alloc">
        /// The allocator to use for all memory constructions of this host_matrix.
        /// </param>
        host_matrix( self_type const& other, allocator_type const& other_alloc ) : alloc_{other_alloc}, data_{nullptr}
        {
            copy( other );
        }

        /// <summary>
        /// copy constructor
        /// </summary>
        /// <param name="other">
        /// Another kinde of a host_matrix to construct from.
        /// </param>
        /// Example
        /// @code{.cpp}
        /// host_matrix<float> ma;
        /// //...
        /// host_matrix<double> mb{ma};
        /// @endcode
        template< typename Other_Type, typename Other_Allocator >
        host_matrix( host_matrix<Other_Type, Other_Allocator> const& other )
        : alloc_{ typename std::allocator_traits<decltype(other.get_allocator())>:: template rebind_alloc<value_type>{other.get_allocator()} }
        {
            resize( other.row(), other.col() );
            std::copy( other.begin(), other.end(), data() );
        }

        /// <summary>
        /// copy constructor
        /// </summary>
        /// <param name="other">
        /// Another kinde of a host_matrix to construct from.
        /// </param>
        /// Example
        template< typename Other_Type, typename Other_Allocator >
        host_matrix( host_matrix<Other_Type, Other_Allocator> const& other, allocator_type const& other_alloc ) : alloc_{other_alloc}, data_{nullptr}
        {
            resize( other.row(), other.col() );
            std::copy( other.begin(), other.end(), data() );
        }

        /// <summary>
        /// Destructor.
        /// </summary>
        ~host_matrix()
        {
            clear();
        }

        /// <summary>
        ///  Default move constructor.
        /// </summary>
        host_matrix( self_type && other ) : host_matrix{ 0, 0 }
        {
            swap(other);
        }

        /// <summary>
        /// copy assignment
        /// </summary>
        /// <param name="rhs">
        /// Another host_matrix to construct from.
        /// </param>
        /// @see copy
        self_type& operator = ( const self_type& rhs )
        {
            copy( rhs );
            return *this;
        }

        /// <summary>
        /// copy assignment
        /// </summary>
        /// <param name="rhs">
        /// Another kinde of a host_matrix to construct from.
        /// </param>
        template< typename Other_Type, typename Other_Allocator >
        self_type& operator = ( host_matrix<Other_Type, Other_Allocator> const& other )
        {
            resize( other.row(), other.col() );
            std::copy( other.begin(), other.end(), data() );
            return *this;
        }

        /// <summary>
        /// default move assignment
        /// </summary>
        self_type& operator = ( self_type && other )
        {
            clear();
            swap(other);
            return *this;
        }

        /// <summary>
        /// Returns the allocator associated with this host_matrix.
        /// </summary>
        /// @return The associated allocator, #alloc_.
        allocator_type get_allocator() const noexcept
        {
            return alloc_;
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
        /// Remove all elements in the host_matrix.Invalidates any references, pointers, iterators referring to host_matrix elements, or iterators past the end.
        /// <summary>
        /// Example
        /// @code{.cpp}
        /// host_matrix<double> m{ 12, 34 };
        /// m.clear();
        /// assert( m.size() == 0 );
        /// assert( m.data() == nullptr );
        /// @endcode
        void clear()
        {
            if ( data_ )
            {
                alloc_.deallocate(data_, size());
                data_ = nullptr;
            }
            row_ = 0;
            col_ = 0;
        }

        /// <summary>
        /// get the column of the host_matrix
        /// </summary>
        /// @return Returns the column of the host_matrix
        ///
        /// Example:
        /// @code{.cpp}
        /// host_matrix<int> m{3, 4};
        /// assert( m.col() == 4 );
        /// @endcode
        size_type col() const noexcept
        {
            return col_;
        }

        /// <summary>
        /// copy the contents of another host_matrix
        /// </summary>
        /// <param name="rhs">
        /// Another host_matrix instance, not necessarily of the same type with the host host_matrix.
        /// </param>
        /// Example
        /// @code{.cpp}
        /// host_matrix<int> a{ 3, 2 };
        /// host_matrix<int> b;
        /// b.copy( a );
        /// @endcode
        void copy( self_type const& rhs )
        {
            resize( rhs.row(), rhs.col() );
            std::copy( rhs.begin(), rhs.end(), data() );
        }

        /// <summary>
        /// Resize the host host_matrix with a new row and a new column, the memory is allocated but uninitialized.
        /// </summary>
        /// <param name="new_row">
        /// New row size.
        /// </param>
        /// <param name="new_col">
        /// New column size.
        /// </param>
        /// @code{.cpp}
        /// host_matrix<int> mi;
        /// mi.resize( 7, 3 );
        /// assert( mi.row() == 7 );
        /// assert( mi.col() == 3 );
        /// @endcode
        self_type& resize( const size_type new_row, const size_type new_col )
        {
            if ( size() == new_row * new_col )
            {
                row_ = new_row;
                col_ = new_col;
                return *this;
            }

            self_type ans{ new_row, new_col, alloc_ };
            swap( ans );
            return *this;
        }

        /// <summary>
        /// get the row of the host_matrix
        /// </summary>
        /// @return Returns the row of the host_matrix, #row_
        ///
        /// Example
        /// @code{.cpp}
        /// host_matrix<double> m{ 12, 34 };
        /// asser( m.row() == 12 );
        /// @endcode
        auto row() const noexcept
        {
            return row_;
        }

        /// <summary>
        /// get the size of the host_matrix
        /// <summary>
        /// @return Returns the size of the host_matrix
        /// @code{.cpp}
        /// host_matrix<float> mf{ 2, 4 };
        /// assert( mf.size() == 2*4 );
        /// @endcode
        auto size() const noexcept
        {
            return row() * col();
        }


        /// <summary>
        /// Swap the contents of two matrices
        /// </summary>
        /// <param name="other">
        /// Anther host_matrix to be exchanged with the host host_matrix, if it is identical to the host host_matrix, the behaviour is undefined.
        /// </param>
        ///
        /// Example usage to swap two matrices
        /// @code{.cpp}
        /// //...
        /// host_matrix<double> m1{ 2, 3 }
        /// host_matrix<double> m2{ 3, 2 }
        /// //...
        /// m1.swap( m2 );
        /// //...
        /// @endcode
        void swap( self_type& other )
        {
            std::swap( row_, other.row_ );
            std::swap( col_, other.col_ );
            std::swap( alloc_, other.alloc_ );
            std::swap( data_, other.data_ );
        }


        /// <summary>
        /// Check if the host_matrix is empty or not
        /// </summary>
        /// @return true if empty, otherwise false
        bool empty() const noexcept
        {
            return size() == 0;
        }

    };//struct

}//namespace

#endif

