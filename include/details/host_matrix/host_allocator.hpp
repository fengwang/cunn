#ifndef CUDA_ALLOCATOR_HPP_INCLUDED_DOPIJ43908USDAFLKJAS89U4398UJDSAFLKJSFDA98J3F
#define CUDA_ALLOCATOR_HPP_INCLUDED_DOPIJ43908USDAFLKJAS89U4398UJDSAFLKJSFDA98J3F

#include "../utility.hpp"
#include "../cuda/cuda.hpp"

namespace cunn
{

    template< typename T > struct host_allocator;

    template<>
    struct host_allocator<void>
    {
        typedef     std::size_t         size_type;
        typedef     std::ptrdiff_t      difference_type;
        typedef     void*               pointer;
        typedef     const void*         const_pointer;
        typedef     void                value_type;
        typedef     std::true_type      propagate_on_container_move_assignment;
        typedef     std::true_type      is_always_equal;

        template< typename U >
        struct rebind
        {
            typedef host_allocator<U>        other;
        };
    };

    template< typename T >
    struct host_allocator
    {
        typedef     std::size_t         size_type;
        typedef     std::ptrdiff_t      difference_type;
        typedef     T                   value_type;
        typedef     value_type*         pointer;
        typedef     const value_type*   const_pointer;
        typedef     std::true_type      propagate_on_container_move_assignment;
        typedef     std::false_type     is_always_equal;

        template< typename U >
        struct rebind
        {
            typedef host_allocator<U>        other;
        };

        host_allocator() noexcept {}
        host_allocator( host_allocator const& ) = default;
        host_allocator( host_allocator&& ) = default;
        host_allocator& operator = ( host_allocator const& ) = default;
        host_allocator& operator = ( host_allocator&& ) = default;

        template< typename U > host_allocator( host_allocator<U> const& other_ ) noexcept {}
        template< typename U > host_allocator( host_allocator<U>&& other_ ) noexcept {}
        template< typename U > host_allocator& operator = ( host_allocator<U> const& other_ ) noexcept { return *this; }
        template< typename U > host_allocator& operator = ( host_allocator<U>&& other_ ) noexcept { return *this; }

        pointer allocate( size_type n_ ) noexcept
        {
            return host_allocate<value_type>( n_ );
        }

        void deallocate( pointer p_, [[maybe_unused]] size_type n_ = 0 ) noexcept
        {
            host_deallocate( p_ );
        }

    };

}//namespace

#endif

