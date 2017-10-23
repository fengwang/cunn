#ifndef FHYPPGNXFCQLVDYJANMBXYHNVIDSDWQMWNDOTBIIRKCQUPYYBACHWGGSKLMBMLPWJFIFHJJBM
#define FHYPPGNXFCQLVDYJANMBXYHNVIDSDWQMWNDOTBIIRKCQUPYYBACHWGGSKLMBMLPWJFIFHJJBM

#include "../utility.hpp"
#include "./cuda.hpp"

namespace cunn
{

    template< typename T > struct cuda_allocator;

    template<>
    struct cuda_allocator<void>
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
            typedef cuda_allocator<U>        other;
        };
    };

    template< typename T >
    struct cuda_allocator
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
            typedef cuda_allocator<U>        other;
        };

        cuda_allocator() noexcept : device_id{ cuda::get() } {}
        explicit cuda_allocator( int device_id_ ) noexcept : device_id( device_id_ ) {}

        template< typename U > cuda_allocator( cuda_allocator<U> const& other_ ) noexcept : device_id( other_.device_id ) {}

        template< typename U > cuda_allocator( cuda_allocator<U>&& other_ ) noexcept : device_id( other_.device_id ) {}

        template< typename U > cuda_allocator& operator = ( cuda_allocator<U> const& other_ ) noexcept { device_id = other_.device_id; return *this; }

        template< typename U > cuda_allocator& operator = ( cuda_allocator<U>&& other_ ) noexcept { device_id = other_.device_id; return *this; }

        pointer allocate( size_type n_ ) noexcept
        {
            activate();
            return cuda::allocate<value_type>( n_ );
        }

        void deallocate( pointer p_, [[maybe_unused]] size_type n_ = 0 ) noexcept
        {
            activate();
            cuda::deallocate( p_ );
        }

        void activate() const
        {
            cuda::set( device_id );
        }

        int device_id;
    };

}//namespace

#endif//FHYPPGNXFCQLVDYJANMBXYHNVIDSDWQMWNDOTBIIRKCQUPYYBACHWGGSKLMBMLPWJFIFHJJBM

