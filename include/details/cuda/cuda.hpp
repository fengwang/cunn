#ifndef DWJUWOBYHOTAGIAAHOTWEGPBLECMWYVTUVGJIDGRUCQRVQEVWSGTVWIICNONGWSPSRIOMJUKQ
#define DWJUWOBYHOTAGIAAHOTWEGPBLECMWYVTUVGJIDGRUCQRVQEVWSGTVWIICNONGWSPSRIOMJUKQ

extern "C"
{
    unsigned long long get_global_mem( int );
    int get_capability( int );
    int devices_count();
    int get_device();
    void set_device( int );
    void reset_device();
    void synchronize_device();
    void cuda_allocate( void** p, unsigned long n );
    void cuda_deallocate( void* p );
    void cuda_host_allocate( void** p, unsigned long n );
    void cuda_host_deallocate( void* p );
    void cuda_memcopy_device_to_device( const void* src, unsigned long n, void* dst );
    void cuda_memcopy_device_to_host( const void* src, unsigned long n, void* dst );
    void cuda_memcopy_host_to_device( const void* src, unsigned long n, void* dst );
    int cuda_device_enable_peer_access_from_to( int device, int peer_device );
    void cuda_device_disable_peer_access_from_to( int device, int peer_device );
    void cuda_uniform_initialize( float* address, unsigned long length );
}

namespace cunn
{
    // get CUdA capability, ( Major * 100 + Minor ), 100, 101, 102, 103, 200, 201, .... 701
    int capability( int id )
    {
        return get_capability( id );
    }

    // return total amount of global memory, in bytes
    inline unsigned long long global_memory( int id )
    {
        return get_global_mem( id );
    }

    // return available GPUS
    inline int get_devices()
    {
        return devices_count();
    }

    // return current working GPU id
    inline int get()
    {
        return get_device();
    }

    // set working GPU id
    inline void set( int id )
    {
        int current_id = get();
        if ( current_id != id )
            set_device( id );
    }

    inline void reset()
    {
        reset_device();
    }

    inline void synchronize()
    {
        synchronize_device();
    }

    template< typename T >
    T* allocate( unsigned long n )
    {
        T* ans;
        cuda_allocate( reinterpret_cast<void**>(&ans), n*sizeof(T) );
        return ans;
    }

    template< typename T >
    void deallocate( T* ptr )
    {
        cuda_deallocate( reinterpret_cast<void*>(ptr) );
    }

    // host pinned allocate
    template< typename T >
    T* host_allocate( unsigned long n )
    {
        T* ans;
        cuda_host_allocate( reinterpret_cast<void**>(&ans), n*sizeof(T) );
        return ans;
    }

    // host pinned deallocate
    template< typename T >
    void host_deallocate( T* ptr )
    {
        cuda_host_deallocate( reinterpret_cast<void*>(ptr) );
    }

    template< typename T >
    void host_to_device( T const* host_begin, T const* host_end, T* device_begin )
    {
        unsigned long const n = sizeof(T)*(host_end-host_begin);
        cuda_memcopy_host_to_device( reinterpret_cast<const void*>(host_begin), n, reinterpret_cast<void*>(device_begin) );
    }

    template< typename T >
    void device_to_host( T* device_begin, T* device_end, T* host_begin )
    {
        unsigned long const n = sizeof(T)*(device_end-device_begin);
        cuda_memcopy_device_to_host( reinterpret_cast<const void*>(device_begin), n, reinterpret_cast<void*>(host_begin) );
    }

    template< typename T >
    void device_to_device( T* device_begin, T* device_end, T* host_begin )
    {
        unsigned long const n = sizeof(T)*(device_end-device_begin);
        cuda_memcopy_device_to_device( reinterpret_cast<const void*>(device_begin), n, reinterpret_cast<void*>(host_begin) );
    }

    struct peer_access
    {
        int device_1_;
        int device_2_;

        // when enableing device 'device_1', direct access memory allocated in device 'device_2'
        peer_access( int device_1 = -1, int device_2 = -1 ) noexcept : device_1_( device_1 ), device_2_( device_2 )
        {
            if ( device_1_ != device_2_ && device_1_ != -1 && device_2_ != -1 )
                cuda_device_enable_peer_access_from_to( device_1_, device_2_ );
        }

        // disable memory access from 'device' to 'peer_device'
        ~peer_access()
        {
            if ( device_1_ != device_2_ && device_1_ != -1 && device_2_ != -1 )
                cuda_device_disable_peer_access_from_to( device_1_, device_2_ );
        }
    };

    inline peer_access make_peer_access( int device_1, int device_2 ) noexcept
    {
        return peer_access{ device_1, device_2 };
    }

    /*

    // TODO: rewrite here;
    void uniform_initialize( float* address, unsigned long length )
    {
        cuda_uniform_initialize( address, length ); // in range (0,1)
		std::string const& rescale_weight_core =
		".version 5.0"
		".target sm_30"
		".address_size 64"
		".visible .entry rescale_weight("
		"	.param .u64 rescale_weight_param_0,"
		"	.param .u64 rescale_weight_param_1"
		")"
		".maxntid 128, 1, 1"
		"{"
		"	.reg .pred 	%p<2>;"
		"	.reg .f32 	%f<3>;"
		"	.reg .b32 	%r<5>;"
		"	.reg .b64 	%rd<7>;"
		"	ld.param.u64 	%rd2, [rescale_weight_param_0];"
		"	ld.param.u64 	%rd3, [rescale_weight_param_1];"
		"	cvta.to.global.u64 	%rd4, %rd3;"
		"	mov.u32 	%r1, %ctaid.x;"
		"	mov.u32 	%r2, %ntid.x;"
		"	mov.u32 	%r3, %tid.x;"
		"	mad.lo.s32 	%r4, %r1, %r2, %r3;"
		"	cvta.to.global.u64 	%rd5, %rd2;"
		"	mul.wide.s32 	%rd6, %r4, 4;"
		"	add.s64 	%rd1, %rd5, %rd6;"
		"	setp.ge.u64	%p1, %rd1, %rd4;"
		"	@%p1 bra 	BB0_2;"
		"	ld.global.f32 	%f1, [%rd1];"
		"	fma.rn.ftz.f32 	%f2, %f1, 0f40000000, 0fBF800000;"
		"	st.global.f32 	[%rd1], %f2;"
		"BB0_2:"
		"	ret;"
		"}";
		std::string const& rescale_weight_core_name = "rescale_weight";
		map( address, address+length, rescale_weight_core, rescale_weight_core_name );
    }
    */

}

#endif//DWJUWOBYHOTAGIAAHOTWEGPBLECMWYVTUVGJIDGRUCQRVQEVWSGTVWIICNONGWSPSRIOMJUKQ
