#include "./warnings.hpp"

SUPPRESS_WARNINGS

#include "./cuda_assert.hpp"

//extern char const* const cunn_ptx; // all kernels in file cunn_ptx.cc

#ifdef __cplusplus
extern "C" {
#endif

//#include "./cunn_ptx.icl"
//TODO: runtime compilation for the ptx file, as it is device sensitive
//#include "./cunn.ptx"

int devices_count()
{
    int devices = 0;
    cuda_assert( cudaGetDeviceCount(&devices) );
    return devices;
}

int get_device()
{
    int current_id;
    cuda_assert( cudaGetDevice( &current_id ) );
    return current_id;
}

void set_device( int id )
{
    int current_id;
    cuda_assert( cudaGetDevice( &current_id ) );
    if ( current_id != id )
        cuda_assert( cudaSetDevice( id ) );
}

int get_capability( int id )
{
    set_device( id );
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, id);
    return deviceProp.major * 100 + deviceProp.minor;
}

unsigned long long get_global_mem( int id )
{
    set_device( id );
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, id);
    return (unsigned long long) deviceProp.totalGlobalMem;
}

void reset_device()
{
    cuda_assert( cudaDeviceReset() );
}

void synchronize_device()
{
    cuda_assert( cudaDeviceSynchronize() );
}

void cuda_allocate( void** p, unsigned long n )
{
    cuda_assert( cudaMalloc( p, n ) );
    cuda_assert( cudaMemset( *p, 0, n ) );
}

void cuda_deallocate( void* p )
{
    if ( p )
        cuda_assert( cudaFree( p ) );
}

void cuda_host_allocate( void** p, unsigned long n )
{
    cuda_assert( cudaMallocHost( p, n ) );
}

void cuda_host_deallocate( void* p )
{
    if ( p )
        cuda_assert( cudaFreeHost( p ) );
}

void cuda_memcopy_host_to_device( const void* src, unsigned long n, void* dst )
{
    if (n) cuda_assert( cudaMemcpy( dst, src, n, cudaMemcpyHostToDevice  ) );
}

void cuda_memcopy_device_to_host( const void* src, unsigned long n, void* dst )
{
    if ( n ) cuda_assert( cudaMemcpy( dst, src, n, cudaMemcpyDeviceToHost  ) );
}

void cuda_memcopy_device_to_device( const void* src, unsigned long n, void* dst )
{
    if (n) cuda_assert( cudaMemcpy( dst, src, n, cudaMemcpyDeviceToDevice  ) );
}

int cuda_device_enable_peer_access_from_to( int device, int peer_device )
{
    if ( device == peer_device )
        return 1;

    int can_access;
    cuda_assert( cudaDeviceCanAccessPeer( &can_access, device, peer_device ) );

    // not accessible
    if ( can_access == 0 )
        return 0;

    set_device( device );
    //cuda_assert( cudaSetEnablePeerAccess( peer_device, 0 ) );
    cuda_assert( cudaDeviceEnablePeerAccess( peer_device, 0 ) );

    return 1;
}

void cuda_device_disable_peer_access_from_to( int device, int peer_device )
{
    if ( device == peer_device )
        return;

    set_device( device );
    cuda_assert( cudaDeviceDisablePeerAccess( peer_device ) );
}

// uniform in range (0, 1)
void cuda_uniform_initialize( float* address, unsigned long length )
{
    curandGenerator_t gen;
    cuda_assert( curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_DEFAULT ) );
    int* dummy = new int;
    unsigned long long seed = (unsigned long long)dummy;
    seed += (unsigned long long)(&seed);
    cuda_assert( curandSetPseudoRandomGeneratorSeed( gen, seed ) );
    cuda_assert( curandGenerateUniform( gen, address, length ) );


    // address *= ( upper_value - lower_value )
    // address += lower_value;


    delete dummy;
}

// C = AB
/*
void cuda_gemm_ab_c( unsigned long a_row, unsigned long a_col, unsigned long b_col, float const* A, float const* B, float* C )
{
    int m = b_col;
    int n = a_row;
    int k = a_col;
    float const alpha = 1.0f;
    float const beta = 0.0f;
    cublasHandle_t handle;
    cuda_assert( cublasCreate(&handle) );
    // Correct
    cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)b_col, (int)a_row, (int)a_col, &alpha, B, (int)b_col, A, (int)a_col, &beta, C, (int)b_col );
    cuda_assert( cudaDeviceSynchronize() );
    cuda_assert( cublasDestroy(handle) );
}
*/

#if 0

// C = A^T B
void cuda_gemm_aTb_c( unsigned long a_row, unsigned long a_col, unsigned long b_col, float const* A, float const* B, float* C )
{
    float const alpha = 1.0f;
    float const beta = 0.0f;
    cublasHandle_t handle;
    cuda_assert( cublasCreate(&handle) );
    //cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_T, (int)b_col, (int)a_row, (int)a_col, &alpha, B, (int)b_col, A, (int)a_col, &beta, C, (int)b_col );
    //cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_T, (int)b_col, (int)a_col, (int)a_row, &alpha, B, (int)b_col, A, (int)a_row, &beta, C, (int)b_col );
    cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_T, (int)b_col, (int)a_col, (int)a_col, &alpha, B, (int)b_col, A, (int)a_col, &beta, C, (int)b_col );
    cuda_assert( cudaDeviceSynchronize() );
    cuda_assert( cublasDestroy(handle) );
}

// C = A B^T
void cuda_gemm_abT_c( unsigned long a_row, unsigned long a_col, unsigned long b_row, float const* A, float const* B, float* C )
//void cuda_gemm_abT_c( unsigned long a_row, unsigned long a_col, unsigned long b_row, float const* A, float const* B, float* C )
{
    float const alpha = 1.0f;
    float const beta = 0.0f;
    cublasHandle_t handle;
    cuda_assert( cublasCreate(&handle) );
    //cublasSgemm( handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)b_row, (int)a_row, (int)a_col, &alpha, B, (int)b_row, A, (int)a_col, &beta, C, (int)b_row );
    //cublasSgemm( handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)b_row, (int)a_row, (int)a_col, &alpha, B, (int)b_row, A, (int)a_col, &beta, C, (int)b_row );
    //cublasSgemm( handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)b_row, (int)a_row, (int)a_col, &alpha, B, (int)a_col, A, (int)a_col, &beta, C, (int)a_col );
    //cublasSgemm( handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)b_row, (int)a_row, (int)a_col, &alpha, B, (int)a_col, A, (int)a_col, &beta, C, (int)b_row );
    cublasSgemm( handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)b_row, (int)a_row, (int)a_col, &alpha, B, (int)a_col, A, (int)a_col, &beta, C, (int)b_row );
    cuda_assert( cudaDeviceSynchronize() );
    cuda_assert( cublasDestroy(handle) );
}
#endif

/*
void cuda_gemm_aTb_c( float const* A, unsigned long a_row, unsigned long a_col,
                      float const* B, unsigned long b_row, unsigned long b_col,
                      float* C, unsigned long c_row, unsigned long c_col )
{
    float const alpha = 1.0f;
    float const beta = 0.0f;
    cublasHandle_t handle;
    cuda_assert( cublasCreate(&handle) );
    cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_T, (int) c_col, (int) c_row, (int) b_row, &alpha, B, (int) b_col, A, (int) a_col, &beta, C, (int) c_col );
    cuda_assert( cudaDeviceSynchronize() );
    cuda_assert( cublasDestroy(handle) );
}

void cuda_gemm_abT_c( float const* A, unsigned long a_row, unsigned long a_col,
                      float const* B, unsigned long b_row, unsigned long b_col,
                      float* C, unsigned long c_row, unsigned long c_col )
{
    float const alpha = 1.0f;
    float const beta = 0.0f;
    cublasHandle_t handle;
    cuda_assert( cublasCreate(&handle) );
    //cublasSgemm( handle, CUBLAS_OP_T, CUBLAS_OP_N, (int) c_col, (int) c_row, (int) b_row, &alpha, B, (int) b_col, A, (int) a_col, &beta, C, (int) c_col );
    cublasSgemm( handle, CUBLAS_OP_T, CUBLAS_OP_N, (int) c_col, (int) c_row, (int) b_row, &alpha, B, (int) b_row, A, (int) a_col, &beta, C, (int) c_col );

#ifdef DEBUGLOG
    printf( "Calling cublasSgemm( handle, true, false, %ld, %ld, %ld, alpha, B, %ld, A, %ld, beta, C, %ld );\n", c_col, c_row, b_row, b_col, a_col, c_col );
#endif

    cuda_assert( cudaDeviceSynchronize() );
    cuda_assert( cublasDestroy(handle) );
}
*/


float nrm2( float const* x, unsigned long n )
{
    cublasHandle_t handle;
    cuda_assert( cublasCreate(&handle) );
    float ans;

    cuda_assert( cublasSnrm2( handle, (int)n, x, 1, &ans ) );

    cuda_assert( cudaDeviceSynchronize() );
    cuda_assert( cublasDestroy(handle) );
    return ans;
}

static void cuda_gemm( float const* A, float const* B, float* C, int m, int n, int k, bool trans_a, bool trans_b )
{
    float const alpha = 1.0f;
    float const beta = 0.0f;
    cublasHandle_t handle;
    cuda_assert( cublasCreate(&handle) );

    cuda_assert( cublasSgemm(   handle,
                                trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                                trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                                n, m, k, &alpha,
                                B,
                                trans_b ? k : n,
                                A,
                                trans_a ? m : k,
                                &beta, C, n
                            )
            );

    cuda_assert( cudaDeviceSynchronize() );
    cuda_assert( cublasDestroy(handle) );
}

void cuda_gemm_ab_c( unsigned long a_row, unsigned long a_col, unsigned long b_col, float const* A, float const* B, float* C )
{
    cuda_gemm( A, B, C, a_row, b_col, a_col, false, false );
}

void cuda_gemm_aTb_c( float const* A, unsigned long a_row, unsigned long a_col,
                      float const* B, unsigned long b_row, unsigned long b_col,
                      float* C, unsigned long c_row, unsigned long c_col )
{
    cuda_gemm( A, B, C, c_row, c_col, a_row, true, false );
}

void cuda_gemm_abT_c( float const* A, unsigned long a_row, unsigned long a_col,
                      float const* B, unsigned long b_row, unsigned long b_col,
                      float* C, unsigned long c_row, unsigned long c_col )
{
    cuda_gemm( A, B, C, c_row, c_col, a_col, false, true );
}






#if 0
void invoke( char const* const func, void** args, unsigned long length )
{
    int tx = 128;
    int ty = 1;
    int tz = 1;
    int gx = static_cast<int>((length+tx-1) / tx);
    int gy = 1;
    int gz = 1;
    int shared_memory_in_byte = 0;

    cuda_assert( cuInit( 0 ) );
    CUdevice cuDevice;
    cuda_assert( cuDeviceGet( &cuDevice, 0 ) );
    CUcontext context;
    cuda_assert( cuCtxCreate( &context, 0, cuDevice ) );
    CUmodule module;
    cuda_assert( cuModuleLoadDataEx( &module, cunn_ptx, 0, 0, 0 ) );
    CUfunction kernel;
    cuda_assert( cuModuleGetFunction( &kernel, module, func ) );
    cuda_assert( cuLaunchKernel( kernel, gx, gy, gz, tx, ty, tz, shared_memory_in_byte, 0, args, nullptr ) );
    cuda_assert( cuCtxSynchronize() );
    cuda_assert( cuModuleUnload( module ) );
    cuda_assert( cuCtxDestroy( context ) );
}
#endif

#ifdef __cplusplus
}
#endif


RESTORE_WARNINGS
