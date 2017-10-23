//https://en.wikipedia.org/wiki/Activation_function



#define generate( NAME ) \
extern "C" \
__global__ void __launch_bounds__( 128 ) \
NAME##_activation( float* __restrict__ start, float* __restrict__ end, float* __restrict__ new_start ) \
{ \
    int const index = blockDim.x * blockIdx.x + threadIdx.x; \
    if ( start + index < end ) \
        new_start[index] = NAME(start[index]); \
} \
extern "C" \
__global__ void __launch_bounds__( 128 ) \
NAME##_derivative( float* __restrict__ start, float* __restrict__ end, float* __restrict__ new_start ) \
{ \
    int const index = blockDim.x * blockIdx.x + threadIdx.x; \
    if ( start + index < end ) \
        new_start[index] = d_##NAME( start[index] ); \
}


__device__ __forceinline__ float identity( float x )
{
    return x;
}

__device__ __forceinline__ float d_identity( float x )
{
    return 1.0f;
}

generate( identity );

__device__ __forceinline__ float logistic( float x )
{
    return 1.0f / ( 1.0f + expf( -x ) );
}

__device__ __forceinline__ float d_logistic( float x )
{
    float const tmp = logistic( x );
    return tmp - tmp*tmp;
}

generate( logistic );

__device__ __forceinline__ float tanh( float x )
{
    return tanhf( x );
}

__device__ __forceinline__ float d_tanh( float x )
{
    float const tmp = tanhf( x );
    return 1.0f - tmp*tmp;
}

generate( tanh );

__device__ __forceinline__ float arctan( float x )
{
    return atanf( x );
}

__device__ __forceinline__ float d_arctan( float x )
{
    return 1.0f / ( 1.0f + x*x );
}

generate( arctan );


__device__ __forceinline__ float softsign( float x )
{
    return x / ( 1.0f + fabsf(x) );
}
__device__ __forceinline__ float d_softsign( float x )
{
    return 1.0f / ( 1.0f + x*x );
}

generate( softsign );


__device__ __forceinline__ float relu( float x )
{
    if ( x < 0.0f ) return 0.0f;
    return x;
}

__device__ __forceinline__ float d_relu( float x )
{
    if ( x < 0.0f ) return 0.0f;
    return 1.0f;
}

generate( relu );

__device__ __forceinline__ float leaky_relu( float x )
{
    if ( x < 0.0f ) return 0.01f * x;
    return x;
}

__device__ __forceinline__ float d_leaky_relu( float x )
{
    if ( x < 0.0f ) return 0.01f;
    return 1.0f;
}

generate( leaky_relu );

// TODO
//prelu
//rrelu
//elu
//selu
//srelu
//apl
//softexponentional


__device__ __forceinline__ float softplus( float x )
{
    return logf( 1.0f + expf(x) );
}
__device__ __forceinline__ float d_softplus( float x )
{
    return 1.0f / ( 1.0f + expf( -x ) );
}

generate( softplus );

__device__ __forceinline__ float bent_identity( float x )
{
    return sqrtf( x*x + 1.0f ) / 2.0f - 0.5f + x;
}

__device__ __forceinline__ float d_bent_identity( float x )
{
    return 0.5f * x / sqrtf( x*x + 1.0f ) + 1.0f;
}

generate( bent_identity );





__device__ __forceinline__ float sinusoid( float x )
{
    return sinf(x);
}

__device__ __forceinline__ float d_sinusoid( float x )
{
    return cosf(x);
}

generate( sinusoid );

__device__ __forceinline__ float sinc( float x )
{
    if ( x == 0.0f ) return 1.0f;
    return sinf(x) / x;

}

__device__ __forceinline__ float d_sinc( float x )
{
    if ( x == 0.0f ) return 0.0f;
    return ( x * cosf(x) - sinf(x) ) / ( x*x );
}

generate( sinc );


__device__ __forceinline__ float gaussian( float x )
{
    return expf( -x*x );
}

__device__ __forceinline__ float d_gaussian( float x )
{
    return -2.0f * x * gaussian( x );
}

generate( gaussian );


