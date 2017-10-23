#ifndef DSFJVUVFFGXWSMNHABGNWFIRAMGMLNDPMUVBVBRGLNGDLARKVGJLINGNFJMTYUMXTJQOBNPOF
#define DSFJVUVFFGXWSMNHABGNWFIRAMGMLNDPMUVBVBRGLNGDLARKVGJLINGNFJMTYUMXTJQOBNPOF

#include <cuda.h>
#include <nvrtc.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h> 
#include <cublas_v2.h>
#include <curand.h>
#include <cublasXt.h>
#include <cufft.h>

#include <cstdio>
#include <cstdlib>
#include <map>

namespace cuda_assert_private
{

    struct cuda_result_assert
    {
        std::map<int,char const*> cufft_error_string;
        std::map<int,char const*> cublas_error_string;
        std::map<int,char const*> curand_error_string;

        cuda_result_assert()
        {
            cufft_error_string  [ CUFFT_SUCCESS                           ] = " 0    The cuFFT operation was successful";
            cufft_error_string  [ CUFFT_INVALID_PLAN                      ] = " 1    cuFFT was passed an invalid plan handle";
            cufft_error_string  [ CUFFT_ALLOC_FAILED                      ] = " 2    cuFFT failed to allocate GPU or CPU memory";
            cufft_error_string  [ CUFFT_INVALID_TYPE                      ] = " 3    No longer used";
            cufft_error_string  [ CUFFT_INVALID_VALUE                     ] = " 4    User specified an invalid pointer or parameter";
            cufft_error_string  [ CUFFT_INTERNAL_ERROR                    ] = " 5    Driver or internal cuFFT library error";
            cufft_error_string  [ CUFFT_EXEC_FAILED                       ] = " 6    Failed to execute an FFT on the GPU";
            cufft_error_string  [ CUFFT_SETUP_FAILED                      ] = " 7    The cuFFT library failed to initialize";
            cufft_error_string  [ CUFFT_INVALID_SIZE                      ] = " 8    User specified an invalid transform size";
            cufft_error_string  [ CUFFT_UNALIGNED_DATA                    ] = " 9    No longer used";
            cufft_error_string  [ CUFFT_INCOMPLETE_PARAMETER_LIST         ] = " 10   Missing parameters in call";
            cufft_error_string  [ CUFFT_INVALID_DEVICE                    ] = " 11   Execution of a plan was on different GPU than plan creation";
            cufft_error_string  [ CUFFT_PARSE_ERROR                       ] = " 12   Internal plan database error ";
            cufft_error_string  [ CUFFT_NO_WORKSPACE                      ] = " 13    No workspace has been provided prior to plan execution";
            cufft_error_string  [ CUFFT_NOT_IMPLEMENTED                   ] = " 14  Function does not implement functionality for parameters given.";
            cufft_error_string  [ CUFFT_LICENSE_ERROR                     ] = " 15  Used in previous versions.";
            cufft_error_string  [ CUFFT_NOT_SUPPORTED                     ] = " 16   Operation is not supported for parameters given.";

            curand_error_string [ CURAND_STATUS_SUCCESS                   ] = "Success";
            curand_error_string [ CURAND_STATUS_VERSION_MISMATCH          ] = "Header file and linked library version do not match. ";
            curand_error_string [ CURAND_STATUS_NOT_INITIALIZED           ] = "Generator not initialized. ";
            curand_error_string [ CURAND_STATUS_ALLOCATION_FAILED         ] = "Memory allocation failed. ";
            curand_error_string [ CURAND_STATUS_TYPE_ERROR                ] = "Generator is wrong type. ";
            curand_error_string [ CURAND_STATUS_OUT_OF_RANGE              ] = "Argument out of range. ";
            curand_error_string [ CURAND_STATUS_LENGTH_NOT_MULTIPLE       ] = "Length requested is not a multple of dimension. ";
            curand_error_string [ CURAND_STATUS_DOUBLE_PRECISION_REQUIRED ] = "GPU does not have double precision required by MRG32k3a. ";
            curand_error_string [ CURAND_STATUS_LAUNCH_FAILURE            ] = "Kernel launch failure. ";
            curand_error_string [ CURAND_STATUS_PREEXISTING_FAILURE       ] = "Preexisting failure on library entry. ";
            curand_error_string [ CURAND_STATUS_INITIALIZATION_FAILED     ] = "Initialization of CUDA failed. ";
            curand_error_string [ CURAND_STATUS_ARCH_MISMATCH             ] = "Architecture mismatch, GPU does not support requested feature.";
            curand_error_string [ CURAND_STATUS_INTERNAL_ERROR            ] = "nternal library error. ";

            cublas_error_string [ CUBLAS_STATUS_SUCCESS                   ] = "Success";
            cublas_error_string [ CUBLAS_STATUS_NOT_INITIALIZED           ] = "The cuBLAS library was not initialized. This is usually caused by the lack of a prior cublasCreate() call, an error in the CUDA Runtime API called by the cuBLAS routine, or an error in the hardware setup. \n To correct: call cublasCreate() prior to the function call; and check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed.";
            cublas_error_string [ CUBLAS_STATUS_ALLOC_FAILED              ] = "Resource allocation failed inside the cuBLAS library. This is usually caused by a cudaMalloc() failure.\n To correct: prior to the function call, deallocate previously allocated memory as much as possible. ";
            cublas_error_string [ CUBLAS_STATUS_INVALID_VALUE             ] = "An unsupported value or parameter was passed to the function (a negative vector size, for example).\n To correct: ensure that all the parameters being passed have valid values.";
            cublas_error_string [ CUBLAS_STATUS_ARCH_MISMATCH             ] = "The function requires a feature absent from the device architecture; usually caused by the lack of support for double precision.\n  To correct: compile and run the application on a device with appropriate compute capability, which is 1.3 for double precision.";
            cublas_error_string [ CUBLAS_STATUS_MAPPING_ERROR             ] = "An access to GPU memory space failed, which is usually caused by a failure to bind a texture. \n To correct: prior to the function call, unbind any previously bound textures.";
            cublas_error_string [ CUBLAS_STATUS_EXECUTION_FAILED          ] = "The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.  \n To correct: check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed.";
            cublas_error_string [ CUBLAS_STATUS_INTERNAL_ERROR            ] = "An internal cuBLAS operation failed. This error is usually caused by a cudaMemcpyAsync() failure. \n To correct: check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed. Also, check that the memory passed as a parameter to the routine is not being deallocated prior to the routine’s completion. ";
            cublas_error_string [ CUBLAS_STATUS_NOT_SUPPORTED             ] = "The functionnality requested is not supported";
            cublas_error_string [ CUBLAS_STATUS_LICENSE_ERROR             ] = "The functionnality requested requires some license and an error was detected when trying to check the current licensing. This error can happen if the license is not present or is expired or if the environment variable NVIDIA_LICENSE_FILE is not set properly.";
        }

        void operator()( const cudaError_t& result, const char* const file, const unsigned long line ) const
        {
            if ( 0 == result ) { return; }
            fprintf( stderr, "%s:%lu: cudaError occured:\n[[ERROR]]: %s\n", file, line, cudaGetErrorString(result) );
            abort();
        }

        void operator()( const CUresult& result, const char* const file, const unsigned long line ) const
        {
            if ( 0 == result ) { return; }

            const char* msg;
            cuGetErrorString( result, &msg );
            const char* name;
            cuGetErrorName( result, &name );

            fprintf( stderr, "%s:%lu: CUresult error occured:\n[[ERROR]]: %s --- %s\n", file, line, name, msg );
            abort();
        }

        void operator()( const nvrtcResult& result, const char* const file, const unsigned long line ) const
        {
            if ( 0 == result ) { return; }

            fprintf( stderr, "%s:%lu: nvrtcResult error occured:\n[[ERROR]]: %s\n", file, line, nvrtcGetErrorString(result) );
            abort();
        }

        void operator()( const cufftResult& result, const char* const file, const unsigned long line ) const
        {
            if ( 0 == result ) { return; }

            if ( auto it =  cufft_error_string.find(result); it == cufft_error_string.end() )
                fprintf( stderr, "%s:%lu: cufft error occured:\n[[ERROR]]: Unkown\n", file, line );
            else
                fprintf( stderr, "%s:%lu: cufft error occured:\n[[ERROR]]: %s\n", file, line, (*it).second );

            abort();
        }

        void operator()( const cublasStatus_t& result, const char* const file, const unsigned long line ) const
        {
            if ( 0 == result ) { return; }

            if ( auto it = cublas_error_string.find(result); it == cublas_error_string.end() )
                fprintf( stderr, "%s:%lu: cublas error occured:\n[[ERROR]]: Unknown\n", file, line );
            else
                fprintf( stderr, "%s:%lu: cublas error occured:\n[[ERROR]]: %s\n", file, line, (*it).second );

            abort();
        }

        void operator()( const curandStatus_t& result, const char* const file, const unsigned long line ) const
        {
            if ( 0 == result ) { return; }

            if ( auto it = curand_error_string.find(result); it == curand_error_string.end() )
                fprintf( stderr, "%s:%lu: curand error occured:\n[[ERROR]]: Unknown\n", file, line );
            else
                fprintf( stderr, "%s:%lu: curand error occured:\n[[ERROR]]: %s\n", file, line, (*it).second );

            abort();
        }

    };

}

#ifdef cuda_assert
#undef cuda_assert
#endif

static cuda_assert_private::cuda_result_assert cuda_result_asserter;

//#define cuda_assert(result) cuda_assert_private::cuda_result_assert{}(result, __FILE__, __LINE__)
#define cuda_assert(result) cuda_result_asserter(result, __FILE__, __LINE__)

#endif//DSFJVUVFFGXWSMNHABGNWFIRAMGMLNDPMUVBVBRGLNGDLARKVGJLINGNFJMTYUMXTJQOBNPOF

