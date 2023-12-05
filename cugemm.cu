#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace std;
using namespace nvcuda;
// from https://github.com/jarro2783/cxxopts
#include "cxxopts.hpp"

#define cudaCheck(err) (cudaErrorCheck(err, __FILE__, __LINE__))
#define cublasCheck(err) (cublasErrorCheck(err, __FILE__, __LINE__))
#define ROUND_UP_TO_NEAREST(M, N) (((M) + (N)-1) / (N))

//using namespace nvcuda;

enum Algo
{
    cublas_hgemm = 0,
    cuda_hgemm,
    tensor_hgemm,
    numAlgos
};

const char *algo2str(Algo a)
{
    switch (a)
    {
    case cublas_hgemm:
        return "cublas_hgemm";
    /*case cuda_hgemm:
        return "cuda_hgemm";*/
    case tensor_hgemm:
        return "tensor_hgemm";
    default:
        return "INVALID";
    }
}

void cudaErrorCheck(cudaError_t error, const char *file, int line);
void cublasErrorCheck(cublasStatus_t status, const char *file, int line);
void randomize_matrix(half *mat, int N);
void const_init_matrix(half *mat, int N, half F);
bool verify_matrix(half *expected, half *actual, int M, int N);
void print_matrix(const half *A, int M, int N, std::ostream &outs);
void runAlgo(Algo algo, cublasHandle_t handle, int M, int N, int K, half alpha, half *A, half *B, half beta, half *C);
void runCublas(cublasHandle_t handle, int M, int N, int K, half alpha, half *A, half *B, half beta, half *C);

const std::string errLogFile = "gemmValidationFailure.txt";

// NB: must use a single generator to avoid duplicates
std::default_random_engine generator(2);
std::uniform_real_distribution<float> distribution(0, 1);

int main(int argc, char **argv)
{
    // command-line flags
    cxxopts::Options options("gemm.cu", "CUDA GEMM kernels");
    options.add_options()("size", "matrix size (N x N)", cxxopts::value<uint16_t>()->default_value("128"))                //
        ("reps", "repeat GEMM this many times", cxxopts::value<uint16_t>()->default_value("1"))                           //
        ("algo", "GEMM algorithm to use, a number in [0,2], 0 is cuBLAS", cxxopts::value<uint16_t>()->default_value("0")) //
        ("validate", "Validate output against cuBLAS", cxxopts::value<bool>()->default_value("true"))                     //
        ("rngseed", "PRNG seed", cxxopts::value<uint>()->default_value("2"))                     //
        ("h,help", "Print usage");

    auto clFlags = options.parse(argc, argv);
    if (clFlags.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    const uint16_t SIZE = clFlags["size"].as<uint16_t>();
    if (SIZE % 32 != 0)
    {
        //std::cout << "--size must be a multiple of 32" << std::endl;
        //exit(EXIT_FAILURE);
    }
    const uint16_t REPS = clFlags["reps"].as<uint16_t>();
    const Algo ALGO = static_cast<Algo>(clFlags["algo"].as<uint16_t>());
    if (ALGO >= numAlgos)
    {
        printf("Invalid algorithm: %d\n", ALGO);
        exit(EXIT_FAILURE);
    }

    const bool VALIDATE = clFlags["validate"].as<bool>();
    const uint SEED = clFlags["rngseed"].as<uint>();
    generator.seed(SEED);
    printf("Multiplying two %u x %u matrices with %u trials using %s algorithm\n", SIZE, SIZE, REPS, algo2str(ALGO));

    cudaCheck(cudaSetDevice(0));

    // Setup cublas
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);

    // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
    // publishing event tasks in the target stream
    cudaEvent_t beg, end;
    cudaCheck(cudaEventCreate(&beg));
    cudaCheck(cudaEventCreate(&end));

    uint16_t m = SIZE, n = SIZE, k = SIZE;

    // GEMM computes C = α*AB+β*C

    // just do pure A*B (for simpler debugging)
    half alpha = 1.0, beta = 0.0, initC = 1.0;

    half *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr;     // host matrices
    half *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC_ref = nullptr; // device matrices

    A = (half *)malloc(sizeof(half) * SIZE * SIZE);
    B = (half *)malloc(sizeof(half) * SIZE * SIZE);
    C = (half *)malloc(sizeof(half) * SIZE * SIZE);
    C_ref = (half *)malloc(sizeof(half) * SIZE * SIZE);

    randomize_matrix(A, SIZE * SIZE);
    randomize_matrix(B, SIZE * SIZE);
    randomize_matrix(C, SIZE * SIZE);

    const_init_matrix(C, SIZE * SIZE, initC);


    cudaCheck(cudaMalloc((void **)&dA, sizeof(half) * SIZE * SIZE));
    cudaCheck(cudaMalloc((void **)&dB, sizeof(half) * SIZE * SIZE));
    cudaCheck(cudaMalloc((void **)&dC, sizeof(half) * SIZE * SIZE));
    cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(half) * SIZE * SIZE));

    cudaCheck(cudaMemcpy(dA, A, sizeof(half) * SIZE * SIZE, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(half) * SIZE * SIZE, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, sizeof(half) * SIZE * SIZE, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, C, sizeof(half) * SIZE * SIZE, cudaMemcpyHostToDevice));

    printf("dimensions(m=n=k) %u, alpha: %f, beta: %f\n", m, __half2float(alpha), __half2float(beta));

    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    if (!VALIDATE)
    {
        printf("disabled validation\n");
    }
    else
    {
        // run cublas to get correct answer in dC_ref
        runCublas(handle, m, n, k, alpha, dA, dB, beta, dC_ref);

        // run user's algorithm, filling in dC
        runAlgo(ALGO, handle, m, n, k, alpha, dA, dB, beta, dC);

        cudaCheck(cudaDeviceSynchronize());

        // copy both results back to host
        cudaMemcpy(C, dC, sizeof(half) * m * n, cudaMemcpyDeviceToHost);
        cudaMemcpy(C_ref, dC_ref, sizeof(half) * m * n, cudaMemcpyDeviceToHost);

        if (verify_matrix(C_ref, C, n, m))
        {
            printf("Validated successfully!\n");
        }
        else
        {
            printf("Failed validation against NVIDIA cuBLAS.\n");
            std::cout << " Logging faulty output into " << errLogFile << "\n";
            std::ofstream fs;
            fs.open(errLogFile, std::ios::out | std::ios::trunc);
            fs << "α=" << __half2float(alpha) << " β=" << __half2float(beta) << std::endl;
            fs << "C matrix initialized to " << __half2float(initC) << std::endl << std::endl;
            fs << "A:" << std::endl;
            print_matrix(A, m, n, fs);
            fs << "B:" << std::endl;
            print_matrix(B, m, n, fs);
            fs << "C:" << std::endl;
            print_matrix(C, m, n, fs);
            fs << "Expected:" << std::endl;
            print_matrix(C_ref, m, n, fs);
            fs.close();
            exit(EXIT_FAILURE);
        }
    }

    // timing run(s)
    cudaEventRecord(beg);
    for (int j = 0; j < REPS; j++)
    {
        // We don't reset dC between runs to save time
        runAlgo(ALGO, handle, m, n, k, alpha, dA, dB, beta, dC);
        cudaCheck(cudaDeviceSynchronize());
    }

    // TODO: measure timing without memory transfers?
    cudaCheck(cudaEventRecord(end));
    cudaCheck(cudaEventSynchronize(beg));
    cudaCheck(cudaEventSynchronize(end));
    float elapsed_time;
    cudaCheck(cudaEventElapsedTime(&elapsed_time, beg, end));
    elapsed_time /= 1000.; // Convert to seconds

    double flops = (double)2 * m * n * k;
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.2f) GFLOPS. size: (%u).\n",
        elapsed_time / REPS,
        (REPS * flops * 1e-9) / elapsed_time,
        m);

    // free CPU and GPU memory
    free(A);
    free(B);
    free(C);
    free(C_ref);
    cudaCheck(cudaFree(dA));
    cudaCheck(cudaFree(dB));
    cudaCheck(cudaFree(dC));
    cudaCheck(cudaFree(dC_ref));
    cublasCheck(cublasDestroy(handle));

    return 0;
}

/** Function to check for errors in CUDA API calls */
void cudaErrorCheck(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s:%d:\n%s: %s\n", file, line,
               cudaGetErrorName(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};

void cublasErrorCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("[CUDA ERROR] at file %s:%d:\n %s: %s\n", file, line,
               cublasGetStatusName(status), cublasGetStatusString(status));
        exit(EXIT_FAILURE);
    }
}

/** Initialize the given matrix `mat` which has `N` contiguous values. Contents of `mat` are set to random values. */
void randomize_matrix(half *mat, int N)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = distribution(generator);
    }
}

void const_init_matrix(half *mat, int N, half F)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = F;
    }
}

/** Print the given MxN matrix `mat` to the provided output stream. */
void print_matrix(const half *A, int M, int N, std::ostream &outs)
{
    outs << "[";
    for (int i = 0; i < M * N; i++)
    {
        if ((i + 1) % N == 0)
        {
            outs << std::fixed << std::setprecision(3) << __half2float(A[i]);
        }
        else
        {
            outs << std::fixed << std::setprecision(3) << __half2float(A[i]) << ", ";
        }
        if ((i + 1) % N == 0)
        {
            if (i + 1 < M * N)
                outs << ";" << std::endl;
        }
    }
    outs << "]" << std::endl << std::endl;
}

bool verify_matrix(half *expected, half *actual, int M, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            half fexp = (expected[(i * N) + j]);
            half fact = (actual[(i * N) + j]);
            double diff = std::fabs(__half2float(fexp) - __half2float(fact));
            if (diff > 0.002)
            {
                printf("Divergence! Should be %5.3f, is %5.3f (diff %5.3f) at [%d,%d]\n",
                       __half2float(fexp), __half2float(fact), __half2float(diff), i, j);
                return false;
            }
        }
    }
    return true;
}

void runCublas(cublasHandle_t handle, int M, int N, int K, half alpha,
               half *A, half *B, half beta, half *C)
{

    cublasStatus_t ok = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, K, A, CUDA_R_16F, K,
                                    &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasCheck(ok);
}

/*__global__ void runBasic(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N)
    {
        float tmp = 0.0;
        // C = α*(AxB)+β*C
        for (int i = 0; i < K; ++i)
        {
            // tmp += __A__[x][i] * __B__[i][y]
            tmp += A[(x * K) + i] * B[(i * N) + y];
        }
        // __C__[x][y]
        C[(x * N) + y] = (alpha * tmp) + (beta * C[x * N + y]);
    }
}
*/

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void tensor_impl(int M, int N, int K, half alpha, half *A, half *B, half beta, half *C)
{
    

    int K_tiles = ROUND_UP_TO_NEAREST(K,WMMA_K);

    int row  = blockIdx.y * WMMA_M;
    int column  = blockIdx.x * WMMA_N;

    if(row >= M && column >= N)
    {
        return;
    }

    nvcuda::wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;

    nvcuda::wmma::fill_fragment(C_frag,0.0);

#pragma unroll
    for(int i = 0; i<K_tiles; ++i)
    {
        nvcuda::wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
        nvcuda::wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> B_frag;

        nvcuda::wmma::load_matrix_sync(A_frag, A + row*K + i*WMMA_K,K);
        nvcuda::wmma::load_matrix_sync(B_frag, B + i*WMMA_K + column*K,K);

        nvcuda::wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

    }

    wmma::store_matrix_sync(C + row * N + column, C_frag, N, wmma::mem_row_major);


}

/*

__global__ void runSharedMem(int M, int N, int K, half alpha, half *A, half *B, half beta, half *C)
{
    // HW2 TODO: Use shared memory to cache square FxF tiles of the A and B matrices in shared memory 
    // (SA and SB, respectively, provided below). Each thread should compute the result for one cell 
    // of the output matrix C.

    // Note, you will also need to change the grid dimensions in the kernel launch below to take into account the value
    // of F (which is a constant, defined above). You should experiment with different values of F to see how it 
    // affects performance.

    __shared__ half SA[F][F];
    __shared__ half SB[F][F];

    const unsigned blkidx = blockIdx.x;
    const unsigned blkidy = blockIdx.y;
    const unsigned bdimx = blockDim.x;
    const unsigned bdimy = blockDim.x;
    const unsigned threadx = threadIdx.x;
    const unsigned thready = threadIdx.y;

    const unsigned column = blockIdx.x * F + threadIdx.x;
    const unsigned row = blockIdx.y * F + threadIdx.y;

    // const unsigned row = blkidy * F + thready;
    // const unsigned column = blkidx * F + threadx;
    if (row < M && column < N)
    {
        half tmp = 0.0;
        // C = α*(AxB)+β*C
        for (int i = 0; i < K/F; ++i)
        {
            // tmp += __A__[x][i] * __B__[i][y]
            //tmp += A[(x * K) + i] * B[(i * N) + y];
            SA[thready][threadx] = A[(row * K) + i*F+threadx];

            //Print debugs
           //printf("A element at address %d = %f \n",(row * K) + i*F+threadx, A[(row * K) + i*F+threadx]);
           //printf(" A Accessed into SA (address: %d,%d): %f , row  = %d , blockid = %d \n",thready,threadx,SA[thready][threadx],row,blockIdx.y);

            //Print debugs
            SB[thready][threadx] = B[(i* F + thready)*K + column];
           //printf("B element at address %d  = %f \n",(i* F + thready)*K + column, B[(i* F + thready)*K + column]);
           //printf(" B Accessed into SA (address: %d,%d): %f \n",thready,threadx,SB[thready][threadx]);

           __syncthreads();

            for(int j = 0; j < F; ++j)
            {
                tmp += SA[thready][j] * SB[j][threadx];

            }
                __syncthreads();
        }
        // __C__[x][y]
        C[(row * K) + column] = (alpha * tmp) + (beta * C[row * K + column]);
    }
    }
*/
/*
const uint G = 4;
const uint F = 32;

__global__ void run_cuda_hgemm(int M, int N, int K, half alpha, half *A, half *B, half beta, half *C)
{
    // HW3 TODO: Copy your runSharedMem() code here and update it so that each thread computes the result for GxG cells 
    // of the output matrix C. Each thread should accumulate temporary results in the local LC matrix, provided below,
    // before writing them to C in global memory.

    // Note, you will also need to change the grid dimensions in the kernel launch below. You should experiment 
    // with different values of F and G to see how they affect performance.

    __shared__ half SA[F][F];
    __shared__ half SB[F][F];

    float LC[G][G] = {0.0};
    half resSA[G] = {0.0}; // Temp 
    half resSB[G] = {0.0}; // Temp

    //const unsigned threadx = threadIdx.x;
    //const unsigned thready = threadIdx.y;

   // const unsigned column = blockIdx.x * F + threadIdx.x;
   // const unsigned row = blockIdx.y * F + threadIdx.y;

    int divider = (F*F)/(G*G);
    int stride  = divider / F;

    const unsigned row_s = threadIdx.x / F;
    const unsigned column_s = threadIdx.x % F;
    const unsigned row_l = (threadIdx.x / (F/G));
    const unsigned column_l = (threadIdx.x % (F/G)) ;

    // Start of the Matrices
    A += blockIdx.y * F * K;
    B += blockIdx.x * F;
    C += blockIdx.y * F * N + blockIdx.x * F;
    


        for (int i = 0; i < K/F; ++i)
        {
            // tmp += __A__[x][i] * __B__[i][y]
            for(int c = 0; c < F; c+=stride)
            {
                SA[row_s + c][column_s] = A[((row_s + c) * K) + i*F+column_s];
                SB[row_s + c][column_s] = B[(i* F + (row_s + c))*K + column_s];
                
            }
            __syncthreads();    

            for(int p = 0 ; p < F ; ++p)
            {
                for(int r = 0; r<G; ++r)
                {
                    resSA[r] = SA[row_l * G + r][p];
                    resSB[r] = SB[p][column_l*G +r];
                    //printf("row = %d, column = %d \n",row_l,column_l);

                }

                for(int m = 0 ; m < G ; ++m)
                {
                    for(int n = 0 ; n < G ; ++n)
                    {
                        LC[m][n] +=  __half2float(resSA[m]) * __half2float(resSB[n]);

                        //printf("\n LC  = %f \n", LC[m][n]);
                    }
                    
                }
                
            }
    
            __syncthreads();
        }
        // __C__[x][y]

        for(int m = 0 ; m < G ; ++m)
        {
            for(int n = 0 ; n < G ; ++n)
            {
                C[(row_l * G + m)*M + column_l * G + n] = (alpha * (LC[m][n])) + (beta * C[(row_l * G + m)*M + column_l * G + n]);
                //printf("\n C at %d,%d  = %f , row  = %d, column = %d \n", (row_l * G + m)*M , column_l * G + n ,C[(row_l * G + m)*M + column_l * G + n], row_l, column_l);
            }
        }
        
}*/


void runAlgo(Algo algo, cublasHandle_t handle, int M, int N, int K, half alpha,
             half *A, half *B, half beta, half *C)
{
    switch (algo)
    {
    case cublas_hgemm:
        runCublas(handle, M, N, K, alpha, A, B, beta, C);
        break;
    // case cuda_hgemm:
    // {
    //     assert(0 == M % F);
    //     assert(0 == N % F);
    //     assert(0 == K % F);
    //     assert(0 == F % G);
    //     assert((F*F) / (G*G) >= F);
    //     // TODO: update your grid here
    //     dim3 gridDim(ROUND_UP_TO_NEAREST(M, F), ROUND_UP_TO_NEAREST(N, F));
    //     //dim3 gridDim(ROUND_UP_TO_NEAREST(M, 32)/F, ROUND_UP_TO_NEAREST(N, 32)/F);
    //     dim3 blockDim((F*F)/(G*G));
    //     run_cuda_hgemm<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    //     break;
    // }
    case tensor_hgemm:
    {   
        dim3 block(32);
        dim3 grid(ROUND_UP_TO_NEAREST(N, WMMA_N), ROUND_UP_TO_NEAREST(M,WMMA_M));

        tensor_impl<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
        break;
    }
    default:
        printf("Invalid algorithm: %d\n", algo);
        exit(EXIT_FAILURE);
    }
    cudaCheck(cudaDeviceSynchronize()); // wait for kernel to finish
    cudaCheck(cudaGetLastError());      // check for errors from kernel run
}
