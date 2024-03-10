#ifndef KERNEL_CODE_H_
#define KERNEL_CODE_H_



__global__ void DenseForwardPass(float* d_M, float* d_N, float* d_P, float * d_B, int M, int N, int P);
__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int M, int N, int P);
__global__ void MatrixSubtractionKernel(float* d_M, float* d_N, float lr, int M, int N);
__global__ void transpose(float * d_N, float * d_M, int M, int N);
__global__ void DenseBackwardPass(

);

#endif 