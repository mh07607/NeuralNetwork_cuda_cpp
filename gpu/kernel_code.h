#ifndef KERNEL_CODE_H_
#define KERNEL_CODE_H_



__global__ void DenseForwardPass(float* d_M, float* d_N, float* d_P, float * d_B, int M, int N, int P);
__global__ void test();
__global__ void DenseBackwardPass();
__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, float * d_B, int M, int N, int P);

#endif 