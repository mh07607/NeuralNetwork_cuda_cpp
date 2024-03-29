#ifndef KERNEL_CODE_H_
#define KERNEL_CODE_H_



__global__ void DenseForwardPass(float* d_M, float* d_N, float* d_P, float * d_B, int M, int N, int P);
__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int M, int N, int P);
__global__ void MatrixSubtractionKernel(float* d_M, float* d_N, float lr, int M, int N);
__global__ void transpose(float * d_N, float * d_M, int M, int N);
__global__ void DenseBackwardPass(
 float * d_outputError,
 float * d_input, 
 float * d_weights, 
 float * d_bias, 
 float * d_inputError, 
 float * d_weightsError, 
 float * d_weights_T, 
 float * d_input_T, 
 float lr, 
 int o_r, 
 int o_c, 
 int i_r, 
 int i_c, 
 int w_r, 
 int w_c
);
__global__ void ActivationForwardPass(int pairNum, float * arr, int M, int N);

__global__ void ActivationBackPass(int pairNum, float * arr, int M, int N);

__global__ void element_wise_mul(float * d_M, float * d_N, int M, int N);

#endif 