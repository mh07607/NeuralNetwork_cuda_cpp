#include "kernel_code.h"
// #include "ActivationAndLossFunctions.h"
// #include <math_functions.h>

//activation functions
__device__ float sigmoid(float x)
{
	return 1.0f / 1.0f + exp(-x);
}

__device__ float sigmoid_prime(float x)
{
	float s = sigmoid(x);
	return s * (1 - s);
}
__device__ float tanh2(float x)
{
	return tanh(x);
}

__device__ float tanh_prime(float x)
{
	return 1.0f - powf(tanh(x), 2.0f);
}

__device__ float relu(float x)
{
	return max(x, 0.0f);
}
__device__ float relu_prime(float x)
{
	return (float)((int)(x >= 0));
}

__device__ float one_minus(float x)
{
	return 1 - x;
}


__global__ void DenseForwardPass(float* d_M, float* d_N, float* d_P, float * d_B, int M, int N, int P) {
	// Calculate the row index of the d_Pelement and d_M
	int Row = blockIdx.y*blockDim.y+threadIdx.y;
	// Calculate the column index of d_P and d_N
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	if ((Row < M) && (Col < N)) {
		float Pvalue = 0;
		// each thread computes one element of the block sub-matrix
		for (int k = 0; k < P; ++k) {
			Pvalue += d_M[Row*P+k]*d_N[k*N+Col];
		}
		d_P[Row*N+Col] = Pvalue + d_B[Row*N+Col];
	}
}

__global__ void transpose(float * d_N, float * d_M, int M, int N){
    int Row = blockIdx.y*blockDim.y+threadIdx.y;
	// Calculate the column index of d_P and d_N
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	if ((Row < M) && (Col < N)) {
        d_N[Col * M + Row] = d_M[Row * N + Col];
    }
}

__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int M, int N, int P) {
	// Calculate the row index of the d_Pelement and d_M
	int Row = blockIdx.y*blockDim.y+threadIdx.y;
	// Calculate the column index of d_P and d_N
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	if ((Row < M) && (Col < N)) {
		float Pvalue = 0;
		// each thread computes one element of the block sub-matrix
		for (int k = 0; k < P; ++k) {
			Pvalue += d_M[Row*P+k]*d_N[k*N+Col];
		}
		d_P[Row*N+Col] = Pvalue;
	}
}

__global__ void MatrixSubtractionKernel(float* d_M, float* d_N, float lr, int M, int N){
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	if((Row < M) && (Col < N)){
		d_M[Row*N+Col] = d_M[Row*N+Col] - d_N[Row*N + Col] * lr;
	}
}

__global__ void ActivationForwardPass(int pairNum, float * arr, int M, int N){
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if((Row < M) && (Col < N)){
        switch(pairNum){
            case 1:
                arr[Row * N + Col] = tanh2(arr[Row * N + Col]);
                break;
            case 2:
                arr[Row * N + Col] = sigmoid(arr[Row * N + Col]);
                break;
            case 3:
                arr[Row * N + Col] = relu(arr[Row * N + Col]);
                break;
        }
    }
}

__global__ void ActivationBackPass(int pairNum, float * arr, int M, int N){
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if((Row < M) && (Col < N)){
        switch(pairNum){
            case 1:
                arr[Row * N + Col] = tanh_prime(arr[Row * N + Col]);
                break;
            case 2:
                arr[Row * N + Col] = sigmoid_prime(arr[Row * N + Col]);
                break;
            case 3:
                arr[Row * N + Col] = relu_prime(arr[Row * N + Col]);
                break;
        }
    }
}

__global__ void element_wise_mul(float * d_M, float * d_N, int M, int N){
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if((Row < M) && (Col < N)){
        d_M[Row * N + Col] = d_M[Row * N + Col] * d_N[Row * N + Col];
    }
}


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
)

    // Eigen::MatrixXf inputError = outputError * weights.transpose(); //calculates dE/dx 
    // Eigen::MatrixXf weightsError = input.transpose() * outputError; //calculates dE/dW

    // //update parameters
    // weights -= weightsError * learningRate;
    // bias -= outputError * learningRate; 

{
    dim3 block_size(32, 32, 1);
    dim3 grid_size;

    grid_size.x = (w_r + block_size.x - 1) / block_size.x;
    grid_size.y = (w_c + block_size.y - 1) / block_size.y;
    transpose<<<grid_size, block_size>>>(d_weights_T, d_weights, w_r, w_c);

    grid_size.x = (i_r + block_size.x - 1) / block_size.x;
    grid_size.y = (i_c + block_size.y - 1) / block_size.y;
    transpose<<<grid_size, block_size>>>(d_input_T, d_input, i_r, i_c);

    __syncthreads();

    grid_size.x = (o_r + block_size.x - 1) / block_size.x;
    grid_size.y = (w_r + block_size.y - 1) / block_size.y;
    MatrixMulKernel<<<grid_size, block_size>>>(d_outputError, d_weights_T, d_inputError, o_r, w_r, w_c);

    grid_size.x = (i_c + block_size.x - 1) / block_size.x;
    grid_size.y = (o_c + block_size.y - 1) / block_size.y;
    MatrixMulKernel<<<grid_size, block_size>>>(d_weightsError, d_input_T, d_outputError, i_c, o_c, o_r);

    __syncthreads();

    grid_size.x = (w_r + block_size.x - 1) / block_size.x;
    grid_size.y = (w_c + block_size.y - 1) / block_size.y;
    MatrixSubtractionKernel<<<grid_size, block_size>>>(d_weights, d_weightsError, lr, w_r, w_c);

    grid_size.x = (o_r + block_size.x - 1) / block_size.x;
    grid_size.y = (o_c + block_size.y - 1) / block_size.y;
    MatrixSubtractionKernel<<<grid_size, block_size>>>(d_bias, d_outputError, lr, o_r, o_c);
}




