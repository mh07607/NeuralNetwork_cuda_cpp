#ifndef ACTIVATION_INC
#define ACTIVATION_INC
#pragma once
#include <cmath>
#include <iostream>
#include <Eigen/Dense>


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
	return std::max(x, 0.0f);
}
__device__ float relu_prime(float x)
{
	return (float)((int)(x >= 0));
}

__device__ float one_minus(float x)
{
	return 1 - x;
}
//loss function and their derivative
float mse(Eigen::MatrixXf& y_true, Eigen::MatrixXf& y_pred)
{
	//printMatrixSize("y_true", y_true);
	//printMatrixSize("y_pred", y_pred);
	auto diff = (y_true - y_pred ).array() ;
	return  ( diff * diff).mean();
	//return ((y_true - y_pred) * (y_true - y_pred)).mean();
}

Eigen::MatrixXf mse_prime(Eigen::MatrixXf& y_true, Eigen::MatrixXf& y_pred)
{
	return  2 * (y_pred - y_true) / (y_true.rows()*y_true.cols());
}

/*
float binary_cross_entropy(Eigen::MatrixXf& y_true, Eigen::MatrixXf& y_pred)
{
	return  (-y_true * y_pred.log()).mean() - (y_true.unaryExpr(one_minus)) * (y_pred.unaryExpr(one_minus)).log());
}

Eigen::MatrixXf binary_cross_entropy_prime(Eigen::MatrixXf& y_true, Eigen::MatrixXf& y_pred)
{
	return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / y_true.size();
}
*/
#endif
