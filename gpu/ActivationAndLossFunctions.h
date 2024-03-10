#ifndef ACTIVATION_INC
#define ACTIVATION_INC
#pragma once
#include <cmath>
#include <iostream>
#include <Eigen/Dense>



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
