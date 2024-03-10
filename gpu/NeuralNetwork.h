#ifndef NEURAL_NET_INC
#define NEURAL_NET_INC
#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <numeric> //std::iota
#include "kernel_code.h"

void printMatrixSize(const std::string msg, const Eigen::MatrixXf& m)
{
	std::cout << msg.c_str() << "[" << m.rows() << "," << m.cols() << "]" << std::endl;
}

class Layer
{
public:
	Layer() :input(), output() {}
	virtual ~Layer() {}

	virtual Eigen::MatrixXf forwardPropagation(Eigen::MatrixXf& input) = 0;
	virtual Eigen::MatrixXf backwardPropagation(Eigen::MatrixXf& output, float learningRate) = 0;

protected:
	Eigen::MatrixXf input;
	Eigen::MatrixXf output;
};

class DenseLayer : public Layer
{
public:
	DenseLayer(int inputSize, int  outputSize) : Layer()
	{
		//Eigen::MatrixXf::Random returns values from [-1,1] we should scale it to [-0.5,0.5]
		weights = Eigen::MatrixXf::Random(inputSize, outputSize).array() * 0.5f;
		bias = Eigen::MatrixXf::Random(1, outputSize).array() * 0.5f;
		output = Eigen::MatrixXf::Random(1, outputSize).array();
	}

	Eigen::MatrixXf forwardPropagation(Eigen::MatrixXf& input)
	{
		this->input = input;
		float * output_arr;
		int output_size = input.rows() * weights.cols() * sizeof(float);
		// std::cout << "Output size: " << input.rows() << weights.cols() << weights.rows() << output_size << std::endl;
		float * d_input;
		float * d_weights;
		float * d_bias;

		cudaMalloc((void **)&output_arr, output_size);
		cudaMalloc((void **)&d_input, input.rows() * input.cols() * sizeof(float));
		cudaMalloc((void **)&d_weights, weights.rows() * weights.cols() * sizeof(float));
		cudaMalloc((void **)&d_bias, bias.rows() * bias.cols() * sizeof(float));

		cudaMemcpy(d_input, input.data(), input.rows() * input.cols() * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_weights, weights.data(), weights.rows() * weights.cols() * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_bias, bias.data(), bias.rows() * bias.cols() * sizeof(float), cudaMemcpyHostToDevice);

		dim3 block_size(32, 32, 1);
		dim3 grid_size;
		grid_size.x = (input.rows() + block_size.x - 1) / block_size.x;
		grid_size.y = (weights.cols() + block_size.y - 1) / block_size.y;
		DenseForwardPass<<<grid_size, block_size>>>
		(d_input, d_weights, output_arr, d_bias, input.rows(), weights.cols(), weights.rows());
		cudaDeviceSynchronize();

		cudaError_t cudaError = cudaGetLastError();
		if(cudaError != cudaSuccess) {
			printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(cudaError));
			//return 1; // return an error code
		}

		cudaMemcpy(this->output.data(), output_arr, output_size, cudaMemcpyDeviceToHost);

		cudaFree(output_arr);
		cudaFree(d_input);
		cudaFree(d_weights);
		cudaFree(d_bias);

		return this->output;
	}

	//computes dE/dW, dE/dB for a given outputError = dE/dY. Returns input_error = dE/dX.
	Eigen::MatrixXf backwardPropagation(Eigen::MatrixXf& outputError, float learningRate)
	{
		// float * d_outputError;
		// float * d_input;
		// float * d_bias;
		// float * d_weights;
		// float * d_inputError;
		// float * d_weightsError;
		// float * d_weights_T;
		// float * d_input_T;

		// int outputError_size = outputError.rows() * outputError.cols() * sizeof(float);
		// int input_size = input.rows() * input.cols() * sizeof(float);
		// int weights_size = weights.rows() * weights.cols() * sizeof(float);
		// int bias_size = bias.rows() * bias.cols() * sizeof(float);
		// int inputError_size = outputError.rows() * weights.rows() * sizeof(float);
		// int weightsError_size = input.cols() * outputError.cols() * sizeof(float);

		// // For retrieving the input error on host
		// float * h_inputError = (float *) malloc (inputError_size);


		// cudaMalloc((void **) &d_outputError, outputError_size);
		// cudaMalloc((void **) &d_input, input_size);
		// cudaMalloc((void **) &d_weights, weights_size);
		// cudaMalloc((void **) &d_inputError, inputError_size);
		// cudaMalloc((void **) &d_weightsError, weightsError_size);
		// cudaMalloc((void **) &d_bias, bias_size);
		// cudaMalloc((void **) &d_input_T, input_size);
		// cudaMalloc((void **) &d_weights_T, weights_size);

		// cudaMemcpy(d_input, input.data(), input_size, cudaMemcpyHostToDevice);
		// cudaMemcpy(d_weights, weights.data(), weights_size, cudaMemcpyHostToDevice);
		// cudaMemcpy(d_outputError, outputError.data(), outputError_size, cudaMemcpyHostToDevice);

		
		// DenseBackwardPass<<<1, 1>>>
		// (d_outputError,
	    //  d_input,
		//  d_weights,
		//  d_bias,
		//  d_inputError,
		//  d_weightsError,
		//  d_weights_T,
		//  d_input_T,
		//  learningRate,
		//  outputError.rows(),
		//  outputError.cols(),
		//  input.rows(),
		//  input.cols(),
		//  weights.rows(),
		//  weights.cols()
		//  );

		
		// cudaDeviceSynchronize();
		
		// // cudaError_t cudaError = cudaGetLastError();
		// // if(cudaError != cudaSuccess) {
		// // 	printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(cudaError));
		// // 	//return 1; // return an error code
		// // }

		// cudaMemcpy(weights.data(), d_weights, weights_size, cudaMemcpyDeviceToHost);
		// cudaMemcpy(bias.data(), d_bias, weights_size, cudaMemcpyDeviceToHost);
		// cudaMemcpy(h_inputError, d_inputError, inputError_size, cudaMemcpyDeviceToHost);

		// Eigen::MatrixXf inputError = Eigen::MatrixXf::Map(h_inputError, outputError.rows(), weights.rows());

			

		// free(h_inputError);
		// cudaFree(d_input);
		// cudaFree(d_weights);
		// cudaFree(d_bias);
		// cudaFree(d_outputError);
		// cudaFree(d_inputError);

		// return inputError;

		Eigen::MatrixXf inputError = outputError * weights.transpose(); //calculates dE/dx 
		Eigen::MatrixXf weightsError = input.transpose() * outputError; //calculates dE/dW

		//update parameters
		weights -= weightsError * learningRate;
		bias -= outputError * learningRate; 

		return inputError;
	}

private:
	Eigen::MatrixXf weights;
	Eigen::MatrixXf bias;
};

class ActivationLayer : public Layer
{
public:
	ActivationLayer(std::function<float(float)> activation,
		std::function<float(float)> activationPrime)
	{
		this->activation = activation;
		this->activationPrime = activationPrime;
	}

	//returns the activated input
	Eigen::MatrixXf forwardPropagation(Eigen::MatrixXf& input)
	{
		this->input = input;
		this->output = input.unaryExpr(activation);
		return this->output;
	}

	//Returns inputRrror = dE / dX for a given output_error = dE / dY.
	//learningRate is not used because there is no "learnable" parameters.
	Eigen::MatrixXf backwardPropagation(Eigen::MatrixXf& outputError, float learningRate)
	{ 
		return (input.unaryExpr(activationPrime).array() * outputError.array()).matrix();
	}

private:
	std::function<float(float)> activation;
	std::function<float(float)> activationPrime;
};

class FlattenLayer :public Layer
{
public:
	Eigen::MatrixXf forwardPropagation(Eigen::MatrixXf& input)
	{
		this->input = input;
		this->output = input;
		this->output.resize(1, input.rows() * input.cols()); //flatten
		return this->output;
	}
	Eigen::MatrixXf backwardPropagation(Eigen::MatrixXf& outputError, float learningRate)
	{
		outputError.resize(input.rows(), input.cols());
		return outputError;
	}
};

class Network
{
public:
	Network() {}
	virtual ~Network() {}

	void add(Layer* layer)
	{
		layers.push_back(layer);
	}

	void use(std::function<float(Eigen::MatrixXf&, Eigen::MatrixXf&)> lossF, std::function<Eigen::MatrixXf(Eigen::MatrixXf&, Eigen::MatrixXf&)> lossDer)
	{
		loss = lossF;
		lossPrime = lossDer;
	}

	std::vector<Eigen::MatrixXf> predict(Eigen::MatrixXf input)
	{
		int samples = input.rows();

		std::vector<Eigen::MatrixXf> result;

		//forward propagation
		// this can be parallelized
		for (int j = 0; j < samples; ++j)
		{
			Eigen::MatrixXf output = input.row(j);
			for (Layer* layer : layers)
				output = layer->forwardPropagation(output);

			result.push_back(output);
		}

		return result;
	}

	//train the network
	virtual void fit(Eigen::MatrixXf x_train, Eigen::MatrixXf y_train, int epochs, float learningRate)
	{ 
		int samples = x_train.rows();
		std::cout << "Samples: " << samples << std::endl;
		printMatrixSize("x_train", x_train);
		printMatrixSize("y_train", y_train);

		std::vector<int> order(samples);
		std::iota(order.begin(), order.end(), 0);

		//training loop
		for (int i = 0; i < epochs; ++i)
		{
			float err = 0.0f;
			
			//feed forward
			std::random_shuffle(order.begin(), order.end());

			//forward propagation
			for (int j = 0; j < samples; ++j)
			{
				int index = order[j];
			    Eigen::MatrixXf output = x_train.row(index); 

				for (Layer* layer : layers)				 	
					output = layer->forwardPropagation(output);
					  
				// compute loss(for display purpose only)
				Eigen::MatrixXf y = y_train.row(index);
				   
				// std::cout << "done with layers" << std::endl;
				err += loss(y, output);
				// std::cout << "loss calculated" << std::endl;
				//backward propagation 
				Eigen::MatrixXf error = lossPrime(y, output); 
				// std::cout << "loss prime calculated" << std::endl;

				for (std::vector<Layer*>::reverse_iterator layer = layers.rbegin(); layer != layers.rend(); ++layer) 
					error = (*layer)->backwardPropagation(error, learningRate); 
					// std::cout << "layer backwards propagated" << std::endl;
				 
			}
			err /= (float)samples;
			std::cout << "Epoch " << (i + 1) << "/" << epochs << " error = " << err << std::endl;
		}
	}

protected:
	std::vector<Layer*> layers;
	std::function<float(Eigen::MatrixXf&, Eigen::MatrixXf&)> loss;
	std::function<Eigen::MatrixXf(Eigen::MatrixXf&, Eigen::MatrixXf&)> lossPrime;
};
#endif