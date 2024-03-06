#ifndef NEURAL_NET_INC
#define NEURAL_NET_INC
#pragma once

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cuda_runtime.h>
#include <numeric> //std::iota

__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int M, int N) {
	// Calculate the row index of the d_Pelement and d_M
	int Row = blockIdx.y*blockDim.y+threadIdx.y;
	// Calculate the column index of d_P and d_N
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	if ((Row < M) && (Col < N)) {
		float Pvalue = 0;
		// each thread computes one element of the block sub-matrix
		for (int k = 0; k < Width; ++k) {
			Pvalue += d_M[Row*M+k]*d_N[k*M+Col];
		}
		d_P[Row*M+Col] = Pvalue;
	}
}

void MatrixMul(float * d_M, float * d_N, float * d_P, float * d_b; int M, int N){
	for(int j = 0; j < N; j++){
		for(int i = 0; i < M; i++){
			d_P[i*M + j] = 0;
			for(int k = 0; k < M; k++){
				d_P[i*m += 
			}
		}
	}
}

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

public:
	Eigen::MatrixXf input;
	Eigen::MatrixXf output;
};


class DenseLayer : public Layer
{
public:
	DenseLayer(int inputSize, int  outputSize)
	{
		//Eigen::MatrixXf::Random returns values from [-1,1] we should scale it to [-0.5,0.5]
		weights = Eigen::MatrixXf::Random(inputSize, outputSize).array() * 0.5f;
		bias = Eigen::MatrixXf::Random(1, outputSize).array() * 0.5f; 
	}

	Eigen::MatrixXf forwardPropagation(Eigen::MatrixXf& input)
	{
		this->input = input;  
		this->output = input * weights + bias;  
		return this->output;
	}

	//computes dE/dW, dE/dB for a given outputError = dE/dY. Returns input_error = dE/dX.
	Eigen::MatrixXf backwardPropagation(Eigen::MatrixXf& outputError, float learningRate)
	{
		Eigen::MatrixXf inputError = outputError * weights.transpose(); //calculates dE/dx 
		Eigen::MatrixXf weightsError = input.transpose() * outputError; //calculates dE/dW

		//update parameters
		weights -= weightsError * learningRate;
		bias -= outputError * learningRate; 

		return inputError;
	}

public:
	Eigen::MatrixXf weights;
	Eigen::MatrixXf bias;
};

class ActivationLayer : public Layer
{
public:
	ActivationLayer(std::function<float(float)> activation,
		std::function<float(float)> activationPrime)
	{
		type = 1;
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

public:
	std::function<float(float)> activation;
	std::function<float(float)> activationPrime;
};

class FlattenLayer :public Layer
{
public:
	Eigen::MatrixXf forwardPropagation(Eigen::MatrixXf& input)
	{
		type = 2;
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



class GPUNetwork
{
public:
	GPUNetwork() {}
	virtual ~GPUNetwork() {}

	void add(Layer* layer)
	{
		layers.push_back(layer);
		Layer * device_layer;
		cudaMalloc((void**)&device_layer, sizeof(Layer));
		cudaMemcpy(device_layer, layer, sizeof(Layer), cudaMemcpyHostToDevice);
		device_layers.push_back(device_layer);
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
			    // Eigen::MatrixXf output = x_train.row(index);
				Eigen::MatrixXf * d_output;
				int dataSize = x_train.row(index).size() * sizeof(float);
				Eigen::MatrixXf * h_output = (Eigen::MatrixXf *) malloc(dataSize);
				printf("x_train row size: %d\n", dataSize);
				cudaMalloc((void **)&d_output, dataSize);
				cudaMemcpy(d_output, x_train.row(index).data(), dataSize, cudaMemcpyHostToDevice);

				int num_layers = layers.size();
				gpuNetwork_forwardPropagation<<<1, 1>>>(device_layers.data(), num_layers, d_output);
				// for (Layer* layer : layers)				 	
				//  	output = layer->forwardPropagation(output);
				cudaMemcpy(h_output, d_output, dataSize, cudaMemcpyDeviceToHost);	  
				// compute loss(for display purpose only)
				Eigen::MatrixXf y = y_train.row(index);
				
				err += loss(y, *h_output);
				
				//backward propagation 
				Eigen::MatrixXf error = lossPrime(y, *h_output); 

				for (std::vector<Layer*>::reverse_iterator layer = layers.rbegin(); layer != layers.rend(); ++layer) 
					error = (*layer)->backwardPropagation(error, learningRate); 
				 
			}
			err /= (float)samples;
			std::cout << "Epoch " << (i + 1) << "/" << epochs << " error = " << err << std::endl;
		}
	}

protected:
	std::vector<Layer*> layers;
	// vector to store layers in device
	std::vector<Layer*> device_layers;
	std::function<float(Eigen::MatrixXf&, Eigen::MatrixXf&)> loss;
	std::function<Eigen::MatrixXf(Eigen::MatrixXf&, Eigen::MatrixXf&)> lossPrime;
};
#endif