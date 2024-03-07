#ifndef NEURAL_NET_INC
#define NEURAL_NET_INC
#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <numeric> //std::iota

__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, float * d_B, int M, int N, int P) {
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
	DenseLayer(int inputSize, int  outputSize)
	{
		//Eigen::MatrixXf::Random returns values from [-1,1] we should scale it to [-0.5,0.5]
		weights = Eigen::MatrixXf::Random(inputSize, outputSize).array() * 0.5f;
		bias = Eigen::MatrixXf::Random(1, outputSize).array() * 0.5f; 
	}

	Eigen::MatrixXf forwardPropagation(Eigen::MatrixXf& input)
	{
		// this->input = input;  
		// this->output = input * weights + bias;  
		// std::cout << "Actual output: " << this->output << std::endl;

		this->input = input;
		float * output_arr;
		int output_size = input.rows() * weights.cols() * sizeof(float);
		// std::cout << "Output size: " << input.rows() << weights.cols() << weights.rows() << output_size << std::endl;
		float * h_output_arr = (float *)malloc(output_size);
		cudaMalloc((void **)&output_arr, output_size);
		dim3 block_size(input.rows(), weights.cols(), 1);
		MatrixMulKernel<<<1, block_size>>>
		(input.data(), weights.data(), output_arr, bias.data(), input.rows(), weights.cols(), weights.rows());
		cudaDeviceSynchronize();
		cudaMemcpy(h_output_arr, output_arr, output_size, cudaMemcpyDeviceToHost);
		this->output = Eigen::MatrixXf::Map(h_output_arr, input.rows(), weights.cols());
		// std::cout << "My kernel output: " << this->output << std::endl;
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