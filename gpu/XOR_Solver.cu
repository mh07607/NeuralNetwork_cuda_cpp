#include <iostream> 
#include <vector>
#include "NeuralNetwork.h"
#include "ActivationAndLossFunctions.h"

int main()
{ 
	std::cout << "Using Eigen ver: " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << std::endl;

	//test the XOR solver
	Eigen::MatrixXf x_train{ {0, 0}, {0, 1}, {1, 0}, {1,1} };
	// For some reason, when you change the file extension from
	// .cpp to .cu, the following matrix size is 1x4 instead of
	// 4x1.
	//Eigen::MatrixXf y_train{ {0}, {1}, {1}, {0} };
	Eigen::MatrixXf y_train(4, 1);
	y_train << 0, 
			1, 
			1, 
			0;

	for (int i = 0; i < x_train.size(); i++)
  		std::cout << *(x_train.data() + i) << "  ";

	GPUNetwork nn;
	nn.add(new GPUDenseLayer(2, 3));
	// pairNum is a new parameter, 1 represents tanh2 and tanh_prime
	nn.add(new GPUActivationLayer(3, 1));
	nn.add(new GPUDenseLayer(3, 1));
	nn.add(new GPUActivationLayer(1, 1));

	nn.use(mse, mse_prime);
	
	//train
	nn.fit(x_train, y_train, 1000, 0.1f, "xor.txt");

	//test
	std::vector<Eigen::MatrixXf> output = nn.predict(x_train);
	for (Eigen::MatrixXf out : output)
		std::cout << out << std::endl; 

	return 0;
}