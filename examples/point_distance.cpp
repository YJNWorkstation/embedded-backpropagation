#include "../backpropagation.h"
#include <iostream>
#include <sstream>
#include <cmath>

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double dsigmoid(double x) { return x * (1.0 - x); }

// Define max distance of points at which the network should return 1
static const double point_distance = 0.3;

// Tell wether points x and y are within the specified distance
bool oracle(double x, double y) {
    return std::abs(x - y) <= point_distance;
}

int main() {
    // Define neural network structure
    using NetType = BPNet<double, sigmoid, dsigmoid, 2, 4, 4, 1>;
    static constexpr std::size_t TrainingCycles = 1000000; // Choose higher number for likely better results
    static constexpr std::size_t ControlCycles = 1000;

    // Set seed of random number generator
    srand(static_cast<unsigned>(time(0)));

    // Create neural network instance and randomize matrices, set learning rate
    // Choose lower learning rate and higher TrainingCycles for more precise results
    NetType net;
    net.setLearningRate(0.01);
    net.randomize(0.0, 1.0);

    // Train network with data (known input-output relations)
    for (std::size_t i = 0; i < TrainingCycles; i++) {
        // Generate random inputs
	Matrix<double, 2, 1> inputs;
	inputs.randomize(0.0, 1.0);

	// Wanted output (1 if within distance, 0 if outside distance)
	Matrix<double, 1, 1> outputs;
	outputs(0, 0) = oracle(inputs(0, 0), inputs(1, 0)) ? 1.0 : 0.0;

	// Train network and get input errors as result
	Matrix<double, 2, 1> errors = net.train(inputs, outputs);
    }

    // Feed input data into the neural network and calculate success rate
    std::size_t correct = 0;
    std::size_t within = 0;
    double average_error = 0.0;

    for (std::size_t i = 0; i < ControlCycles; i++) {
        // Generate random inputs
	Matrix<double, 2, 1> inputs;
        inputs.randomize(0.0, 1.0);

        // Wanted output (1 if within distance, 0 if outside distance)
        double actual = oracle(inputs(0, 0), inputs(1, 0)) ? 1.0 : 0.0;
        double wrong = !oracle(inputs(0, 0), inputs(1, 0)) ? 1.0 : 0.0;

        // Count how many inputs are within the circle
	if (oracle(inputs(0, 0), inputs(1, 0))) {
	    within++;
	}

	// Get network output
	double output = net.get(inputs)(0, 0);
        
	// Error
        average_error += std::abs(actual - output) / static_cast<double>(ControlCycles);

	// Success if output was closer to correct result
	if (std::abs(actual - output) < std::abs(wrong - output)) {
	    correct++;
	}
    }

    // Success calculations
    double success_percentage = static_cast<double>(correct) / static_cast<double>(ControlCycles) * 100.0;
    double within_percentage = static_cast<double>(within) / static_cast<double>(ControlCycles) * 100.0;

    // Output result
    std::cout << "Of the " << ControlCycles << " control runs (" << within_percentage << "% withing the distance) " 
	    << success_percentage << "% resulted in the correct output with an average error of " << average_error << "." << std::endl;

    // End of program
    return 0;
}
