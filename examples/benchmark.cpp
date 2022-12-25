#include "../backpropagation.h"
#include "benchmark.h"
#include <iostream>

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double dsigmoid(double x) { return x * (1.0 - x); }

using NetType = BPNet<double, sigmoid, dsigmoid, 2, 256, 256, 256, 256, 1>;

static NetType net;
static const Matrix<double, 2, 1> input { 0 };
static const Matrix<double, 1, 1> output { 0 };

void call_train() {
    net.train(input, output);
}

void call_get() {
    net.get(input);
}

int main() {
    std::cout << "Train:" << std::endl;
    benchmark<>(call_train, 100);
    std::cout << "Get:" << std::endl;
    benchmark<>(call_get, 100);

    // End of program
    return 0;
}
