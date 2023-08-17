#include "backpropagation.h"
#include "benchmark.h"
#include <iostream>
#include <sys/resource.h>

bool setStackSize(rlim_t stackSize);

float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
float dsigmoid(float x) { return x * (1.f - x); }

using NetType = BPNet<float, sigmoid, dsigmoid, 500, 500, 500, 500, 500>;

static NetType net;
static const Matrix<float, 500, 1> input { 0 };
static const Matrix<float, 500, 1> output { 0 };

void call_train() {
    net.train(input, output);
}

void call_get() {
    net.get(input);
}

int main() {
    if (!setStackSize(2 * 4 * 1024 * 1024)) {
        std::cout << "Failed to increase stack size!" << std::endl;
	return 0;
    }

    std::cout << "Network currently has 1000000 weights and a total size of " << sizeof(NetType) << " bytes!" << std::endl;
    std::cout << "Train: ";
    benchmark<>(call_train, 100);
    std::cout << "Get: ";
    benchmark<>(call_get, 100);

    // End of program
    return 0;
}

bool setStackSize(rlim_t stackSize) {
    struct rlimit rl;

    if (getrlimit(RLIMIT_STACK, &rl) != 0) {
        return false;
    }

    if (rl.rlim_cur >= stackSize) {
        return true;
    }

    rl.rlim_cur = stackSize;

    return setrlimit(RLIMIT_STACK, &rl) == 0;
}
