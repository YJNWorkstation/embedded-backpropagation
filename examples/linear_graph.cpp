#include "../backpropagation.h"
#include <iostream>
#include <sstream>

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

template<typename T, std::size_t M, std::size_t N>
void printMatrix(const Matrix<T, M, N>& matrix) {
    for (std::size_t m = 0; m < M; m++) {
        for (std::size_t n = 0; n < N; n++) {
	    std::cout << to_string_with_precision(matrix(m, n), 2) << " ";
	}
	std::cout << std::endl;
    }
}

int main() {
    Matrix<float, 16, 8> m{ 0.f };

    printMatrix(m);
    std::cout << std::endl;
    printMatrix(~m);

    return 0;
}
