#pragma once
#include <functional>
#include <cstddef>

template<typename T, std::size_t M, std::size_t N>
class Matrix {
public:
    static constexpr std::size_t ROWS = M;
    static constexpr std::size_t COLUMNS = N;
private:
    T data[M][N];
public:
    Matrix() = default;
    Matrix(T v);
    Matrix(std::function<T(std::size_t, std::size_t)> f); 
    
    // scalar multiplication
    Matrix<T, M, N>& operator*=(T rhs);
    Matrix<T, M, N> operator*(T rhs);

    // elementwise multiplication
    Matrix<T, M, N>& operator*=(const Matrix<T, M, N>& rhs);

    // scalar addition
    Matrix<T, M, N>& operator+=(T rhs);
    Matrix<T, M, N> operator+(T rhs);

    // matrix addition
    Matrix<T, M, N>& operator+=(const Matrix<T, M, N>& rhs);
    Matrix<T, M, N> operator+(const Matrix<T, M, N>& rhs);
    
    // matrix subtraction
    Matrix<T, M, N>& operator-=(const Matrix<T, M, N>& rhs);
    Matrix<T, M, N> operator-(const Matrix<T, M, N>& rhs);

    // map
    Matrix<T, M, N>& operator<<=(std::function<T(T)> f);
    Matrix<T, M, N> operator<<(std::function<T(T)> f);

    // indexing
    T& operator()(std::size_t m, std::size_t n);
    const T& operator()(std::size_t m, std::size_t n) const;

    // transpose
    Matrix<T, N, M> operator~() const;
    
    // matrix multiplication
    template<std::size_t J, std::size_t K>
    Matrix<T, M, K> operator*(const Matrix<T, J, K>& rhs);
};

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N>::Matrix(T v) {
    for (std::size_t m = 0; m < M; m++) {
        for (std::size_t n = 0; n < N; n++) {
	    data[m][n] = v;
	}
    }
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N>::Matrix(std::function<T(std::size_t, std::size_t)> f) {
    for (std::size_t m = 0; m < M; m++) {
        for (std::size_t n = 0; n < N; n++) {
	    data[m][n] = f(m, n);
	}
    }
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N>& Matrix<T, M, N>::operator*=(T rhs) {
    for (std::size_t m = 0; m < M; m++) {
        for (std::size_t n = 0; n < N; n++) {
	    data[m][n] *= rhs;
	}
    }
    return *this;
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> Matrix<T, M, N>::operator*(T rhs) {
    Matrix<T, M, N> result = *this;
    result *= rhs;
    return result;
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N>& Matrix<T, M, N>::operator*=(const Matrix<T, M, N>& rhs) {
    for (std::size_t m = 0; m < M; m++) {
        for (std::size_t n = 0; n < N; n++) {
            data[m][n] *= rhs(m, n);
        }
    }
    return *this;
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N>& Matrix<T, M, N>::operator+=(T rhs) {
    for (std::size_t m = 0; m < M; m++) {
        for (std::size_t n = 0; n < N; n++) {
            data[m][n] += rhs;
        }
    }
    return *this;
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> Matrix<T, M, N>::operator+(T rhs) {
    Matrix<T, M, N> result = *this;
    result += rhs;
    return result;
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N>& Matrix<T, M, N>::operator+=(const Matrix<T, M, N>& rhs) {
    for (std::size_t m = 0; m < M; m++) {
        for (std::size_t n = 0; n < N; n++) {
            data[m][n] += rhs(m, n);
        }
    }
    return *this;
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> Matrix<T, M, N>::operator+(const Matrix<T, M, N>& rhs) {
    Matrix<T, M, N> result = *this;
    result += rhs;
    return result;
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N>& Matrix<T, M, N>::operator-=(const Matrix<T, M, N>& rhs) {
    for (std::size_t m = 0; m < M; m++) {
        for (std::size_t n = 0; n < N; n++) {
            data[m][n] -= rhs(m, n);
        }
    }
    return *this;
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> Matrix<T, M, N>::operator-(const Matrix<T, M, N>& rhs) {
    Matrix<T, M, N> result = *this;
    result -= rhs;
    return result;
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N>& Matrix<T, M, N>::operator<<=(std::function<T(T)> f) {
    for (std::size_t m = 0; m < M; m++) {
        for (std::size_t n = 0; n < N; n++) {
            data[m][n] = f(data[m][n]);
        }
    }
    return *this;
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> Matrix<T, M, N>::operator<<(std::function<T(T)> f) {
    Matrix<T, M, N> result = *this;
    result <<= f;
    return result;
}

template<typename T, std::size_t M, std::size_t N>
T& Matrix<T, M, N>::operator()(std::size_t m, std::size_t n) {
    return data[m][n];
}

template<typename T, std::size_t M, std::size_t N>
const T& Matrix<T, M, N>::operator()(std::size_t m, std::size_t n) const {
    return data[m][n];
}

template<typename T, std::size_t M, std::size_t N>
Matrix<T, N, M> Matrix<T, M, N>::operator~() const {
    Matrix<T, N, M> result;
    for (std::size_t m = 0; m < M; m++) {
        for (std::size_t n = 0; n < N; n++) {
            result(n, m) = data[m][n];
        }
    }
    return result;
}

template<typename T, std::size_t M, std::size_t N>
template<std::size_t J, std::size_t K>
Matrix<T, M, K> Matrix<T, M, N>::operator*(const Matrix<T, J, K>& rhs) {
    Matrix<T, M, K> result;
    for (std::size_t i = 0; i < Matrix<T, M, K>::ROWS; i++) {
        for (std::size_t j = 0; j < Matrix<T, M, K>::COLUMNS; j++) {
	    T result_entry = static_cast<T>(0.0);
	    for (std::size_t k = 0; k < Matrix<T, M, N>::COLUMNS; k++) {
	        result_entry += (*this)(i, k) * rhs(k, j);
	    }
	    result(i, j) = result_entry;
	}
    }
    return result;
}

/*
 * T = either float or double
 * A = activation function
 * D = derivative of activation function
 * L = layer sizes
 */
//template<typename T, std::function<> A, std::function<> D, size_t... L>
