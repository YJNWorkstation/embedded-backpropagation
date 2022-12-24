#pragma once
#include <functional>
#include <cstddef>
#include <random>

template<typename T>
T random(T min, T max) {
    return (static_cast<T>(rand()) / static_cast<T>(RAND_MAX)) * (max - min) + min;
}

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
    
    // randomize
    void randomize(T min, T max);

    // scalar multiplication
    Matrix<T, M, N>& operator*=(T rhs);
    Matrix<T, M, N> operator*(T rhs) const;

    // elementwise multiplication
    Matrix<T, M, N>& operator*=(const Matrix<T, M, N>& rhs);

    // scalar addition
    Matrix<T, M, N>& operator+=(T rhs);
    Matrix<T, M, N> operator+(T rhs) const;

    // matrix addition
    Matrix<T, M, N>& operator+=(const Matrix<T, M, N>& rhs);
    Matrix<T, M, N> operator+(const Matrix<T, M, N>& rhs) const;
    
    // matrix subtraction
    Matrix<T, M, N>& operator-=(const Matrix<T, M, N>& rhs);
    Matrix<T, M, N> operator-(const Matrix<T, M, N>& rhs) const;

    // map
    Matrix<T, M, N>& operator<<=(std::function<T(T)> f);
    Matrix<T, M, N> operator<<(std::function<T(T)> f) const;

    // indexing
    T& operator()(std::size_t m, std::size_t n);
    const T& operator()(std::size_t m, std::size_t n) const;

    // transpose
    Matrix<T, N, M> operator~() const;
    
    // matrix multiplication
    template<std::size_t J, std::size_t K>
    Matrix<T, M, K> operator*(const Matrix<T, J, K>& rhs);
};

template<typename T, T(*Activation)(T), T(*Derivative)(T), std::size_t I, std::size_t... L>
class BPNet {
private:
    using SubNetType = BPNet<T, Activation, Derivative, L...>;
public:
    static constexpr T(*ACTIVATION)(T) = Activation;
    static constexpr T(*DERIVATIVE)(T) = Derivative;
    static constexpr std::size_t INPUTS = I;
    static constexpr std::size_t OUTPUTS = SubNetType::OUTPUTS;
    static constexpr T LEARNING_RATE = static_cast<T>(0.002);
private:
    static constexpr std::size_t NEXT = SubNetType::INPUTS;

    Matrix<T, NEXT, INPUTS> m_weight;
    Matrix<T, NEXT, 1> m_bias;
    SubNetType m_sub;
public:
    void randomize(T min, T max) {
        m_weight.randomize(min, max);
	m_bias.randomize(min, max);
	m_sub.randomize(min, max);
    }

    Matrix<T, OUTPUTS, 1> get(const Matrix<T, INPUTS, 1>& input) {
	return m_sub.get(((m_weight * input) + m_bias) << ACTIVATION);
    }

    Matrix<T, INPUTS, 1> train(const Matrix<T, INPUTS, 1>& input, const Matrix<T, OUTPUTS, 1>& output) {
        Matrix<T, NEXT, 1> temp = ((m_weight * input) + m_bias) << ACTIVATION;
        Matrix<T, NEXT, 1> errors = m_sub.train(temp, output);
        
	Matrix<T, NEXT, 1> gradient = temp << DERIVATIVE;
        gradient *= errors;
	gradient *= LEARNING_RATE;
	
	Matrix<T, 1, INPUTS> hidden = ~input;
	Matrix<T, NEXT, INPUTS> deltas = gradient * hidden;

	m_weight += deltas;
	m_bias += gradient;

	return ~m_weight * errors;
    }
};

template<typename T, T(*Activation)(T), T(*Derivative)(T), std::size_t O>
class BPNet<T, Activation, Derivative, O> {
public:
    static constexpr T(*ACTIVATION)(T) = Activation;
    static constexpr T(*DERIVATIVE)(T) = Derivative;
    static constexpr std::size_t INPUTS = O;
    static constexpr std::size_t OUTPUTS = O;
    static constexpr T LEARNING_RATE = static_cast<T>(0.002);
public:
    void randomize(T min, T max) {}
    Matrix<T, OUTPUTS, 1> get(const Matrix<T, INPUTS, 1>& input) { return input; } 
    Matrix<T, INPUTS, 1> train(const Matrix<T, INPUTS, 1>& input, const Matrix<T, OUTPUTS, 1>& output) { return output - input; }
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
void Matrix<T, M, N>::randomize(T min, T max) {
    for (std::size_t m = 0; m < M; m++) {
        for (std::size_t n = 0; n < N; n++) {
            data[m][n] = random<T>(min, max);
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
Matrix<T, M, N> Matrix<T, M, N>::operator*(T rhs) const {
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
Matrix<T, M, N> Matrix<T, M, N>::operator+(T rhs) const {
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
Matrix<T, M, N> Matrix<T, M, N>::operator+(const Matrix<T, M, N>& rhs) const {
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
Matrix<T, M, N> Matrix<T, M, N>::operator-(const Matrix<T, M, N>& rhs) const {
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
Matrix<T, M, N> Matrix<T, M, N>::operator<<(std::function<T(T)> f) const {
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
