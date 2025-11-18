#ifndef VALARRAY_INC
#define VALARRAY_INC

#include <cstddef>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <vector>

namespace std {

// Custom copy function with vectorization pragma
template<typename InputIt, typename OutputIt>
OutputIt copy1(InputIt first, InputIt last, OutputIt d_first) {
    #pragma vector
    while (first != last) {
        *d_first++ = *first++;
    }
    return d_first;
}

// Custom accumulate function with vectorization pragma
template<typename InputIt, typename T>
T accumulate1(InputIt first, InputIt last, T init) {
    #pragma vector
    for (; first != last; ++first) {
        init = init + *first;
    }
    return init;
}

// Forward declaration
template<typename T> class valarray;

// std::slice implementation
class slice {
public:
    slice() : start_(0), size_(0), stride_(1) {}

    slice(size_t start, size_t size, size_t stride)
        : start_(start), size_(size), stride_(stride) {}

    size_t start() const { return start_; }
    size_t size() const { return size_; }
    size_t stride() const { return stride_; }

private:
    size_t start_;
    size_t size_;
    size_t stride_;
};

// Proxy class for slice access
template<typename T>
class slice_array {
public:
    slice_array(valarray<T>* arr, const slice& s)
        : arr_(arr), slice_(s) {}

    // Assignment from valarray
    slice_array& operator=(const valarray<T>& other) {
        size_t idx = slice_.start();
        #pragma vector
        for (size_t i = 0; i < slice_.size(); ++i) {
            (*arr_)[idx] = other[i];
            idx += slice_.stride();
        }
        return *this;
    }

    // Compound assignment operators
    slice_array& operator+=(const valarray<T>& other) {
        size_t idx = slice_.start();
        #pragma vector
        for (size_t i = 0; i < slice_.size(); ++i) {
            (*arr_)[idx] += other[i];
            idx += slice_.stride();
        }
        return *this;
    }

    slice_array& operator-=(const valarray<T>& other) {
        size_t idx = slice_.start();
        #pragma vector
        for (size_t i = 0; i < slice_.size(); ++i) {
            (*arr_)[idx] -= other[i];
            idx += slice_.stride();
        }
        return *this;
    }

    slice_array& operator*=(const valarray<T>& other) {
        size_t idx = slice_.start();
        #pragma vector
        for (size_t i = 0; i < slice_.size(); ++i) {
            (*arr_)[idx] *= other[i];
            idx += slice_.stride();
        }
        return *this;
    }

private:
    valarray<T>* arr_;
    slice slice_;
};

// Main valarray implementation - wrapper around std::vector
template<typename T>
class valarray {
public:
    // Constructors
    valarray() : data_() {}

    explicit valarray(size_t n) : data_(n, T()) {}

    valarray(const T& val, size_t n) : data_(n, val) {}

    valarray(const T* p, size_t n) : data_(n) {
        #pragma vector
        for (size_t i = 0; i < n; ++i) {
            data_[i] = p[i];
        }
    }

    // Copy constructor
    valarray(const valarray& other) : data_(other.data_) {}

    // Move constructor
    valarray(valarray&& other) noexcept : data_(std::move(other.data_)) {}

    // Destructor (vector handles cleanup automatically)
    ~valarray() = default;

    // Assignment operators
    valarray& operator=(const valarray& other) {
        if (this != &other) {
            data_ = other.data_;
        }
        return *this;
    }

    valarray& operator=(valarray&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
        }
        return *this;
    }

    valarray& operator=(const T& val) {
        #pragma vector
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] = val;
        }
        return *this;
    }

    // Element access
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

    // Slice access
    slice_array<T> operator[](const slice& s) {
        return slice_array<T>(this, s);
    }

    // Size and resize
    size_t size() const { return data_.size(); }

    void resize(size_t n, T val = T()) {
        data_.resize(n, val);
    }

    // Sum
    T sum() const {
        T result = T();
        #pragma vector
        for (size_t i = 0; i < data_.size(); ++i) {
            result += data_[i];
        }
        return result;
    }

    // Compound assignment with scalar
    valarray& operator*=(const T& val) {
        #pragma vector
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] *= val;
        }
        return *this;
    }

    valarray& operator/=(const T& val) {
        #pragma vector
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] /= val;
        }
        return *this;
    }

    valarray& operator+=(const T& val) {
        #pragma vector
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] += val;
        }
        return *this;
    }

    valarray& operator-=(const T& val) {
        #pragma vector
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] -= val;
        }
        return *this;
    }

    // Compound assignment with valarray
    valarray& operator*=(const valarray& other) {
        #pragma vector
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] *= other.data_[i];
        }
        return *this;
    }

    valarray& operator/=(const valarray& other) {
        #pragma vector
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] /= other.data_[i];
        }
        return *this;
    }

    valarray& operator+=(const valarray& other) {
        #pragma vector
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] += other.data_[i];
        }
        return *this;
    }

    valarray& operator-=(const valarray& other) {
        #pragma vector
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] -= other.data_[i];
        }
        return *this;
    }

    // Iterator support (for std::copy, std::begin, std::end)
    T* begin() { return data_.data(); }
    const T* begin() const { return data_.data(); }
    T* end() { return data_.data() + data_.size(); }
    const T* end() const { return data_.data() + data_.size(); }

private:
    std::vector<T> data_;
};

// Binary operators - valarray op valarray
template<typename T>
valarray<T> operator+(const valarray<T>& lhs, const valarray<T>& rhs) {
    valarray<T> result(lhs.size());
    #pragma vector
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] + rhs[i];
    }
    return result;
}

template<typename T>
valarray<T> operator-(const valarray<T>& lhs, const valarray<T>& rhs) {
    valarray<T> result(lhs.size());
    #pragma vector
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] - rhs[i];
    }
    return result;
}

template<typename T>
valarray<T> operator*(const valarray<T>& lhs, const valarray<T>& rhs) {
    valarray<T> result(lhs.size());
    #pragma vector
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] * rhs[i];
    }
    return result;
}

template<typename T>
valarray<T> operator/(const valarray<T>& lhs, const valarray<T>& rhs) {
    valarray<T> result(lhs.size());
    #pragma vector
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] / rhs[i];
    }
    return result;
}

// Binary operators - valarray op scalar
template<typename T>
valarray<T> operator+(const valarray<T>& lhs, const T& val) {
    valarray<T> result(lhs.size());
    #pragma vector
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] + val;
    }
    return result;
}

template<typename T>
valarray<T> operator-(const valarray<T>& lhs, const T& val) {
    valarray<T> result(lhs.size());
    #pragma vector
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] - val;
    }
    return result;
}

template<typename T>
valarray<T> operator*(const valarray<T>& lhs, const T& val) {
    valarray<T> result(lhs.size());
    #pragma vector
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] * val;
    }
    return result;
}

template<typename T>
valarray<T> operator/(const valarray<T>& lhs, const T& val) {
    valarray<T> result(lhs.size());
    #pragma vector
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] / val;
    }
    return result;
}

// Binary operators - scalar op valarray
template<typename T>
valarray<T> operator+(const T& val, const valarray<T>& rhs) {
    valarray<T> result(rhs.size());
    #pragma vector
    for (size_t i = 0; i < rhs.size(); ++i) {
        result[i] = val + rhs[i];
    }
    return result;
}

template<typename T>
valarray<T> operator-(const T& val, const valarray<T>& rhs) {
    valarray<T> result(rhs.size());
    #pragma vector
    for (size_t i = 0; i < rhs.size(); ++i) {
        result[i] = val - rhs[i];
    }
    return result;
}

template<typename T>
valarray<T> operator*(const T& val, const valarray<T>& rhs) {
    valarray<T> result(rhs.size());
    #pragma vector
    for (size_t i = 0; i < rhs.size(); ++i) {
        result[i] = val * rhs[i];
    }
    return result;
}

template<typename T>
valarray<T> operator/(const T& val, const valarray<T>& rhs) {
    valarray<T> result(rhs.size());
    #pragma vector
    for (size_t i = 0; i < rhs.size(); ++i) {
        result[i] = val / rhs[i];
    }
    return result;
}

// Mathematical functions
template<typename T>
valarray<T> sqrt(const valarray<T>& arr) {
    valarray<T> result(arr.size());
    #pragma vector
    for (size_t i = 0; i < arr.size(); ++i) {
        result[i] = std::sqrt(arr[i]);
    }
    return result;
}

template<typename T>
valarray<T> exp(const valarray<T>& arr) {
    valarray<T> result(arr.size());
    #pragma vector
    for (size_t i = 0; i < arr.size(); ++i) {
        result[i] = std::exp(arr[i]);
    }
    return result;
}

template<typename T>
valarray<T> tanh(const valarray<T>& arr) {
    valarray<T> result(arr.size());
    #pragma vector
    for (size_t i = 0; i < arr.size(); ++i) {
        result[i] = std::tanh(arr[i]);
    }
    return result;
}

template<typename T>
valarray<T> pow(const valarray<T>& arr, const T& exponent) {
    valarray<T> result(arr.size());
    #pragma vector
    for (size_t i = 0; i < arr.size(); ++i) {
        result[i] = std::pow(arr[i], exponent);
    }
    return result;
}

// begin/end free functions for valarray
template<typename T>
T* begin(valarray<T>& v) {
    return v.begin();
}

template<typename T>
const T* begin(const valarray<T>& v) {
    return v.begin();
}

template<typename T>
T* end(valarray<T>& v) {
    return v.end();
}

template<typename T>
const T* end(const valarray<T>& v) {
    return v.end();
}

} // namespace std

#endif // VALARRAY_INC
