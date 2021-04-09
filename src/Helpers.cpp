#include <blaze/Blaze.h>

#include <vector>

using BVec = blaze::DynamicVector<double>;
using BMat = blaze::DynamicMatrix<double>;

namespace helpers {

// Helper function: Swaps the i-th and k-th row
void swap_rows(BMat& A, size_t i, size_t k) {
    for (size_t j = 0; j < A.columns(); ++j) {
        double tmp = A(i, j);
        A(i, j)    = A(k, j);
        A(k, j)    = tmp;
    }
}

// Finds the largest absolute element in column j starting from row i, returns
// the row index.
size_t arg_max_remainder(const BMat& A, size_t i, size_t j) {
    auto max_idx   = i;
    auto max_value = std::abs(A(i, j));
    for (size_t k = i; k < A.rows(); ++k) {
        if (std::abs(A(k, j)) > max_value) {
            max_idx   = k;
            max_value = std::abs(A(k, j));
        }
    }
    return max_idx;
};

// Fills column j starting from row i with zeros, i.e. A(i:n, j) = 0
void zero_out(BMat& A, size_t i, size_t j) {
    for (size_t k = i; k < A.rows(); ++k) { A(k, j) = 0.0; }
};

// TO DO:

// 1) Implementiere eine ref funktion, die (A,b) auf maximale zeilenrank
// umformt und somit anzahl an zeilen reduziert

// Produces the reduced row echelon form of A
// Based on matlabs rref, e.g. compare 'edit rref' in Matlab.
std::vector<size_t> rref(BMat& A, double tol = 1.0e-12) {
    auto m   = A.rows();
    auto n   = A.columns();
    size_t i = 0;
    size_t j = 0;
    std::vector<size_t> jb;
    while (i < m && j < n) {
        // Find value and index of largest element in the remainder of column j
        auto k = arg_max_remainder(A, i, j);
        auto p = std::abs(A(k, j));
        if (p <= tol) {
            // The column is negligible, zero it out
            zero_out(A, i, j);
            ++j;  // go to the next column
        } else {
            // Remember column index
            jb.push_back(j);
            // Swap i-th and k-th rows
            swap_rows(A, i, k);
            // Divide the pivot row by the pivot element
            auto pivot_row = blaze::row(A, i);
            pivot_row /= A(i, j);
            // Subtract multiples of the pivot row from all the other rows below
            // the pivot row
            for (size_t k = i + 1; k < m; ++k) {
                auto factor = A(k, j);
                for (size_t r = j; r < n; ++r) { A(k, r) -= factor * A(i, r); }
            }
            ++i;
            ++j;
        }
    }
    return jb;
}

// Free helper function for parsing
std::vector<double> flatten(const std::vector<std::vector<double>>& A) {
    std::size_t total_size = 0;
    for (const auto& row : A) { total_size += row.size(); }
    std::vector<double> result;
    result.reserve(total_size);
    for (const auto& row : A) {
        result.insert(result.end(), row.begin(), row.end());
    }
    return result;
}

}  // end namespace helpers