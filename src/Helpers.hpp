#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <blaze/Blaze.h>

#include <vector>

using BVec = blaze::DynamicVector<double>;
using BMat = blaze::DynamicMatrix<double>;

namespace helpers {

// Helper function: Swaps the i-th and k-th row
void swap_rows(BMat& A, size_t i, size_t k);

// Finds the largest absolute element in column j starting from row i, returns
// the row index.
size_t arg_max_remainder(const BMat& A, size_t i, size_t j);

// Fills column j starting from row i with zeros, i.e. A(i:n, j) = 0
void zero_out(BMat& A, size_t i, size_t j);

std::vector<size_t> rref(BMat& A, double tol = 1.0e-12);

std::vector<double> flatten(const std::vector<std::vector<double>>& A);

}  // end namespace helpers

#endif  // end HELPERS_HPP