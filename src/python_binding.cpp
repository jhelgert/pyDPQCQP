#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>

#include "Helpers.hpp"
#include "QCQP.hpp"

using json = nlohmann::json;
using ms = std::chrono::milliseconds;

namespace py = pybind11;

template <typename T>
inline py::array_t<T> toPyArray(std::vector<T>&& passthrough) {
    // Pass result back to python
    auto* transferToHeapGetRawPtr = new std::vector<T>(std::move(passthrough));

    const py::capsule freeWhenDone(transferToHeapGetRawPtr, [](void* toFree) {
        delete static_cast<std::vector<T>*>(toFree);
    });

    auto passthroughNumpy =
        py::array_t<T>({transferToHeapGetRawPtr->size()}, // shape
                       {sizeof(T)},                       // strides
                       transferToHeapGetRawPtr->data(),   // ptr
                       freeWhenDone);
    return passthroughNumpy;
}

py::tuple solveDPQCQP(py::array_t<double> _Q0, py::array_t<double> _c0,
                      double r0, py::array_t<double> _Qs,
                      py::array_t<double> _cs, py::array_t<double> _rs,
                      py::array_t<double> _A1, py::array_t<double> _b1,
                      py::array_t<double> _A2, py::array_t<double> _b2,
                      py::array_t<double> _alpha_vals,
                      py::array_t<double> fixed_vals,
                      bool use_nontrivial_alphas = false) {
    // Dimensions
    size_t n = _c0.size();
    size_t p = _Qs.size();
    size_t m1 = _b1.size();
    size_t m2 = _b2.size();

    // Create stl vectors out of the numpy arrays

    // objective
    std::vector<double> Q0(_Q0.data(), _Q0.data() + _Q0.size());
    std::vector<double> c0(_c0.data(), _c0.data() + _c0.size());

    // quadratic constraints
    std::vector<std::vector<double>> Qs;
    std::vector<std::vector<double>> cs;
    for (py::ssize_t k = 0; k < _Qs.shape(0); ++k) {
        Qs.push_back(std::vector<double>(_Qs.data() + k * n * n,
                                         _Qs.data() + (k + 1) * n * n));
        cs.push_back(
            std::vector<double>(_cs.data() + k * n, _cs.data() + (k + 1) * n));
    }
    std::vector<double> rs(_rs.data(), _rs.data() + _rs.size());

    // affin linear inequality constraints
    std::vector<double> A1(_A1.data(), _A1.data() + _A1.size());
    std::vector<double> b1(_b1.data(), _b1.data() + _b1.size());

    // Our choices of fixed lagrangian multipliers alpha
    std::vector<double> alpha_vals(_alpha_vals.data(),
                                   _alpha_vals.data() + _alpha_vals.size());

    // affin linear equality constraints
    // Allocate A2 with dimension (m2 + n x n) and initialize
    // the (m2 x n) top submatrix with A2 from the json file
    // We will later incorporate the active bounds into A2
    // and b2
    BMat A2_tmp = blaze::zero<double>(m2 + n, n);
    BVec b2_tmp = blaze::zero<double>(m2 + n);
    auto A2_tmp_view = blaze::submatrix(A2_tmp, 0UL, 0UL, m2, n);
    auto b2_tmp_view = blaze::subvector(b2_tmp, 0UL, m2);
    A2_tmp_view = BMat(m2, n, _A2.data());
    b2_tmp_view = BVec(m2, _b2.data());

    // Set the integer variables all to fixed_vals
    std::vector<size_t> active_indices(fixed_vals.size(), 0UL);
    std::iota(active_indices.begin(), active_indices.end(), 0UL);
    std::vector<double> active_bounds(fixed_vals.size(), 0.0);

    // Create the QCQP problem
    QCQP prob1(Q0, c0, r0, Qs, cs, rs, A1, b1, alpha_vals,
               use_nontrivial_alphas);

    // Start timing of rref
    auto ts_rref = std::chrono::steady_clock::now();

    // incorporate active bounds into A2 and transform A2 to
    // max. rank (i.e. call rref)
    auto [A2, b2] = apply_active_bounds(active_indices, active_bounds, A2_tmp,
                                        b2_tmp, n, m2);
    auto te_rref = std::chrono::steady_clock::now();
    auto t_rref = std::chrono::duration_cast<ms>(te_rref - ts_rref).count();

    // Find the best dual bound
    auto start = std::chrono::steady_clock::now();
    prob1.findBestDualBound(A2, b2);
    auto end = std::chrono::steady_clock::now();
    auto t_calc = std::chrono::duration_cast<ms>(end - start).count();

    // Get the best lagrangian multiplier mu and alpha
    auto best_mu = prob1.getBestMu();
    auto best_alpha = prob1.getBestAlpha();

    auto mu_vec =
        std::vector<double>(best_mu.data(), best_mu.data() + best_mu.size());

    auto alpha_vec = std::vector<double>(best_alpha.data(),
                                         best_alpha.data() + best_alpha.size());

    return py::make_tuple(prob1.getBestObjVal(), t_calc, t_rref,
                          toPyArray<double>(std::move(alpha_vec)),
                          toPyArray<double>(std::move(mu_vec)));
}

PYBIND11_MODULE(_dpqcqp_cpp_wrapper, m) {
    m.def("solveDPQCQP", &solveDPQCQP, "Solves the Dual Problem of a QCQP.");
}