#include "QCQP.hpp"

#include <blaze/Math.h>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Helpers.hpp"

using BVec = blaze::DynamicVector<double>;
using BMat = blaze::DynamicMatrix<double>;

void printMatrix(const BMat& A) {
    puts("--------------------------------------------------------");
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.columns(); ++j) {
            printf("%9.2lf ", A(i, j));
        }
        std::cout << "\n";
    }
    puts("--------------------------------------------------------");
}

QCQP::QCQP(std::string json_filename, std::vector<double>& _alphas_tmp,
           bool use_nontrivial_alphas)
    : alphas_tmp(_alphas_tmp) {
    // Read the json file
    std::ifstream i(json_filename);
    json j;
    i >> j;

    // std::cout << "hi1\n";

    // Parse and initialize the Vectors and Matrices
    std::vector<std::vector<double>> Q0_tmp = j["Q0"];
    std::vector<std::vector<double>> A1_tmp = j["A1"];
    // std::vector<std::vector<double>> A2_tmp2 = j["A2"];
    std::vector<double> c0_tmp = j["c0"];
    std::vector<double> b1_tmp = j["b1"];
    // std::vector<double> b2_tmp2 = j["b2"];
    std::vector<double> lb_tmp = j["lb"];
    std::vector<double> ub_tmp = j["ub"];
    // double r0 = j["r0"];
    std::vector<size_t> B_tmp = j["B"];
    std::vector<size_t> J_tmp = j["J"];

    p = j["p"];
    n = j["n"];
    m1 = j["m1"];
    m2 = j["m2"];
    l = B_tmp.size();
    r = J_tmp.size();

    // Initialize the blaze matrices and vectors
    // Hier evtl. std::move verwenden?

    Q0 = BMat(n, n, helpers::flatten(Q0_tmp).data());
    A1 = BMat(m1, n, helpers::flatten(A1_tmp).data());
    A1T = blaze::trans(A1);
    c0 = BVec(n, c0_tmp.data());
    b1 = BVec(m1, b1_tmp.data());
    rs = blaze::zero<double>(p);

    // Primal quadr. ineq. constraints
    for (size_t i = 0; i < p; ++i) {
        std::vector<std::vector<double>> Qi_tmp =
            j[std::string("Q") + std::to_string(i + 1)];
        std::vector<double> ci_tmp =
            j[std::string("c") + std::to_string(i + 1)];
        Qs.push_back(BMat(n, n, helpers::flatten(Qi_tmp).data()));
        cs.push_back(BVec(n, ci_tmp.data()));
        rs[i] = j[std::string("r") + std::to_string(i + 1)];
    }

    // Lagrange Multipliers
    // lambda = 0
    // lambda = blaze::zero<double>(m1);
    for (const auto& val : alphas_tmp) {
        // Create Vector alpha = (val, val, val, ...., val)
        // and add it to alphas
        alphas.push_back(blaze::uniform(p, val));
    }

    if (use_nontrivial_alphas) {
        // add the non-uniform alpha vectors, i.e.
        // alpha = (alpha_val 0 ... 0)
        // alpha = (0 ... 0 alpha_val 0 .. 0)
        for (const auto& val : alphas_tmp) {
            for (size_t i = 0; i < p; ++i) {
                BVec tmp = blaze::zero<double>(p);
                tmp[i] = val;
                alphas.push_back(tmp);
            }
        }
    }

    // Precalculate Qalpha, Qalphainv, calpha, ralpha for the given
    // alphas.
    for (const auto& alpha : alphas) {
        auto ralpha = r0 + blaze::dot(alpha, rs);
        auto calpha = c0;
        auto Qalpha = Q0;
        // calpha = c0 + alpha_1 * c_1 + .... + alpha_p * c_p
        // Qalpha = Q0 + alpha_1 * Q_1 + .... + alpha_p * Q_p
        for (size_t i = 0; i < p; ++i) {
            calpha += alpha[i] * cs[i];
            Qalpha += alpha[i] * Qs[i];
        }
        // Calculate the inverse Qalpha via Cholesky decomposition
        auto Qalphainv = Qalpha;
        blaze::invert<blaze::byLLH>(Qalphainv);

        // Add to Qalphas, Qalphainvs, calphas, ralphas
        // Qalphas.push_back(Qalpha);
        Qalphainvs.push_back(Qalphainv);
        calphas.push_back(calpha);
        ralphas.push_back(ralpha);
    }
}

QCQP::QCQP(std::vector<double>& _Q0, std::vector<double>& _c0, double _r0,
           std::vector<std::vector<double>>& _Qs,
           std::vector<std::vector<double>>& _cs, std::vector<double>& _rs,
           std::vector<double>& _A1, std::vector<double>& _b1,
           std::vector<double>& _alphas_tmp, bool use_nontrivial_alphas)
    : n(_c0.size()), p(_Qs.size()), m1(_b1.size()), Q0(BMat(n, n, _Q0.data())),
      c0(BVec(n, _c0.data())), r0(_r0) {
    // Quadratic constraints
    for (size_t i = 0; i < p; ++i) {
        Qs.push_back(BMat(n, n, _Qs[i].data()));
        cs.push_back(BVec(n, _cs[i].data()));
    }
    rs = BVec(p, _rs.data());

    A1 = BMat(m1, n, _A1.data());
    A1T = blaze::trans(A1);
    b1 = BVec(m1, _b1.data());

    // Lagrange Multipliers
    // lambda = 0
    // lambda = blaze::zero<double>(m1);
    for (const auto& val : _alphas_tmp) {
        // Create Vector alpha = (val, val, val, ...., val)
        // and add it to alphas
        alphas.push_back(blaze::uniform(p, val));
    }

    if (use_nontrivial_alphas) {
        // add the non-uniform alpha vectors, i.e.
        // alpha = (alpha_val 0 ... 0)
        // alpha = (0 ... 0 alpha_val 0 .. 0)
        for (const auto& val : _alphas_tmp) {
            for (size_t i = 0; i < p; ++i) {
                BVec tmp = blaze::zero<double>(p);
                tmp[i] = val;
                alphas.push_back(tmp);
            }
        }
    }

    // Precalculate Qalpha, Qalphainv, calpha, ralpha for the given
    // alphas.
    for (const auto& alpha : alphas) {
        auto ralpha = r0 + blaze::dot(alpha, rs);
        auto calpha = c0;
        auto Qalpha = Q0;
        // calpha = c0 + alpha_1 * c_1 + .... + alpha_p * c_p
        // Qalpha = Q0 + alpha_1 * Q_1 + .... + alpha_p * Q_p
        for (size_t i = 0; i < p; ++i) {
            calpha += alpha[i] * cs[i];
            Qalpha += alpha[i] * Qs[i];
        }
        // Calculate the inverse Qalpha via Cholesky decomposition
        auto Qalphainv = Qalpha;
        blaze::invert<blaze::byLLH>(Qalphainv);

        // Add to Qalphas, Qalphainvs, calphas, ralphas
        // Qalphas.push_back(Qalpha);
        Qalphainvs.push_back(Qalphainv);
        calphas.push_back(calpha);
        ralphas.push_back(ralpha);
    }
}

// Free function: apply the active bounds

std::pair<BMat, BVec>
apply_active_bounds(const std::vector<size_t>& indices,
                    const std::vector<double>& active_bounds, BMat& A2_tmp,
                    BVec& b2_tmp, size_t n, size_t m2) {
    for (size_t k = 0; k < indices.size(); ++k) {
        A2_tmp(m2 + k, indices[k]) = 1.0;
        b2_tmp[m2 + k] = active_bounds[k];
    }

    BMat A2_tmp_T = blaze::trans(A2_tmp);
    auto jb = helpers::rref(A2_tmp_T);
    // Set new dimension of A2
    auto new_m2 = jb.size();
    // Extract the base of A2 (i.e. transform A2 to full rank)

    auto A2 = BMat(blaze::submatrix(A2_tmp, 0UL, 0UL, new_m2, n)); // copy
    // auto A2T = blaze::trans(A2);
    auto b2 = BVec(blaze::elements(b2_tmp, jb)); // copy
    // std::cout << "b2:\n" << b2;

    return std::make_pair(A2, b2);
}

// Calculates the Dual Objective
double QCQP::calcDualObjective(const BMat& Qalphainv, const BVec& calpha,
                               double ralpha, const BMat& A2, const BMat& A2T,
                               const BVec& b2) noexcept {
    // ----- Calculate the lagrangian multiplier mu -----
    // tmp1 = A2 * Qalpha^(-1) * calpha + b2
    BVec tmp1 = A2 * Qalphainv * calpha + b2;
    // tmp2 = A2 * Qalpha(-1) * A2^T
    BMat tmp2 = A2 * Qalphainv * A2T;
    // Calculate the inverse of tmp2 via Cholesky decomposition
    // and calculate mu
    // Invert via cholesky decomposition
    blaze::invert<blaze::byLLH>(tmp2);
    BVec mu = -1.0 * tmp2 * tmp1;
    mus.push_back(mu);
    // ----------------------------------------------------------
    // tmp = calpha + A1^T * lambda + A2^T * mu
    BVec tmp = calpha;
    // tmp += A1T * lambda; // lambda = 0
    tmp += A2T * mu;
    // -0.5 * tmp^T * Qalpha^(-1) * tmp + ralpha - b1^T * lambda - b2^T * mu
    // We want to prevent unncessary temporary objects
    // (Unfortunately, BLAZE doesn't provide a suited expression template
    // for our kind of calculation).
    double objVal = 0.0;
    objVal += -0.5 * blaze::trans(tmp) * Qalphainv * tmp;
    objVal += ralpha;
    // objVal -= blaze::dot(b1, lambda); // lambda = 0
    objVal -= blaze::dot(b2, mu);
    // std::cout << "objVal = " << objVal << "\n";
    return objVal;
}

void QCQP::findBestDualBound(const BMat& A2, const BVec& b2) noexcept {
    auto A2T = blaze::trans(A2);
    // printf("Before loop inside findBestDualBound()\n");
    size_t i_best = 0;
    for (size_t i = 0; i < alphas.size(); ++i) {
        // printf("inside loop findBestDualbOund()\n");
        double objVal = calcDualObjective(Qalphainvs[i], calphas[i], ralphas[i],
                                          A2, A2T, b2);
        // printf("objVal = %lf\n", objVal);
        if (objVal > bestObjVal) {
            bestObjVal = objVal;
            i_best = i;
        }
    }
    // Set the best mu
    best_mu = mus[i_best];
    // Set best alpha
    bestAlpha = alphas[i_best];
}

double QCQP::getBestObjVal() const noexcept { return bestObjVal; }

BVec QCQP::getBestAlpha() const noexcept { return bestAlpha; }

BVec QCQP::getBestMu() const noexcept { return best_mu; }
