#ifndef QCQP_HPP
#define QCQP_HPP

#include <blaze/Math.h>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Helpers.hpp"

// Aliases
using BVec    = blaze::DynamicVector<double>;
using BVecIdx = blaze::DynamicVector<size_t>;
using BMat    = blaze::DynamicMatrix<double>;
using json    = nlohmann::json;

class QCQP {
   private:
    size_t n{0};   // number of primal variables (n = B.size() + J.size())
    size_t p{0};   // number of quadratic inequality constraints
    size_t m1{0};  // number of affin-linear inequality constraints
    size_t m2{0};  // number of affin-linear equality constraints
    BVecIdx B;     // set of contiguous variables indices
    BVecIdx J;     // set of integer variables indices
    size_t l{0};   // number of primal contigous variables
    size_t r{0};   // number of primal integer variables
    // -------------------------------------------------------------------------
    BMat Q0;  // primal objective function:
    BVec c0;  // f(x) = 0.5 * xT * Q0 * x + c0T * x + r0
    double r0;
    // -------------------------------------------------------------------------
    // primal quadr. ineq. constrs. g_i(x) = 0.5 x' * Qi * x + ci' * x + ri <= 0
    std::vector<BMat> Qs;  // vector of matrices Q1, ..., Qp
    std::vector<BVec> cs;  // vector of vectors c1, ..., cp
    BVec rs;               // vector of scalars r1, ..., rp
    // -------------------------------------------------------------------------
    // primal constraint h_1(x) = A1 * x - b1 <= 0
    BMat A1;
    BMat A1T;  // transpose of A1
    BVec b1;
    // -------------------------------------------------------------------------
    std::vector<BMat> Qalphainvs;  // Vector of different matrices Qalphainv
    // std::vector<BMat> Qalphas;     // Vector of different matrices Qalpha
    std::vector<BVec> calphas;  // Vector of different vectors calpha
    std::vector<double> ralphas;
    double bestObjVal{-1.79769e308};
    BVec bestAlpha;  //
    // -------------------------------------------------------------------------
    // dual variables (lagrangian multipliers)
    std::vector<double> alphas_tmp;
    std::vector<BVec> alphas;
    std::vector<BVec> mus;
    BVec best_mu;
    // -------------------------------------------------------------------------
   public:
    // No default constructor
    QCQP() = delete;

    //
    explicit QCQP(std::string json_filename, std::vector<double>& _alphas_tmp,
                  bool use_nontrivial_alphas = false);

    explicit QCQP(std::vector<double>& Q0, std::vector<double>& c0, double r0,
                  std::vector<std::vector<double>>& Qs,
                  std::vector<std::vector<double>>& cs, std::vector<double>& rs,
                  std::vector<double>& A1, std::vector<double>& b1,
                  std::vector<double>& _alphas_tmp, bool use_nontrivial_alphas = false);

    // Destructor
    ~QCQP() = default;

    // ..
    // void applyActiveBounds(const std::vector<size_t>& indices,
    //                        const std::vector<double>& active_bounds)
    //                        noexcept;

    // // Calculates ralpha, calpha, Qalpha for given alpha
    // void calcHelper(const BVec& alpha) noexcept;

    // Calculates the lagrangian multiplier mu according to the formula
    double calcDualObjective(const BMat& Qalphainv, const BVec& calpha,
                             double ralpha, const BMat& A2, const BMat& A2T,
                             const BVec& b2) noexcept;
    void findBestDualBound(const BMat& A2, const BVec& b2) noexcept;
    double getBestObjVal() const noexcept;
    BVec getBestAlpha() const noexcept;
    BVec getBestMu() const noexcept;
};

// Free function
std::pair<BMat, BVec> apply_active_bounds(
    const std::vector<size_t>& indices,
    const std::vector<double>& active_bounds, BMat& A2_tmp, BVec& b2_tmp,
    size_t n, size_t m2);

#endif  // End QCQP_HPP