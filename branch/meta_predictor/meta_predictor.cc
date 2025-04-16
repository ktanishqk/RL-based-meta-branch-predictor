#include "meta_predictor.h"
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <numeric>

//---------------------------------------------------------------------------
// EpsilonGreedyBandit implementation

EpsilonGreedyBandit::EpsilonGreedyBandit(int num_arms, double epsilon)
    : branch_predictor(nullptr),
      num_arms_(num_arms),
      epsilon_(epsilon),
      counts_(num_arms, 0),
      values_(num_arms, 0.0)
{
}

int EpsilonGreedyBandit::select_arm() {
    int total_counts = std::accumulate(counts_.begin(), counts_.end(), 0);
    if (total_counts < num_arms_) return total_counts; // Force trying each arm once

    double best_value = -1e9;
    int best_arm = 0;
    for (int i = 0; i < num_arms_; ++i) {
        double avg = values_[i];
        double confidence = sqrt(2 * log(total_counts) / counts_[i]);
        double ucb = avg + confidence;
        if (ucb > best_value) {
            best_value = ucb;
            best_arm = i;
        }
    }
    return best_arm;
}

void EpsilonGreedyBandit::update(int arm, double reward) {
    assert(arm >= 0 && arm < num_arms_);
    counts_[arm] += 1;
    double n = counts_[arm];
    values_[arm] = ((n - 1) / n) * values_[arm] + (reward / n);
}

//---------------------------------------------------------------------------
// meta_predictor implementation

meta_predictor::meta_predictor(double epsilon)
    : epsilon_(epsilon),
      last_chosen_arm_(-1),
      last_prediction_(false)
{
    // Create one instance of each predictor.
    arms_.push_back(new perceptron(nullptr));
    arms_.push_back(new bimodal(nullptr));
    arms_.push_back(new gshare(nullptr));
    arms_.push_back(new hashed_perceptron(nullptr));

    // Pre-create 64 bandit buckets for IP bucket values 0...63.
    for (size_t i = 0; i < 64; ++i) {
        bandit_buckets_.emplace(i, EpsilonGreedyBandit(4, epsilon_));
    }
}

meta_predictor::meta_predictor(O3_CPU* cpu, double epsilon) : meta_predictor(epsilon) {
    // Optionally use the 'cpu' pointer if needed.
}

meta_predictor::meta_predictor(O3_CPU* cpu) : meta_predictor(cpu, 0.05) {
    // Uses a default epsilon of 0.05.
}

bool meta_predictor::predict_branch(champsim::address ip) {
    // Determine IP bucket using ip.bits % 64.
    size_t bucket = static_cast<size_t>(ip.bits) % 64;
    // Use the appropriate bandit for this IP bucket.
    last_chosen_arm_ = bandit_buckets_.at(bucket).select_arm();
    bool prediction = false;
    switch (last_chosen_arm_) {
    case 0:
        prediction = static_cast<perceptron*>(arms_[0])->predict_branch(ip);
        break;
    case 1:
        prediction = static_cast<bimodal*>(arms_[1])->predict_branch(ip);
        break;
    case 2:
        prediction = static_cast<gshare*>(arms_[2])->predict_branch(ip);
        break;
    case 3:
        prediction = static_cast<hashed_perceptron*>(arms_[3])->predict_branch(ip);
        break;
    default:
        prediction = false;
        break;
    }
    last_prediction_ = prediction;
    return prediction;
}

void meta_predictor::last_branch_result(champsim::address ip,
                                        champsim::address branch_target,
                                        bool taken,
                                        uint8_t branch_type) {
    // Dispatch the update to the chosen predictor.
    switch (last_chosen_arm_) {
    case 0:
        static_cast<perceptron*>(arms_[0])->last_branch_result(ip, branch_target, taken, branch_type);
        break;
    case 1:
        static_cast<bimodal*>(arms_[1])->last_branch_result(ip, branch_target, taken, branch_type);
        break;
    case 2:
        static_cast<gshare*>(arms_[2])->last_branch_result(ip, branch_target, taken, branch_type);
        break;
    case 3:
        static_cast<hashed_perceptron*>(arms_[3])->last_branch_result(ip, branch_target, taken, branch_type);
        break;
    default:
        break;
    }
    double reward = (last_prediction_ == taken) ? 1.0 : -0.5;
    // Determine IP bucket using ip.bits % 64.
    size_t bucket = static_cast<size_t>(ip.bits) % 64;
    bandit_buckets_.at(bucket).update(last_chosen_arm_, reward);
}