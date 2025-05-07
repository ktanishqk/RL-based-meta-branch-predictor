#include "meta_predictor_ucb.h"
#include <algorithm>
#include <cmath>
#include <iostream>

// --- UCB1Bandit Implementation ---

UCB1Bandit::UCB1Bandit(int num_arms)
    : num_arms_(num_arms),
      counts_(num_arms, 0),
      values_(num_arms, 0.0),
      total_pulls_(0) {}

double UCB1Bandit::ucb_score(int arm) const {
    if (counts_[arm] == 0)
        return std::numeric_limits<double>::max();
    double exploitation = values_[arm];
    double exploration = std::sqrt(2.0 * std::log(static_cast<double>(total_pulls_)) / counts_[arm]);
    return exploitation + exploration;
}

int UCB1Bandit::select_arm() {
    double best_score = -std::numeric_limits<double>::max();
    int best_arm = 0;
    for (int i = 0; i < num_arms_; ++i) {
        double score = ucb_score(i);
        if (score > best_score) {
            best_score = score;
            best_arm = i;
        }
    }
    return best_arm;
}

void UCB1Bandit::update(int arm, double reward) {
    counts_[arm]++;
    total_pulls_++;
    double n = static_cast<double>(counts_[arm]);
    values_[arm] = ((n - 1.0) / n) * values_[arm] + (reward / n);
}

// --- meta_predictor_ucb Implementation ---

meta_predictor_ucb::meta_predictor_ucb()
    : last_chosen_arm_(-1),
      last_prediction_(false)
{
    arms_.push_back(new perceptron(nullptr));
    arms_.push_back(new bimodal(nullptr));
    arms_.push_back(new gshare(nullptr));
    arms_.push_back(new hashed_perceptron(nullptr));
}

meta_predictor_ucb::meta_predictor_ucb(O3_CPU* cpu)
    : meta_predictor_ucb() {}

bool meta_predictor_ucb::predict_branch(champsim::address ip) {
    size_t bucket = static_cast<uint64_t>(ip.bits);

    if (bandit_buckets_.find(bucket) == bandit_buckets_.end()) {
        bandit_buckets_.emplace(bucket, UCB1Bandit(4));
    }

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

void meta_predictor_ucb::last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type) {
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

    size_t bucket = static_cast<uint64_t>(ip.bits);

    if (bandit_buckets_.find(bucket) == bandit_buckets_.end()) {
        bandit_buckets_.emplace(bucket, UCB1Bandit(4));
    }

    double reward = (last_prediction_ == taken) ? 1.0 : -0.5;
    bandit_buckets_.at(bucket).update(last_chosen_arm_, reward);
}
