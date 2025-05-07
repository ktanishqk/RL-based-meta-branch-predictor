#include "meta_predictor.h"
#include <algorithm>
#include <random>
#include <iostream> 
#include <cassert>
#include <cmath>
#include <numeric>

// --- EpsilonGreedyBandit Implementation ---

EpsilonGreedyBandit::EpsilonGreedyBandit(int num_arms, double initial_epsilon, double decay_rate)
    : num_arms_(num_arms),
      initial_epsilon_(initial_epsilon),
      decay_rate_(decay_rate),
      epsilon_(initial_epsilon),
      total_updates_(0),
      counts_(num_arms, 0),
      values_(num_arms, 0.0) {}

int EpsilonGreedyBandit::select_arm() {
    for (int i = 0; i < num_arms_; ++i) {
        if (counts_[i] == 0)
            return i;
    }
    double r = static_cast<double>(rand()) / RAND_MAX;
    if (r < epsilon_) {
        return rand() % num_arms_;
    }
    double best_value = -1e9;
    int best_arm = 0;
    for (int i = 0; i < num_arms_; ++i) {
        if (values_[i] > best_value) {
            best_value = values_[i];
            best_arm = i;
        }
    }
    return best_arm;
}

void EpsilonGreedyBandit::update(int arm, double reward) {
    counts_[arm] += 1;
    total_updates_++;
    double n = counts_[arm];
    values_[arm] = ((n - 1) / n) * values_[arm] + (reward / n);
}

void EpsilonGreedyBandit::step() {
    epsilon_ = initial_epsilon_ * exp(-decay_rate_ * static_cast<double>(total_updates_));
}

// --- meta_predictor Implementation ---

meta_predictor::meta_predictor(double initial_epsilon, double decay_rate)
    : last_chosen_arm_(-1),
      last_prediction_(false),
      initial_epsilon_(initial_epsilon),
      decay_rate_(decay_rate)
{
    arms_.push_back(new perceptron(nullptr));
    arms_.push_back(new bimodal(nullptr));
    arms_.push_back(new gshare(nullptr));
    arms_.push_back(new hashed_perceptron(nullptr));
}

meta_predictor::meta_predictor(O3_CPU* cpu, double initial_epsilon, double decay_rate)
    : meta_predictor(initial_epsilon, decay_rate) {}

bool meta_predictor::predict_branch(champsim::address ip) {
    size_t bucket = static_cast<uint64_t>(ip.bits);

    if (bandit_buckets_.find(bucket) == bandit_buckets_.end()) {
        bandit_buckets_.emplace(bucket, EpsilonGreedyBandit(4, initial_epsilon_, decay_rate_));
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

void meta_predictor::last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type) {
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
        bandit_buckets_.emplace(bucket, EpsilonGreedyBandit(4, initial_epsilon_, decay_rate_));
    }

    double reward = (last_prediction_ == taken) ? 1.0 : -0.5;
    bandit_buckets_.at(bucket).update(last_chosen_arm_, reward);
    bandit_buckets_.at(bucket).step();
}
