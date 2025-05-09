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

    // Initialize UCB bandits for each bucket
    for (size_t i = 0; i < num_buckets_; ++i) {
        bandit_buckets_.emplace(i, UCB1Bandit(4)); // 4 branch predictor arms
    }
}

meta_predictor_ucb::meta_predictor_ucb(O3_CPU* cpu)
    : meta_predictor_ucb() {}

bool meta_predictor_ucb::predict_branch(champsim::address ip) {
    size_t raw_bucket = static_cast<uint64_t>(ip.bits) % num_buckets_;
    maybe_expand_buckets(raw_bucket);

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
    case 4:
        prediction = static_cast<tage*>(arms_[4])->predict_branch(ip);
        break;
    case 5:
        prediction = static_cast<loop*>(arms_[5])->predict_branch(ip);
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
    case 4:
        static_cast<tage*>(arms_[4])->last_branch_result(ip, branch_target, taken, branch_type);
        break;
    case 5:
        static_cast<loop*>(arms_[5])->last_branch_result(ip, branch_target, taken, branch_type);
        break;
    default:
        break;
    }

    size_t raw_bucket = static_cast<uint64_t>(ip.bits) % num_buckets_;
    maybe_expand_buckets(raw_bucket);

    // Update UCB bandit with reward (1.0 for correct prediction, -0.5 for wrong prediction)
    double reward = (last_prediction_ == taken) ? 1.0 : -0.5;
    bandit_buckets_.at(raw_bucket).update(last_chosen_arm_, reward);
}

void meta_predictor_ucb::maybe_expand_buckets(size_t bucket) {
    if (bucket >= num_buckets_) {
        size_t new_size = std::max(num_buckets_ * 2, bucket + 1);
        for (size_t i = num_buckets_; i < new_size; ++i) {
            bandit_buckets_.emplace(i, UCB1Bandit(4));
        }
        num_buckets_ = new_size;
    }
}