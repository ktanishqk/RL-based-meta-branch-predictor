#include <cassert>
#include <cmath>
#include <cstdlib>

#include "meta_predictor.h"
#include "perceptron.h"
#include "bimodal.h"
#include "gshare.h"
#include "hashed_perceptron.h"

// EpsilonGreedyBandit implementation

EpsilonGreedyBandit::EpsilonGreedyBandit(int num_arms, double epsilon)
    : num_arms_(num_arms),
      epsilon_(epsilon),
      counts_(num_arms, 0),
      values_(num_arms, 0.0) {}

int EpsilonGreedyBandit::select_arm() {
    double prob = static_cast<double>(rand()) / RAND_MAX;
    if (prob < epsilon_) {
        // Choose a random arm.
        return rand() % num_arms_;
    } else {
        // Choose the arm with the highest average reward.
        int best_arm = 0;
        double best_value = values_[0];
        for (int i = 1; i < num_arms_; ++i) {
            if (values_[i] > best_value) {
                best_value = values_[i];
                best_arm = i;
            }
        }
        return best_arm;
    }
}

void EpsilonGreedyBandit::update(int arm, double reward) {
    assert(arm >= 0 && arm < num_arms_);
    counts_[arm] += 1;
    double n = counts_[arm];
    values_[arm] = ((n - 1) / n) * values_[arm] + (reward / n);
}

// MetaPredictor implementation

MetaPredictor::MetaPredictor() 
    : bandit_(4, 0.1),
      last_chosen_arm_(-1),
      last_prediction_(false) {
    // Create one instance of each predictor.
    // For simplicity we cast the pointers to void*. 
    arms_.push_back(static_cast<void*>(new perceptron(nullptr)));
    arms_.push_back(static_cast<void*>(new bimodal(nullptr)));
    arms_.push_back(static_cast<void*>(new gshare(nullptr)));
    arms_.push_back(static_cast<void*>(new hashed_perceptron(nullptr)));
}

MetaPredictor::~MetaPredictor() {
    // Free the allocated predictor instances.
    if (!arms_.empty()) {
        delete static_cast<perceptron*>(arms_[0]);
        delete static_cast<bimodal*>(arms_[1]);
        delete static_cast<gshare*>(arms_[2]);
        delete static_cast<hashed_perceptron*>(arms_[3]);
    }
}

bool MetaPredictor::predict_branch(champsim::address ip) {
    // Use epsilon-greedy bandit to select an arm.
    last_chosen_arm_ = bandit_.select_arm();

    // Call the corresponding predictor's predict_branch method.
    bool prediction = false;
    switch (last_chosen_arm_) {
    case 0:
        prediction = (static_cast<perceptron*>(arms_[0]))->predict_branch(ip);
        break;
    case 1:
        prediction = (static_cast<bimodal*>(arms_[1]))->predict_branch(ip);
        break;
    case 2:
        prediction = (static_cast<gshare*>(arms_[2]))->predict_branch(ip);
        break;
    case 3:
        prediction = (static_cast<hashed_perceptron*>(arms_[3]))->predict_branch(ip);
        break;
    default:
        prediction = false;
    }
    last_prediction_ = prediction;
    return prediction;
}

void MetaPredictor::last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type) {
    // Forward the branch result to the chosen predictor.
    switch (last_chosen_arm_) {
    case 0:
        (static_cast<perceptron*>(arms_[0]))->last_branch_result(ip, branch_target, taken, branch_type);
        break;
    case 1:
        (static_cast<bimodal*>(arms_[1]))->last_branch_result(ip, branch_target, taken, branch_type);
        break;
    case 2:
        (static_cast<gshare*>(arms_[2]))->last_branch_result(ip, branch_target, taken, branch_type);
        break;
    case 3:
        (static_cast<hashed_perceptron*>(arms_[3]))->last_branch_result(ip, branch_target, taken, branch_type);
        break;
    default:
        break;
    }
    
    // Compute reward: 1.0 if prediction was correct, 0.0 otherwise,
    // and update the bandit.
    double reward = (last_prediction_ == taken) ? 1.0 : 0.0;
    bandit_.update(last_chosen_arm_, reward);
}