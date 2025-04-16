#ifndef META_PREDICTOR_H
#define META_PREDICTOR_H

#include <cstdint>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <numeric>

#include "../../inc/address.h"
#include "modules.h"

#include "../bimodal/bimodal.h"
#include "../gshare/gshare.h"
#include "../hashed_perceptron/hashed_perceptron.h"
#include "../perceptron/perceptron.h"

// Epsilon-Greedy bandit that selects from available predictors.
class EpsilonGreedyBandit : public champsim::modules::branch_predictor {
public:
    EpsilonGreedyBandit(int num_arms, double epsilon = 0.05);
    int select_arm();
    void update(int arm, double reward);
    void set_epsilon(double new_epsilon);  // <-- New setter

private:
    int num_arms_;
    double epsilon_;
    std::vector<int> counts_;
    std::vector<double> values_;
};

class meta_predictor {
public:
    // New constructor with adjustable epsilon.
    meta_predictor(double epsilon = 0.05);
    meta_predictor(O3_CPU* cpu, double epsilon);
    meta_predictor(O3_CPU* cpu);
    
    // Predicts a branch outcome.
    bool predict_branch(champsim::address ip);

    // Updates the chosen predictor with the branch outcome.
    void last_branch_result(champsim::address ip,
                            champsim::address branch_target,
                            bool taken,
                            uint8_t branch_type);

private:
    // List of available predictors.
    std::vector<champsim::modules::branch_predictor*> arms_;

    // Epsilon value used to initialize bandit buckets.
    double epsilon_;

    // Map from IP bucket (ip.bits % 64) to its own bandit.
    std::unordered_map<size_t, EpsilonGreedyBandit> bandit_buckets_;

    // Remember the chosen predictor and its prediction for update.
    int last_chosen_arm_;
    bool last_prediction_;

    // New fields for epsilon decay.
    size_t branch_count_;
    double initial_epsilon_;
    double decay_rate_;
};

#endif // META_PREDICTOR_H