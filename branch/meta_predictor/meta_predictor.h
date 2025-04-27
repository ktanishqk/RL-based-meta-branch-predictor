// meta_predictor.h (updated with dynamic bucket grow)

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
    EpsilonGreedyBandit(int num_arms, double initial_epsilon = 0.05, double decay_rate = 0.0001);

    int select_arm();
    void update(int arm, double reward);
    void step();

private:
    int num_arms_;
    double initial_epsilon_;
    double decay_rate_;
    double epsilon_;
    size_t total_updates_;

    std::vector<int> counts_;
    std::vector<double> values_;
};

class meta_predictor {
public:
    meta_predictor(double initial_epsilon = 0.05, double decay_rate = 0.0001);
    meta_predictor(O3_CPU* cpu, double initial_epsilon = 0.05, double decay_rate = 0.0001);

    bool predict_branch(champsim::address ip);
    void last_branch_result(champsim::address ip,
                            champsim::address branch_target,
                            bool taken,
                            uint8_t branch_type);

private:
    void maybe_expand_buckets(size_t bucket);

    std::vector<champsim::modules::branch_predictor*> arms_;
    std::unordered_map<size_t, EpsilonGreedyBandit> bandit_buckets_;

    size_t last_chosen_arm_;
    bool last_prediction_;

    size_t num_buckets_;
    double initial_epsilon_;
    double decay_rate_;
};

#endif // META_PREDICTOR_H
