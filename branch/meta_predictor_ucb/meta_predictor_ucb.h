#ifndef META_PREDICTOR_UCB_H
#define META_PREDICTOR_UCB_H

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

// --- Helper class to manage UCB1 algorithm per bucket ---
class UCB1Bandit {
public:
    UCB1Bandit(int num_arms);

    int select_arm();
    void update(int arm, double reward);

private:
    int num_arms_;
    std::vector<int> counts_;            // Number of times each arm was pulled
    std::vector<double> values_;         // Average reward for each arm
    uint64_t total_pulls_;               // Total number of arm pulls
    
    double ucb_score(int arm) const;     // Calculate UCB score for an arm
};

class meta_predictor_ucb {
public:
    meta_predictor_ucb();
    meta_predictor_ucb(O3_CPU* cpu);

    bool predict_branch(champsim::address ip);
    void last_branch_result(champsim::address ip,
                            champsim::address branch_target,
                            bool taken,
                            uint8_t branch_type);

private:
    void maybe_expand_buckets(size_t bucket);

    std::vector<champsim::modules::branch_predictor*> arms_;
    std::unordered_map<size_t, UCB1Bandit> bandit_buckets_;

    int last_chosen_arm_;
    bool last_prediction_;

    size_t num_buckets_;
};

#endif // META_PREDICTOR_UCB_H