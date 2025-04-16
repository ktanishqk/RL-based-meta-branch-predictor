#ifndef META_PREDICTOR_H
#define META_PREDICTOR_H

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "../../inc/address.h"
#include "modules.h"

#include "../bimodal/bimodal.h"
#include "../gshare/gshare.h"
#include "../hashed_perceptron/hashed_perceptron.h"
#include "../perceptron/perceptron.h"

// Epsilon-Greedy bandit that selects from available predictors.
class EpsilonGreedyBandit : champsim::modules::branch_predictor {
public:
    EpsilonGreedyBandit(int num_arms, double epsilon = 0.1);
    int select_arm();
    void update(int arm, double reward);

private:
    int num_arms_;
    double epsilon_;
    std::vector<int> counts_;
    std::vector<double> values_;
};

class MetaPredictor {
public:
    MetaPredictor();
    ~MetaPredictor();

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

    // Epsilon-greedy bandit for predictor selection.
    EpsilonGreedyBandit bandit_;

    // Remember the chosen predictor and its prediction for update.
    int last_chosen_arm_;
    bool last_prediction_;
};

#endif // META_PREDICTOR_H