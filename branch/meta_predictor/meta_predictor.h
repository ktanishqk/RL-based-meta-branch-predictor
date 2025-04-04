#ifndef META_PREDICTOR_H
#define META_PREDICTOR_H

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "../../inc/address.h" // ChampSim address type
#include "modules.h" // (if needed by your predictors)

// Include the four branch predictors
#include "branch/bimodal/bimodal.h"
#include "branch/gshare/gshare.h"
#include "branch/hashed_perceptron/hashed_perceptron.h"
#include "branch/perceptron/perceptron.h"

// Epsilon-Greedy bandit to select from available predictors.
class EpsilonGreedyBandit
{
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

class MetaPredictor // removed inheritance from champsim::modules::branch_predictor
{
public:
  MetaPredictor();

  // Functions to predict branch and update after outcome.
  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type);

private:
  // List of available predictors.
  std::vector<void*> arms_; // we'll cast to appropriate predictor type when used
  EpsilonGreedyBandit bandit_;

  // Save the arm chosen (and its prediction) to update after branch outcome.
  int last_chosen_arm_;
  bool last_prediction_;
};

#endif