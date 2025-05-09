#ifndef BRANCH_GSKEW_H
#define BRANCH_GSKEW_H

#include <array>
#include <vector>
#include <bitset>

#include "modules.h"
#include "msl/fwcounter.h"

class gskew : public champsim::modules::branch_predictor
{
  // GSKEW parameters
  static constexpr std::size_t NUM_TABLES = 3;          // Number of tables
  static constexpr std::size_t TABLE_SIZE = 1 << 14;    // 16K entries per table 
  static constexpr std::size_t COUNTER_BITS = 2;        // Prediction counter bits
  static constexpr std::size_t HISTORY_LENGTH = 27;     // Global history length

  // Tables of prediction counters
  std::array<std::array<champsim::msl::fwcounter<COUNTER_BITS>, TABLE_SIZE>, NUM_TABLES> tables;
  
  // Meta predictor for selecting between majority vote and biased vote
  std::array<champsim::msl::fwcounter<COUNTER_BITS>, TABLE_SIZE / 4> meta_predictor;
  
  // Global history register
  std::bitset<HISTORY_LENGTH> global_history;
  
  // Speculative global history register (for predict)
  std::bitset<HISTORY_LENGTH> spec_global_history;
  
  // Structure to store prediction info
  struct prediction_info {
    champsim::address ip;
    bool prediction;
    std::array<std::size_t, NUM_TABLES> indices;
    std::size_t meta_index;
    std::array<bool, NUM_TABLES> table_predictions;
  };
  
  // Buffer for update
  std::vector<prediction_info> prediction_buffer;
  
  // Hash functions for indexing tables
  std::size_t hash_table0(champsim::address ip, const std::bitset<HISTORY_LENGTH>& history) const;
  std::size_t hash_table1(champsim::address ip, const std::bitset<HISTORY_LENGTH>& history) const;
  std::size_t hash_table2(champsim::address ip, const std::bitset<HISTORY_LENGTH>& history) const;
  std::size_t hash_meta(champsim::address ip, const std::bitset<HISTORY_LENGTH>& history) const;
  
  // Compute majority or biased vote
  bool compute_prediction(const std::array<bool, NUM_TABLES>& table_predictions, bool use_bias) const;

public:
  using branch_predictor::branch_predictor;
  
  bool predict_branch(champsim::address pc);
  void last_branch_result(champsim::address pc, champsim::address branch_target, bool taken, uint8_t branch_type);
};

#endif