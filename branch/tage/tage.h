#ifndef BRANCH_TAGE_H
#define BRANCH_TAGE_H

#include <array>
#include <bitset>
#include <vector>

#include "modules.h"
#include "msl/fwcounter.h"

class tage : public champsim::modules::branch_predictor
{
private:
  // Constants
  static constexpr std::size_t NUM_TABLES = 4;
  static constexpr std::size_t BASE_TABLE_SIZE = 4096;
  static constexpr std::size_t TAGGED_TABLE_SIZE = 1024;
  static constexpr std::size_t TAG_SIZE = 8;
  static constexpr std::size_t COUNTER_BITS = 3;
  static constexpr std::size_t USEFULNESS_BITS = 2;
  static constexpr std::size_t MAX_HISTORY_LENGTH = 32;
  
  using bits = champsim::data::bits;
  
  // History lengths for the tagged tables (geometric progression)
  constexpr static std::array<std::size_t, NUM_TABLES> history_lengths = {
    4, 8, 16, 32
  };
  
  // Base table (bimodal predictor)
  std::array<champsim::msl::fwcounter<COUNTER_BITS>, BASE_TABLE_SIZE> base_table;
  
  // Entry in a tagged table
  struct tagged_entry {
    champsim::msl::fwcounter<COUNTER_BITS> counter;
    std::bitset<TAG_SIZE> tag;
    champsim::msl::fwcounter<USEFULNESS_BITS> u;
    bool valid = false;
  };
  
  // Tagged tables
  std::array<std::array<tagged_entry, TAGGED_TABLE_SIZE>, NUM_TABLES> tagged_tables;
  
  // Global branch history register
  std::bitset<MAX_HISTORY_LENGTH> global_history;
  
  // Last prediction information for update
  struct prediction_info {
    std::size_t provider_index;
    std::size_t provider_way;
    std::size_t alt_index;
    std::size_t alt_way;
    bool provider_hit;
    bool alt_hit;
    bool prediction;
  };
  
  prediction_info last_prediction;
  
  // Random seed for allocating new entries
  unsigned int seed = 0;
  
  // Helper functions
  std::size_t get_base_index(champsim::address ip) const;
  std::size_t get_table_index(champsim::address ip, std::size_t table_idx) const;
  std::bitset<TAG_SIZE> compute_tag(champsim::address ip, std::size_t table_idx) const;
  
public:
  tage();
  
  using branch_predictor::branch_predictor;
  
  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type);
};

#endif