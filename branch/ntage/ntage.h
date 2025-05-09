#ifndef BRANCH_NTAGE_H
#define BRANCH_NTAGE_H

#include <array>
#include <vector>
#include <bitset>
#include <cmath>

#include "modules.h"
#include "msl/fwcounter.h"
#include "msl/bits.h"
#include "../hashed_perceptron/folded_shift_register.h"

class ntage : public champsim::modules::branch_predictor
{
  using bits = champsim::data::bits;
  
  ///////////////////////////////////////////////
  // Neural Component
  ///////////////////////////////////////////////
  static constexpr std::size_t NEURON_SIZE = 1024;
  static constexpr std::size_t WEIGHT_BITS = 7;
  static constexpr std::size_t NUM_WEIGHTS = 16;
  static constexpr std::size_t LOCAL_HIST_LEN = 13;
  
  // Neural predictor weights
  using weight_type = champsim::msl::sfwcounter<WEIGHT_BITS>;
  struct neuron {
    weight_type bias{0};
    std::array<weight_type, NUM_WEIGHTS> weights{};
  };
  
  std::array<neuron, NEURON_SIZE> neurons{};
  
  // Local history table
  static constexpr std::size_t LOCAL_HIST_SIZE = 256;
  std::array<std::bitset<LOCAL_HIST_LEN>, LOCAL_HIST_SIZE> local_histories{};
  
  ///////////////////////////////////////////////
  // TAGE Component
  ///////////////////////////////////////////////
  static constexpr std::size_t NUM_TABLES = 4;
  static constexpr std::size_t BIMODAL_SIZE = 1 << 12;
  static constexpr std::size_t TABLE_SIZE = 1 << 10;
  static constexpr std::size_t TAG_WIDTH = 8;
  static constexpr std::size_t COUNTER_BITS = 3;
  static constexpr std::size_t USE_BITS = 2;
  static constexpr bits MIN_HISTORY{2};
  static constexpr bits MAX_HISTORY{250};
  
  // Bimodal table
  std::array<champsim::msl::fwcounter<COUNTER_BITS>, BIMODAL_SIZE> bimodal_table{};
  
  // Tagged tables components 
  struct table_entry {
    champsim::msl::fwcounter<COUNTER_BITS> counter{};
    champsim::msl::fwcounter<USE_BITS> useful{};
    uint16_t tag = 0;
  };
  
  // Tagged tables
  std::array<std::array<table_entry, TABLE_SIZE>, NUM_TABLES> tagged_tables{};
  
  // History lengths for each table (geometric progression)
  std::array<bits, NUM_TABLES> history_lengths{};
  
  // History registers for indexing and tags
  using history_type = folded_shift_register<bits{TAG_WIDTH}>;
  std::array<history_type, NUM_TABLES> history_registers{};
  
  // Global branch history
  std::vector<bool> global_history;
  
  // Path history (branch addresses)
  std::vector<uint16_t> path_history;
  
  // Random number for reset
  unsigned int seed = 0;
  
  ///////////////////////////////////////////////
  // Combined state
  ///////////////////////////////////////////////
  struct prediction_state {
    // TAGE component
    bool provider_hit = false;
    std::size_t provider_index = 0;
    std::size_t provider_entry = 0;
    bool alt_prediction = false;
    bool alt_hit = false;
    std::size_t alt_index = 0;
    std::size_t alt_entry = 0;
    std::array<std::size_t, NUM_TABLES> indices;
    std::array<uint16_t, NUM_TABLES> tags;
    
    // Neural component
    std::size_t neuron_index = 0;
    int output = 0;
    std::size_t local_history_index = 0;
    std::bitset<LOCAL_HIST_LEN> local_history;
  };
  
  prediction_state last_prediction;
  
  ///////////////////////////////////////////////
  // Utility functions
  ///////////////////////////////////////////////
  uint16_t compute_tag(champsim::address pc, std::size_t table_idx) const;
  std::size_t compute_index(champsim::address pc, std::size_t table_idx) const;
  void update_histories(bool taken, uint16_t branch_pc);
  void periodic_reset();
  
  // Neural functions
  std::size_t neuron_hash(champsim::address pc) const;
  std::size_t local_history_hash(champsim::address pc) const;
  int compute_output(std::size_t neuron_idx, const std::bitset<LOCAL_HIST_LEN>& history) const;
  void train_neuron(std::size_t neuron_idx, const std::bitset<LOCAL_HIST_LEN>& history, bool taken);
  void update_local_history(std::size_t index, bool taken);

public:
  explicit ntage(O3_CPU* cpu);
  bool predict_branch(champsim::address pc);
  void last_branch_result(champsim::address pc, champsim::address branch_target, bool taken, uint8_t branch_type);
};

#endif