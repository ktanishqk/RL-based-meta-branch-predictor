#ifndef BRANCH_LOOP_H
#define BRANCH_LOOP_H

#include <array>
#include <bitset>

#include "modules.h"
#include "msl/fwcounter.h"
#include "../bimodal/bimodal.h" // Used as a fallback predictor

class loop : public champsim::modules::branch_predictor
{
private:
  // Constants
  static constexpr std::size_t LOOP_TABLE_SIZE = 256;
  static constexpr std::size_t TAG_BITS = 12;
  static constexpr std::size_t CONFIDENCE_COUNTER_BITS = 2;
  static constexpr std::size_t ITER_COUNTER_BITS = 10;
  static constexpr std::size_t PRIME = 251;
  
  // Entry in the loop predictor table
  struct loop_entry {
    std::bitset<TAG_BITS> tag;
    champsim::msl::fwcounter<CONFIDENCE_COUNTER_BITS> confidence;
    uint16_t iter_count;
    uint16_t current_iter;
    bool dir;
    bool valid;
  };
  
  // Loop predictor table
  std::array<loop_entry, LOOP_TABLE_SIZE> loop_table;
  
  // Bimodal predictor as a fallback
  bimodal bimodal_predictor;
  
  // Last prediction information for update
  struct prediction_info {
    std::size_t index;
    bool loop_prediction;
    bool bimodal_prediction;
    bool used_loop;
  };
  
  prediction_info last_prediction;
  
  // Hash function for table indexing
  [[nodiscard]] static constexpr auto hash(champsim::address ip) { return ip.to<unsigned long>() % PRIME; }
  
  // Compute tag for a given PC
  [[nodiscard]] static constexpr auto compute_tag(champsim::address ip) {
    return std::bitset<TAG_BITS>((ip.to<unsigned long>() >> 8) & ((1 << TAG_BITS) - 1));
  }
  
public:
  loop();
  loop(O3_CPU* cpu);
  
  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type);
};

#endif