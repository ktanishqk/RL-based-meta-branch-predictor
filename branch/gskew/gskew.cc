#include "gskew.h"

#include <algorithm>

// Hash function for table 0 (simple XOR)
std::size_t gskew::hash_table0(champsim::address ip, const std::bitset<HISTORY_LENGTH>& history) const
{
  auto pc = ip.to<std::size_t>();
  std::size_t history_hash = 0;
  
  // Convert history to hash value
  for (std::size_t i = 0; i < HISTORY_LENGTH; i++) {
    if (history[i]) {
      history_hash ^= (1ULL << (i % 16));
    }
  }
  
  return (pc ^ history_hash) % TABLE_SIZE;
}

// Hash function for table 1 (different bit patterns)
std::size_t gskew::hash_table1(champsim::address ip, const std::bitset<HISTORY_LENGTH>& history) const
{
  auto pc = ip.to<std::size_t>();
  std::size_t history_hash = 0;
  
  // Different bit pattern from table 0
  for (std::size_t i = 0; i < HISTORY_LENGTH; i++) {
    if (history[i]) {
      history_hash ^= (1ULL << ((i + 1) % 16));
    }
  }
  
  return ((pc >> 1) ^ history_hash) % TABLE_SIZE;
}

// Hash function for table 2 (another different pattern)
std::size_t gskew::hash_table2(champsim::address ip, const std::bitset<HISTORY_LENGTH>& history) const
{
  auto pc = ip.to<std::size_t>();
  std::size_t history_hash = 0;
  
  // Yet another pattern
  for (std::size_t i = 0; i < HISTORY_LENGTH; i++) {
    if (history[i]) {
      history_hash ^= (1ULL << ((i + 2) % 16));
    }
  }
  
  return ((pc >> 2) ^ (pc << 1) ^ history_hash) % TABLE_SIZE;
}

// Hash function for meta predictor
std::size_t gskew::hash_meta(champsim::address ip, const std::bitset<HISTORY_LENGTH>& history) const
{
  auto pc = ip.to<std::size_t>();
  std::size_t history_hash = 0;
  
  // Use first 16 bits of history for meta predictor
  for (std::size_t i = 0; i < 16 && i < HISTORY_LENGTH; i++) {
    if (history[i]) {
      history_hash ^= (1ULL << i);
    }
  }
  
  return ((pc >> 3) ^ history_hash) % (TABLE_SIZE / 4);
}

// Compute prediction from table votes
bool gskew::compute_prediction(const std::array<bool, NUM_TABLES>& table_predictions, bool use_bias) const
{
  if (use_bias) {
    // Biased vote: table 0 has higher weight
    if (table_predictions[0] == table_predictions[1] || 
        table_predictions[0] == table_predictions[2]) {
      return table_predictions[0];
    } else {
      return table_predictions[1]; // Third vote
    }
  } else {
    // Majority vote
    int taken_votes = 0;
    for (auto pred : table_predictions) {
      if (pred) taken_votes++;
    }
    return taken_votes > (int)(NUM_TABLES / 2);
  }
}

bool gskew::predict_branch(champsim::address pc)
{
  // Calculate indices for each table
  std::array<std::size_t, NUM_TABLES> indices = {
    hash_table0(pc, spec_global_history),
    hash_table1(pc, spec_global_history),
    hash_table2(pc, spec_global_history)
  };
  
  // Get meta predictor index
  std::size_t meta_index = hash_meta(pc, spec_global_history);
  
  // Get predictions from each table
  std::array<bool, NUM_TABLES> table_predictions;
  for (std::size_t i = 0; i < NUM_TABLES; i++) {
    const auto& counter = tables[i][indices[i]];
    table_predictions[i] = counter.value() > (counter.maximum / 2);
  }
  
  // Determine if we should use biased voting (meta predictor)
  bool use_bias = meta_predictor[meta_index].value() > (meta_predictor[meta_index].maximum / 2);
  
  // Get final prediction
  bool prediction = compute_prediction(table_predictions, use_bias);
  
  // Save prediction info for update
  prediction_info info = {
    pc,
    prediction,
    indices,
    meta_index,
    table_predictions
  };
  
  prediction_buffer.push_back(info);
  if (prediction_buffer.size() > 100) {
    prediction_buffer.erase(prediction_buffer.begin());
  }
  
  // Update speculative global history
  spec_global_history <<= 1;
  spec_global_history.set(0, prediction);
  
  return prediction;
}

void gskew::last_branch_result(champsim::address pc, champsim::address branch_target, bool taken, uint8_t branch_type)
{
  // Find prediction info for this branch
  auto it = std::find_if(prediction_buffer.begin(), prediction_buffer.end(),
                         [pc](const prediction_info& info) { return info.ip == pc; });
  
  if (it == prediction_buffer.end())
    return; // Skip update if prediction info not found
  
  prediction_info info = *it;
  prediction_buffer.erase(it);
  
  // Update real global history
  global_history <<= 1;
  global_history.set(0, taken);
  
  // If prediction was wrong, correct the speculative history
  if (info.prediction != taken) {
    spec_global_history = global_history;
  }
  
  // Update table counters
  for (std::size_t i = 0; i < NUM_TABLES; i++) {
    tables[i][info.indices[i]] += taken ? 1 : -1;
  }
  
  // Update meta predictor - should we use biased voting?
  bool majority_prediction = compute_prediction(info.table_predictions, false);
  bool biased_prediction = compute_prediction(info.table_predictions, true);
  
  if (majority_prediction != biased_prediction) {
    // Only update meta predictor if there's a difference between methods
    if (biased_prediction == taken) {
      meta_predictor[info.meta_index] += 1; // Bias was better
    } else {
      meta_predictor[info.meta_index] -= 1; // Majority was better
    }
  }
}