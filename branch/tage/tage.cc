#include "tage.h"
#include <cstdlib>

std::size_t tage::get_base_index(champsim::address ip) const
{
  return ip.to<std::size_t>() % BASE_TABLE_SIZE;
}

std::size_t tage::get_table_index(champsim::address ip, std::size_t table_idx) const
{
  // Fold history to get a value within index range
  std::size_t history_val = 0;
  for (std::size_t i = 0; i < history_lengths[table_idx]; i++) {
    history_val = (history_val << 1) | global_history[i];
  }
  
  // Combine PC and history for the index
  return (ip.to<std::size_t>() ^ history_val) % TAGGED_TABLE_SIZE;
}

std::bitset<tage::TAG_SIZE> tage::compute_tag(champsim::address ip, std::size_t table_idx) const
{
  // Fold history in a different way for tag computation
  std::size_t history_val = 0;
  for (std::size_t i = 0; i < history_lengths[table_idx]; i += 2) {
    history_val = (history_val << 1) | global_history[i];
  }
  
  // Simple tag computation from PC and a different fold of history
  std::size_t tag_val = (ip.to<std::size_t>() >> 2) ^ (history_val << 1);
  return std::bitset<TAG_SIZE>(tag_val & ((1 << TAG_SIZE) - 1));
}

bool tage::predict_branch(champsim::address ip)
{
  // Reset last prediction info
  last_prediction = {NUM_TABLES, 0, NUM_TABLES, 0, false, false, false};
  
  // Check for tag matches in the tagged tables, from longest to shortest history
  for (int table_idx = NUM_TABLES - 1; table_idx >= 0; table_idx--) {
    auto index = get_table_index(ip, table_idx);
    auto tag = compute_tag(ip, table_idx);
    auto& entry = tagged_tables[table_idx][index];
    
    if (entry.valid && entry.tag == tag) {
      // Tag hit, use this table as provider
      if (last_prediction.provider_index == NUM_TABLES) {
        last_prediction.provider_index = table_idx;
        last_prediction.provider_way = index;
        last_prediction.provider_hit = true;
      } 
      // If we already have a provider, this becomes the alternate
      else if (last_prediction.alt_index == NUM_TABLES) {
        last_prediction.alt_index = table_idx;
        last_prediction.alt_way = index;
        last_prediction.alt_hit = true;
        break; // We have both provider and alternate now
      }
    }
  }
  
  bool prediction;
  
  // If no tag hit, use the base predictor
  if (last_prediction.provider_index == NUM_TABLES) {
    auto base_index = get_base_index(ip);
    prediction = base_table[base_index].value() > (base_table[base_index].maximum / 2);
    last_prediction.provider_index = NUM_TABLES; // Special value for base predictor
    last_prediction.provider_way = base_index;
  }
  // Tag hit, use the tagged predictor
  else {
    auto& entry = tagged_tables[last_prediction.provider_index][last_prediction.provider_way];
    prediction = entry.counter.value() > (entry.counter.maximum / 2);
    
    // If no alternate was found from tag hit, use the base predictor or shorter history table
    if (last_prediction.alt_index == NUM_TABLES) {
      last_prediction.alt_index = NUM_TABLES; // Use base predictor as alternate
      last_prediction.alt_way = get_base_index(ip);
    }
  }
  
  last_prediction.prediction = prediction;
  return prediction;
}

void tage::last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type)
{
  // Update the global history shift register
  global_history <<= 1;
  global_history.set(0, taken);
  
  // Update the prediction tables
  bool prediction_correct = (last_prediction.prediction == taken);
  
  // Update base predictor if it was used
  if (last_prediction.provider_index == NUM_TABLES) {
    base_table[last_prediction.provider_way] += taken ? 1 : -1;
  }
  // Update tagged predictor if it was used
  else {
    auto& provider_entry = tagged_tables[last_prediction.provider_index][last_prediction.provider_way];
    provider_entry.counter += taken ? 1 : -1;
    
    // Update usefulness counter on correct prediction
    if (prediction_correct) {
      // If alternate prediction would have been wrong, increment usefulness
      bool alt_prediction;
      if (last_prediction.alt_index == NUM_TABLES) {
        alt_prediction = base_table[last_prediction.alt_way].value() > (base_table[last_prediction.alt_way].maximum / 2);
      } else {
        alt_prediction = tagged_tables[last_prediction.alt_index][last_prediction.alt_way].counter.value() > 
                         (tagged_tables[last_prediction.alt_index][last_prediction.alt_way].counter.maximum / 2);
      }
      
      if (alt_prediction != taken) {
        provider_entry.u += 1;
      }
    }
  }
  
  // Allocate new entries if prediction was wrong
  if (!prediction_correct) {
    // Try to allocate a new entry in a table with longer history than provider
    for (std::size_t table_idx = 0; table_idx < NUM_TABLES; table_idx++) {
      // Only consider tables with longer histories than the provider
      if (last_prediction.provider_index == NUM_TABLES || table_idx > last_prediction.provider_index) {
        auto index = get_table_index(ip, table_idx);
        auto tag = compute_tag(ip, table_idx);
        auto& entry = tagged_tables[table_idx][index];
        
        // Allocate if entry is invalid or usefulness is 0
        if (!entry.valid || entry.u.value() == 0) {
          entry.valid = true;
          entry.tag = tag;
          entry.counter = champsim::msl::fwcounter<COUNTER_BITS>(taken ? (1 << (COUNTER_BITS-1)) : ((1 << (COUNTER_BITS-1)) - 1));
          entry.u = champsim::msl::fwcounter<USEFULNESS_BITS>(0);
          break; // Allocate only one entry per misprediction
        }
      }
    }
    
    // Decrease usefulness counters periodically to age out old entries
    if ((seed++ & 0xFF) == 0) {
      for (auto& table : tagged_tables) {
        for (auto& entry : table) {
          if (entry.valid && entry.u.value() > 0) {
            entry.u -= 1;
          }
        }
      }
    }
  }
}