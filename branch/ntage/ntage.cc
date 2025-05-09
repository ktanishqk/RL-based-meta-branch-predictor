#include "ntage.h"

#include <numeric>
#include <algorithm>

ntage::ntage(O3_CPU* cpu) 
    : branch_predictor(cpu),
      global_history(champsim::to_underlying(MAX_HISTORY), false),
      path_history(16, 0)
{
    // Initialize history lengths in geometric series
    for (std::size_t i = 0; i < NUM_TABLES; i++) {
        double ratio = std::pow(double(champsim::to_underlying(MAX_HISTORY)) / 
                               champsim::to_underlying(MIN_HISTORY),
                               1.0 / (NUM_TABLES - 1));
        history_lengths[i] = bits{static_cast<unsigned>(
            champsim::to_underlying(MIN_HISTORY) * std::pow(ratio, i))};
    }

    // Initialize history registers with appropriate lengths
    for (std::size_t i = 0; i < NUM_TABLES; i++) {
        history_registers[i] = history_type(history_lengths[i]);
    }
}

uint16_t ntage::compute_tag(champsim::address pc, std::size_t table_idx) const
{
    uint16_t tag = pc.slice_lower<champsim::data::bits{16}>().to<uint16_t>();
    tag ^= static_cast<uint16_t>(pc.slice_lower<champsim::data::bits{16}>().to<uint16_t>() << 1);
    tag ^= static_cast<uint16_t>(history_registers[table_idx].value() & ((1ULL << TAG_WIDTH) - 1));
    
    // Mix in path history for longer history tables
    if (table_idx > 1) {
        uint16_t path_hash = 0;
        for (std::size_t i = 0; i < std::min(path_history.size(), std::size_t(16)); i++) {
            path_hash ^= path_history[i];
        }
        tag ^= path_hash;
    }
    
    return tag & ((1 << TAG_WIDTH) - 1);
}

std::size_t ntage::compute_index(champsim::address pc, std::size_t table_idx) const
{
    std::size_t index = pc.slice_lower<champsim::data::bits{14}>().to<std::size_t>();
    index ^= history_registers[table_idx].value() & ((1 << 14) - 1);
    
    // Incorporate path history for tables with longer history
    if (table_idx > 0) {
        uint16_t path_hash = 0;
        for (std::size_t i = 0; i < std::min(path_history.size(), std::size_t(16)); i++) {
            uint16_t shift_amount = static_cast<uint16_t>(i % 5);
            uint16_t shifted_value = static_cast<uint16_t>(path_history[i] << shift_amount);
            path_hash ^= shifted_value;
        }
        index ^= path_hash;
    }
    
    return index % TABLE_SIZE;
}

// Hash function for neural predictor
std::size_t ntage::neuron_hash(champsim::address pc) const
{
    auto addr = pc.to<uint32_t>();
    return (addr ^ (addr >> 7)) % NEURON_SIZE;
}

// Hash function for local history table
std::size_t ntage::local_history_hash(champsim::address pc) const
{
    auto addr = pc.to<uint32_t>();
    return (addr ^ (addr >> 11)) % LOCAL_HIST_SIZE;
}

// Compute neuron output (dot product of weights and history)
int ntage::compute_output(std::size_t neuron_idx, const std::bitset<LOCAL_HIST_LEN>& history) const
{
    const auto& n = neurons[neuron_idx];
    
    // Start with bias
    int output = static_cast<int>(n.bias.value());
    
    // Add weighted history bits
    for (std::size_t i = 0; i < LOCAL_HIST_LEN && i < NUM_WEIGHTS; i++) {
        if (history[i]) {
            output += static_cast<int>(n.weights[i].value());
        } else {
            output -= static_cast<int>(n.weights[i].value());
        }
    }
    
    return output;
}

// Train neuron using perceptron learning rule
void ntage::train_neuron(std::size_t neuron_idx, const std::bitset<LOCAL_HIST_LEN>& history, bool taken)
{
    auto& n = neurons[neuron_idx];
    
    // Update bias
    n.bias += taken ? 1 : -1;
    
    // Update weights
    for (std::size_t i = 0; i < LOCAL_HIST_LEN && i < NUM_WEIGHTS; i++) {
        if (history[i] == taken) {
            n.weights[i] += 1; // Positive correlation
        } else {
            n.weights[i] -= 1; // Negative correlation
        }
    }
}

// Update local history register
void ntage::update_local_history(std::size_t index, bool taken)
{
    auto& history = local_histories[index];
    history <<= 1;
    history.set(0, taken);
}

void ntage::update_histories(bool taken, uint16_t branch_pc)
{
    // Update global history
    if (global_history.size() > 0) {
        global_history.pop_back();
        global_history.insert(global_history.begin(), taken);
    }
    
    // Update path history
    if (path_history.size() > 0) {
        path_history.pop_back();
        path_history.insert(path_history.begin(), branch_pc & 0x3F); // Keep lower 6 bits
    }
    
    // Update folded history registers
    for (auto& hist : history_registers) {
        hist.push_back(taken);
    }
}

void ntage::periodic_reset()
{
    // Periodically reset u bits with some probability
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    if ((seed & 0x3ffffff) == 0) {
        // Approximately every 2^26 branches, reset u bits
        for (auto& table : tagged_tables) {
            for (auto& entry : table) {
                entry.useful = 0;
            }
        }
    }
}

bool ntage::predict_branch(champsim::address pc)
{
    // Reset state for new prediction
    prediction_state state;
    
    // Get neural prediction
    std::size_t neuron_idx = neuron_hash(pc);
    std::size_t local_hist_idx = local_history_hash(pc);
    const auto& local_hist = local_histories[local_hist_idx];
    
    int output = compute_output(neuron_idx, local_hist);
    bool neural_prediction = output >= 0;
    
    // Save neural state
    state.neuron_index = neuron_idx;
    state.output = output;
    state.local_history_index = local_hist_idx;
    state.local_history = local_hist;
    
    // Now do TAGE prediction
    // Compute bimodal index
    std::size_t bimodal_index = pc.to<std::size_t>() % BIMODAL_SIZE;
    bool bimodal_prediction = bimodal_table[bimodal_index].value() >= (bimodal_table[bimodal_index].maximum / 2);
    
    // Provider and alternate predictions default to bimodal
    bool provider_prediction = bimodal_prediction;
    bool alt_prediction = bimodal_prediction;
    
    // Find provider component (longest matching history)
    state.provider_hit = false;
    state.alt_hit = false;
    
    // Compute indices and tags for all tables
    for (std::size_t i = 0; i < NUM_TABLES; i++) {
        state.indices[i] = compute_index(pc, i);
        state.tags[i] = compute_tag(pc, i);
    }
    
    // Look for tag matches, starting from the longest history
    for (int i = NUM_TABLES - 1; i >= 0; i--) {
        std::size_t index = state.indices[i];
        uint16_t tag = state.tags[i];
        
        if (tagged_tables[i][index].tag == tag) {
            if (!state.provider_hit) {
                // Found provider
                state.provider_hit = true;
                state.provider_index = i;
                state.provider_entry = index;
                provider_prediction = tagged_tables[i][index].counter.value() >= 
                                     (tagged_tables[i][index].counter.maximum / 2);
            } else if (!state.alt_hit) {
                // Found alternate
                state.alt_hit = true;
                state.alt_index = i;
                state.alt_entry = index;
                alt_prediction = tagged_tables[i][index].counter.value() >= 
                                (tagged_tables[i][index].counter.maximum / 2);
                break; // No need to look further
            }
        }
    }
    
    if (!state.provider_hit) {
        // Use bimodal as provider
        state.provider_index = NUM_TABLES; // Special value for bimodal
        state.provider_entry = bimodal_index;
    }
    
    if (!state.alt_hit) {
        // Use bimodal as alternate
        state.alt_index = NUM_TABLES; // Special value for bimodal
        state.alt_entry = bimodal_index;
    }
    
    // Store state for update
    state.alt_prediction = alt_prediction;
    last_prediction = state;
    
    // Final prediction: use provider prediction from TAGE, but for high-confidence
    // predictions from the neural predictor, use that instead
    if (std::abs(output) > 14) {
        // Neural predictor is highly confident, use it
        return neural_prediction;
    } else {
        // Otherwise use TAGE
        return provider_prediction;
    }
}

void ntage::last_branch_result(champsim::address pc, champsim::address branch_target, bool taken, uint8_t branch_type)
{
    // Get prediction info from state
    bool provider_hit = last_prediction.provider_hit;
    std::size_t provider_index = last_prediction.provider_index;
    std::size_t provider_entry = last_prediction.provider_entry;
    bool alt_prediction = last_prediction.alt_prediction;
    
    // Neural component
    std::size_t neuron_idx = last_prediction.neuron_index;
    int output = last_prediction.output;
    std::size_t local_hist_idx = last_prediction.local_history_index;
    const auto& local_hist = last_prediction.local_history;
    
    // Calculate actual predictions
    bool provider_prediction;
    if (provider_hit) {
        provider_prediction = tagged_tables[provider_index][provider_entry].counter.value() >= 
                             (tagged_tables[provider_index][provider_entry].counter.maximum / 2);
    } else {
        provider_prediction = bimodal_table[provider_entry].value() >= (bimodal_table[provider_entry].maximum / 2);
    }
    
    bool neural_prediction = output >= 0;
    
    // Train neural predictor (always)
    train_neuron(neuron_idx, local_hist, taken);
    
    // Update local history
    update_local_history(local_hist_idx, taken);
    
    // Update TAGE useful counter if predictions differ
    if (provider_hit && provider_prediction != alt_prediction) {
        if (provider_prediction == taken) {
            // Provider was correct
            tagged_tables[provider_index][provider_entry].useful += 1;
        } else {
            // Provider was wrong
            tagged_tables[provider_index][provider_entry].useful -= 1;
        }
    }
    
    // Update provider counter
    if (provider_hit) {
        tagged_tables[provider_index][provider_entry].counter += taken ? 1 : -1;
    } else {
        bimodal_table[provider_entry] += taken ? 1 : -1;
    }
    
    // Allocate new entries when prediction is wrong
    bool final_prediction = std::abs(output) > 14 ? neural_prediction : provider_prediction;
    
    if (final_prediction != taken) {
        // Try to allocate in a random table with longer history than provider
        std::size_t start_table = provider_hit ? provider_index + 1 : 0;
        
        // Find tables with no useful entries
        std::vector<std::size_t> candidates;
        for (std::size_t i = start_table; i < NUM_TABLES; i++) {
            std::size_t index = last_prediction.indices[i];
            if (tagged_tables[i][index].useful.value() == 0) {
                candidates.push_back(i);
            }
        }
        
        // If no unused entries, reduce u bits and try again
        if (candidates.empty()) {
            for (std::size_t i = start_table; i < NUM_TABLES; i++) {
                std::size_t index = last_prediction.indices[i];
                if (tagged_tables[i][index].useful.value() > 0) {
                    tagged_tables[i][index].useful -= 1;
                }
            }
            
            // Try again to find candidates
            for (std::size_t i = start_table; i < NUM_TABLES; i++) {
                std::size_t index = last_prediction.indices[i];
                if (tagged_tables[i][index].useful.value() == 0) {
                    candidates.push_back(i);
                }
            }
        }
        
        // Allocate in a random candidate table
        if (!candidates.empty()) {
            seed = (seed * 1103515245 + 12345) & 0x7fffffff;
            std::size_t selected = candidates[seed % candidates.size()];
            std::size_t index = last_prediction.indices[selected];
            
            // Initialize new entry
            tagged_tables[selected][index].tag = last_prediction.tags[selected];
            tagged_tables[selected][index].counter = taken ? 
                (tagged_tables[selected][index].counter.maximum / 2) + 1 : 
                (tagged_tables[selected][index].counter.maximum / 2) - 1;
            tagged_tables[selected][index].useful = 0;
        }
    }
    
    update_histories(taken, pc.slice_lower<champsim::data::bits{16}>().to<uint16_t>());
    
    // Periodic reset of useful counters
    periodic_reset();
}