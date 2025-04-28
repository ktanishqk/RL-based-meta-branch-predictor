#include "loop.h"

loop::loop() : branch_predictor(nullptr), bimodal_predictor(nullptr)
{
  // Initialize the loop table
  for (auto& entry : loop_table) {
    entry.valid = false;
    entry.confidence = 0;
    entry.iter_count = 0;
    entry.current_iter = 0;
    entry.dir = false;
  }
}

loop::loop(O3_CPU* cpu) : branch_predictor(cpu), bimodal_predictor(cpu)
{
  // Initialize the loop table
  for (auto& entry : loop_table) {
    entry.valid = false;
    entry.confidence = 0;
    entry.iter_count = 0;
    entry.current_iter = 0;
    entry.dir = false;
  }
}

bool loop::predict_branch(champsim::address ip)
{
  // Get the prediction from the bimodal predictor as a fallback
  bool bimodal_prediction = bimodal_predictor.predict_branch(ip);
  
  // Compute index and tag for the loop predictor
  auto index = hash(ip);
  auto tag = compute_tag(ip);
  
  // Check if we have a valid entry with a matching tag
  bool loop_prediction = bimodal_prediction;
  bool use_loop_predictor = false;
  
  if (loop_table[index].valid && loop_table[index].tag == tag) {
    // Tag hit
    auto& entry = loop_table[index];
    
    // Check if the confidence is high enough and we're not at the predicted exit iteration
    if (entry.confidence.value() >= 3) { // Maximum confidence
      // If we're at the predicted exit iteration, predict the opposite of the loop body direction
      if (entry.current_iter + 1 == entry.iter_count) {
        loop_prediction = !entry.dir;
      } else {
        // Otherwise, predict the loop body direction
        loop_prediction = entry.dir;
      }
      use_loop_predictor = true;
    }
  }
  
  // Save information for the update phase
  last_prediction.index = index;
  last_prediction.loop_prediction = loop_prediction;
  last_prediction.bimodal_prediction = bimodal_prediction;
  last_prediction.used_loop = use_loop_predictor;
  
  // Return the final prediction
  return use_loop_predictor ? loop_prediction : bimodal_prediction;
}

void loop::last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type)
{
  // Update the bimodal predictor
  bimodal_predictor.last_branch_result(ip, branch_target, taken, branch_type);
  
  auto index = last_prediction.index;
  auto tag = compute_tag(ip);
  auto& entry = loop_table[index];
  
  // If we have a tag match
  if (entry.valid && entry.tag == tag) {
    // If the entry is still being trained
    if (entry.confidence.value() < 3) {
      if (taken != entry.dir) {
        // This iteration is different from the pattern - could be a loop exit
        if (entry.current_iter == entry.iter_count) {
          // We correctly predicted the iteration count
          entry.confidence += 1;
          entry.current_iter = 0;
        } else {
          // The iteration count was wrong, retrain
          entry.iter_count = entry.current_iter + 1;
          entry.confidence -= 1;
          if (entry.confidence.value() == 0) {
            // Lost all confidence, invalidate the entry
            entry.valid = false;
          }
          entry.current_iter = 0;
        }
      } else {
        // Same direction as previous iterations in the loop
        entry.current_iter += 1;
        if (entry.current_iter > entry.iter_count) {
          // Iter count was too small, retrain
          entry.iter_count = 0;
          entry.confidence -= 1;
          if (entry.confidence.value() == 0) {
            // Lost all confidence, invalidate the entry
            entry.valid = false;
          }
        }
      }
    } else {
      // The entry is already trained with high confidence
      if (taken != entry.dir) {
        // This iteration is different from the pattern - should be a loop exit
        if (entry.current_iter == entry.iter_count - 1) {
          // We correctly predicted the exit iteration
          entry.current_iter = 0;
        } else {
          // The iteration count was wrong, retrain
          entry.confidence -= 1;
          entry.iter_count = entry.current_iter + 1;
          entry.current_iter = 0;
        }
      } else {
        // Same direction as previous iterations in the loop
        entry.current_iter += 1;
        if (entry.current_iter >= entry.iter_count) {
          // We should have exited the loop but didn't
          entry.confidence -= 1;
          entry.iter_count = 0;
          entry.current_iter = 0;
        }
      }
    }
  } else {
    // No tag match, allocate a new entry if the bimodal predictor was wrong
    if (last_prediction.bimodal_prediction != taken) {
      entry.valid = true;
      entry.tag = tag;
      entry.confidence = 1; // Start with low confidence
      entry.iter_count = 0;
      entry.current_iter = 0;
      entry.dir = taken; // Record the direction for the loop body
    }
  }
}