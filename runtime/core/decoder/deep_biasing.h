// Copyright (c) 2024 (Alan Fang)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DEEPBIAS_ASR_DEEPBIAS_H_
#define DEEPBIAS_ASR_DEEPBIAS_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "fst/fst.h"

namespace wenet {

struct DeepBiasConfig {
  int max_contexts = 5000;
  int max_context_length = 100;
  float deep_bias_score = 1.0;
};

class DeepBias {
 public:
  explicit DeepBias(DeepBiasConfig config);
  void SetHotwords(
    const std::vector<std::string>& contexts,
    const std::shared_ptr<fst::SymbolTable>& unit_table
  );
  void PadContextUnits();
  std::vector<std::vector<int>> GetContextData() {
    return context_data_;
  };

  std::vector<int> GetContextDataLens() {
    return context_data_lengths_;
  };

  const float GetBiasingScore() const {
    return config_.deep_bias_score;
  }

 private:
  DeepBiasConfig config_;
  std::vector<std::vector<int>> context_data_;
  std::vector<int> context_data_lengths_;
};

}  // namespace wenet

#endif  // DEEPBIAS_ASR_DEEPBIAS_H_