// Copyright (c) 2023 (Alan Fang)
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

#include <memory>

#include "fst/determinize.h"

#include "decoder/context_graph.h"
#include "decoder/deep_biasing.h"


namespace wenet {

DeepBias::DeepBias(DeepBiasConfig config) : config_(config) {}

void DeepBias::PadContextUnits() {
  // find context list max_length
  size_t max_length = 0;
  for (const auto& context : context_data_) {
      max_length = std::max(max_length, context.size());
  }

  // padding
  for (auto& context : context_data_) {
      // padding with -1
      context.resize(max_length, -1);
  }
}

void DeepBias::SetHotwords(
  const std::vector<std::string>& contexts,
  const std::shared_ptr<fst::SymbolTable>& unit_table) {
  for (const auto& context : contexts) {
    std::vector<int> units;
    bool no_oov = wenet::SplitContextToUnits(context, unit_table, &units);
    if (!no_oov) {
      LOG(WARNING) << "Ignore unknown unit found during compilation.";
      continue;
    }
    context_data_.emplace_back(units);
    context_data_lengths_.emplace_back(units.size());
  }
  this->PadContextUnits();
}

}  // namespace wenet