// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/transfer_scope_cache.h"

namespace paddle {
namespace framework {

static std::unordered_map<size_t, Scope*>* x = nullptr;
static std::unordered_set<Scope*>* y = nullptr;

std::unordered_map<size_t, Scope*>& global_transfer_data_cache() {
  if (x == nullptr) x = new std::unordered_map<size_t, Scope*>;
  return *x;
}

std::unordered_set<Scope*>& global_transfer_scope_cache() {
  if (y == nullptr) y = new std::unordered_set<Scope*>;
  return *y;
}

Scope* TryCreateTransferScope(OpKernelType type0, OpKernelType type1,
                              const Scope* scope) {
  Scope* new_scope{nullptr};
  size_t infer_cache_key =
      CombineHash(OpKernelType::Hash()(type0), OpKernelType::Hash()(type1));
  infer_cache_key =
      CombineHash(infer_cache_key, std::hash<const Scope*>()(scope));

  auto it = global_transfer_data_cache().find(infer_cache_key);
  if (it != global_transfer_data_cache().end()) {
    new_scope = global_transfer_data_cache()[infer_cache_key];
  } else {
    new_scope = &scope->NewScope();
    global_transfer_data_cache()[infer_cache_key] = new_scope;
  }
  global_transfer_scope_cache().insert(new_scope);
  return new_scope;
}

void RemoveKidsFromTransferScopeCache(Scope* scope) {
  auto it = global_transfer_scope_cache().find(scope);
  if (it != global_transfer_scope_cache().end()) {
    global_transfer_scope_cache().erase(it);
  }
  for (auto* s : scope->kids()) {
    auto it = global_transfer_scope_cache().find(s);
    if (it != global_transfer_scope_cache().end()) {
      global_transfer_scope_cache().erase(it);
    }
  }

  // remove global transfer data cache
  auto& cache = global_transfer_data_cache();
  for (auto it = cache.begin(); it != cache.end();) {
    if (it->second == scope)
      it = cache.erase(it);
    else
      it++;
  }
}

}  // namespace framework
}  // namespace paddle
