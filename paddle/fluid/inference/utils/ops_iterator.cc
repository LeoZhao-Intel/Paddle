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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <sstream>
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/inference/analysis/analyzer.h"

DEFINE_bool(mkldnn, false, "list ops supported by mkldnn");
DEFINE_bool(all_attrs, false, "list all attributes of op");
DEFINE_bool(grad, false, "list all grad ops");

bool Run() {
  // TODO(Leozhao-Intel): just make sure linking necessary libs, need find
  // better way
  paddle::inference::analysis::Analyzer().Run(NULL);
  return true;
}

// Generate a dot file describing the structure of graph.
// To use this tool, run command: ./ops_iterator [options...]
// Options:
//     --mkldnn: list ops supported by mkldnn
//     --all_attrs: list all attributes of op
//     --all_grads: list all grad ops
int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  //  paddle::framework::InitDevices(false);

  auto maps = paddle::framework::OpInfoMap::Instance().map();
  int count = 0;

  std::cout << "OP                            Attrs\n";
  for (auto& it : maps) {
    std::stringstream all_attrs;
    bool mkldnn = false;
    bool grad = false;

    if (it.second.proto_) {
      for (auto& attr : it.second.Proto().attrs()) {
        if (FLAGS_all_attrs) all_attrs << attr.name() << " ";

        if (attr.name() == "use_mkldnn") mkldnn = true;
      }
    } else {
      grad = true;
    }

    if (FLAGS_mkldnn && !mkldnn) continue;

    if (FLAGS_grad && !grad) continue;

    if (FLAGS_all_attrs) {
      std::cout.setf(std::ios::left);
      std::cout.width(30);
      std::cout << it.first;
      std::cout.unsetf(std::ios::left);
      std::cout << all_attrs.str() << "\n";
    } else {
      std::cout << it.first << "\n";
    }
    count++;
  }

  std::cout << "Total Ops number is " << count << "\n";
  return 0;
}

// USE_PASS(infer_clean_graph_pass);
// USE_PASS(graph_viz_pass);
// USE_PASS(graph_to_program_pass);
