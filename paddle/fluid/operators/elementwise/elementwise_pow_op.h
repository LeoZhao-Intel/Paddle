/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cmath>
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"

namespace paddle {
namespace operators {

template <typename T>
struct PowFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return std::pow(a, b); }
};

template <typename DeviceContext, typename T>
class ElementwisePowKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using LoDTensor = framework::LoDTensor;

    auto* x = ctx.Input<LoDTensor>("X");
    auto* y = ctx.Input<LoDTensor>("Y");
    auto* z = ctx.Output<LoDTensor>("Out");

    z->Resize(x->dims());
    z->set_lod(x->lod());
    z->set_layout(x->layout());
    z->mutable_data<T>(ctx.GetPlace());
    int axis = ctx.Attr<int>("axis");
    ElementwiseComputeEx<PowFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                          PowFunctor<T>(), z);
  }
};

}  // namespace operators
}  // namespace paddle
