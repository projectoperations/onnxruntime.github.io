// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <unordered_map>
#include <string>

#include <core/graph/graph.h>

#include "op_builder_factory.h"

namespace onnxruntime {
namespace qnn {

OpBuilderRegistrations::OpBuilderRegistrations() {
  {
    CreateSimpleOpBuilder("Add", *this);
    CreateSimpleOpBuilder("Mul", *this);
    CreateSimpleOpBuilder("Abs", *this);
    CreateSimpleOpBuilder("And", *this);
    CreateSimpleOpBuilder("Ceil", *this);
    CreateSimpleOpBuilder("Cast", *this);
    CreateSimpleOpBuilder("Cos", *this);
    CreateSimpleOpBuilder("Div", *this);
    CreateSimpleOpBuilder("Equal", *this);
    CreateSimpleOpBuilder("Exp", *this);
    CreateSimpleOpBuilder("Floor", *this);
    CreateSimpleOpBuilder("Greater", *this);
    CreateSimpleOpBuilder("GreaterOrEqual", *this);
    CreateSimpleOpBuilder("LeakyRelu", *this);
    CreateSimpleOpBuilder("Less", *this);
    CreateSimpleOpBuilder("LessOrEqual", *this);
    CreateSimpleOpBuilder("Log", *this);
    CreateSimpleOpBuilder("Max", *this);
    CreateSimpleOpBuilder("Min", *this);
    CreateSimpleOpBuilder("Neg", *this);
    CreateSimpleOpBuilder("Not", *this);
    CreateSimpleOpBuilder("Or", *this);
    CreateSimpleOpBuilder("Pow", *this);
    CreateSimpleOpBuilder("PRelu", *this);
    CreateSimpleOpBuilder("Relu", *this);
    CreateSimpleOpBuilder("Round", *this);
    CreateSimpleOpBuilder("Where", *this);
    CreateSimpleOpBuilder("Sigmoid", *this);
    CreateSimpleOpBuilder("Sin", *this);
    CreateSimpleOpBuilder("Softmax", *this);
    CreateSimpleOpBuilder("Sqrt", *this);
    CreateSimpleOpBuilder("Sub", *this);
    CreateSimpleOpBuilder("Tanh", *this);
    CreateSimpleOpBuilder("Transpose", *this);

    CreateSimpleOpBuilder("LogSoftmax", *this);
    CreateSimpleOpBuilder("MatMul", *this);
    CreateSimpleOpBuilder("Concat", *this);
  }

  {
    CreateReduceOpBuilder("ReduceMax", *this);
    CreateReduceOpBuilder("ReduceMean", *this);
    CreateReduceOpBuilder("ReduceMin", *this);
    CreateReduceOpBuilder("ReduceProd", *this);
    CreateReduceOpBuilder("ReduceSum", *this);
  }

  {
    CreateConvOpBuilder("Conv", *this);
  }

  {
    CreatePoolOpBuilder("GlobalAveragePool", *this);
    CreatePoolOpBuilder("MaxPool", *this);
  }

  {
    CreateQdqOpBuilder("QuantizeLinear", *this);
    CreateQdqOpBuilder("DequantizeLinear", *this);
  }

  {
    CreateReshapeOpBuilder("Reshape", *this);
    CreateReshapeOpBuilder("Flatten", *this);
    CreateReshapeOpBuilder("Squeeze", *this);
    CreateReshapeOpBuilder("Unsqueeze", *this);
  }

  {
    CreateGemmOpBuilder("Gemm", *this);
  }

  {
    CreateGatherOpBuilder("Gather", *this);
  }

  {
    CreateArgMaxMinOpBuilder("ArgMax", *this);
    CreateArgMaxMinOpBuilder("ArgMin", *this);
  }

  {
    CreateClipOpBuilder("Clip", *this);
  }

  {
    CreateSliceOpBuilder("Slice", *this);
  }

  {
    CreateConvOpBuilder("ConvTranspose", *this);
  }

  {
    CreateSplitOpBuilder("Split", *this);
  }

  {
    CreateResizeOpBuilder("Resize", *this);
  }

  {
    CreateTopKOpBuilder("TopK", *this);
  }

  {
    CreateTileOpBuilder("Tile", *this);
  }

  {
    CreateInstanceNormOpBuilder("InstanceNormalization", *this);
  }
}

const IOpBuilder* GetOpBuilder(const std::string& onnx_op_type) {
  static const OpBuilderRegistrations op_registrations;
  return op_registrations.GetOpBuilderByOnnxOpType(onnx_op_type);
}

}  // namespace qnn
}  // namespace onnxruntime
