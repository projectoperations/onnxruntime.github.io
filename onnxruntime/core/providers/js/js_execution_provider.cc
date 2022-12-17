// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "js_execution_provider.h"

#include "core/graph/function_utils.h"
#include "core/framework/compute_capability.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/kernel_registry.h"
#include "core/providers/shared/node_unit/node_unit.h"
#include "allocator.h"
#include "data_transfer.h"
#include "js_kernel_lookup.h"

namespace onnxruntime {

namespace js {
template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

class Memcpy final : public OpKernel {
 public:
  Memcpy(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    const auto* X = ctx->Input<Tensor>(0);
    Tensor* Y = ctx->Output(0, X->Shape());
    auto* data_transfer = Info().GetDataTransferManager().GetDataTransfer(X->Location().device, Y->Location().device);
    return data_transfer->CopyTensorAsync(*X, *Y, *ctx->GetComputeStream());
  }
};

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .ExecQueueId(0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .ExecQueueId(1)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

#define KERNEL_CREATE_INFO_VERSIONED(Start, End, Op) \
  BuildKernelCreateInfo<                             \
      ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, Start, End, Op)>

#define KERNEL_CREATE_INFO(Start, Op) \
  BuildKernelCreateInfo<              \
      ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, Start, Op)>

#define KERNEL_CREATE_INFO_TYPED(Start, type, Op) \
  BuildKernelCreateInfo<                          \
      ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, Start, type, Op)>

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 6, 12, Abs);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 13, Abs);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 6, 12, Neg);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 13, Neg);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 6, 12, Floor);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 13, Floor);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 6, 12, Ceil);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 13, Ceil);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 6, 12, Reciprocal);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 13, Reciprocal);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 6, 12, Sqrt);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 13, Sqrt);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 6, 12, Exp);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 13, Exp);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 9, 12, Erf);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 13, Erf);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 7, Sin);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 7, Cos);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 7, Tan);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 7, Asin);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 7, Acos);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 7, Atan);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 9, Sinh);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 9, Cosh);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 9, Asinh);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 9, Acosh);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 9, Atanh);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 6, 10, Clip);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 11, 11, Clip);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 12, 12, Clip);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 13, Clip);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 6, Elu);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 7, 12, Add);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 13, 13, Add);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 14, Add);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 7, 12, Sub);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 13, 13, Sub);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 14, Sub);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 7, 12, Mul);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 13, 13, Mul);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 14, Mul);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 7, 12, Div);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 13, 13, Div);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 14, Div);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 7, 11, Pow);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 12, 12, Pow);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 13, 14, Pow);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 15, Pow);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 1, 12, Shape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 13, 14, Shape);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 15, Shape);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 5, 12, Reshape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 13, 13, Reshape);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 14, Reshape);

//class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSInternalNHWCDomain, 1, 10, float, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSInternalNHWCDomain, 11, float, Conv);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 1, 10, float, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 11, float, Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 7, 8, float, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 9, 10, float, Gemm);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 11, float, Gemm);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 11, float, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 1, float, GlobalAveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 8, 9, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 10, 10, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 11, 11, float, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 12, float, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 1, float, GlobalMaxPool);
// class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 1, 12, Softmax);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 13, Softmax);

// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSInternalNHWCDomain, 10, uint8_t, QLinearConv);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSInternalNHWCDomain, 10, int8_t, QLinearConv);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSInternalNHWCDomain, 1, QLinearAveragePool);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider,
//                                       kDynamicDomainByCreate, 1, QLinearSoftmax);

std::unique_ptr<KernelRegistry> RegisterKernels() {
  auto kernel_registry = std::make_unique<onnxruntime::KernelRegistry>();

  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list becoming empty after ops-reducing
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,

      // element-wise operators
      // unary - math
      KERNEL_CREATE_INFO_VERSIONED(6, 12, Abs),
      KERNEL_CREATE_INFO(13, Abs),
      KERNEL_CREATE_INFO_VERSIONED(6, 12, Neg),
      KERNEL_CREATE_INFO(13, Neg),
      KERNEL_CREATE_INFO_VERSIONED(6, 12, Floor),
      KERNEL_CREATE_INFO(13, Floor),
      KERNEL_CREATE_INFO_VERSIONED(6, 12, Ceil),
      KERNEL_CREATE_INFO(13, Ceil),
      KERNEL_CREATE_INFO_VERSIONED(6, 12, Reciprocal),
      KERNEL_CREATE_INFO(13, Reciprocal),
      KERNEL_CREATE_INFO_VERSIONED(6, 12, Sqrt),
      KERNEL_CREATE_INFO(13, Sqrt),
      KERNEL_CREATE_INFO_VERSIONED(6, 12, Exp),
      KERNEL_CREATE_INFO(13, Exp),
      KERNEL_CREATE_INFO_VERSIONED(9, 12, Erf),
      KERNEL_CREATE_INFO(13, Erf),

      KERNEL_CREATE_INFO(7, Sin),
      KERNEL_CREATE_INFO(7, Cos),
      KERNEL_CREATE_INFO(7, Tan),
      KERNEL_CREATE_INFO(7, Asin),
      KERNEL_CREATE_INFO(7, Acos),
      KERNEL_CREATE_INFO(7, Atan),
      KERNEL_CREATE_INFO(9, Sinh),
      KERNEL_CREATE_INFO(9, Cosh),
      KERNEL_CREATE_INFO(9, Asinh),
      KERNEL_CREATE_INFO(9, Acosh),
      KERNEL_CREATE_INFO(9, Atanh),

      // activations
      KERNEL_CREATE_INFO_VERSIONED(6, 10, Clip),
      KERNEL_CREATE_INFO_VERSIONED(11, 11, Clip),
      KERNEL_CREATE_INFO_VERSIONED(12, 12, Clip),
      KERNEL_CREATE_INFO(13, Clip),
      KERNEL_CREATE_INFO(6, Elu),

      // binary - math
      KERNEL_CREATE_INFO_VERSIONED(7, 12, Add),
      KERNEL_CREATE_INFO_VERSIONED(13, 13, Add),
      KERNEL_CREATE_INFO(14, Add),
      KERNEL_CREATE_INFO_VERSIONED(7, 12, Sub),
      KERNEL_CREATE_INFO_VERSIONED(13, 13, Sub),
      KERNEL_CREATE_INFO(14, Sub),
      KERNEL_CREATE_INFO_VERSIONED(7, 12, Mul),
      KERNEL_CREATE_INFO_VERSIONED(13, 13, Mul),
      KERNEL_CREATE_INFO(14, Mul),
      KERNEL_CREATE_INFO_VERSIONED(7, 12, Div),
      KERNEL_CREATE_INFO_VERSIONED(13, 13, Div),
      KERNEL_CREATE_INFO(14, Div),
      KERNEL_CREATE_INFO_VERSIONED(7, 11, Pow),
      KERNEL_CREATE_INFO_VERSIONED(12, 12, Pow),
      KERNEL_CREATE_INFO_VERSIONED(13, 14, Pow),
      KERNEL_CREATE_INFO(15, Pow),

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 1, 12, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 13, 14, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 15, Shape)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 5, 12, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 13, 13, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 14, Reshape)>,

      //BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSInternalNHWCDomain, 1, 10, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSInternalNHWCDomain, 11, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 1, 10, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 11, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 7, 8, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 9, 10, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 11, float, Gemm)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 11, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 1, float, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 8, 9, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 10, 10, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 11, 11, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 12, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 1, float, GlobalMaxPool)>,
      // // layout insensitive, use ONNX-domain directly
      // BuildKernelCreateInfo<
      //     ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 13, Softmax)>,
      // BuildKernelCreateInfo<
      //     ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 1, 12, Softmax)>,

      // //  quantization op
      // KERNEL_CREATE_INFO_TYPED(10, uint8_t, QLinearConv),
      // KERNEL_CREATE_INFO_TYPED(10, int8_t, QLinearConv),
      // KERNEL_CREATE_INFO(1, QLinearAveragePool),
      // BuildKernelCreateInfo<
      //     ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kDynamicDomainByCreate, 1, QLinearSoftmax)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_THROW_IF_ERROR(kernel_registry->Register(std::move(info)));
    }
  }

  return kernel_registry;
}

}  // namespace js

using namespace js;

JsExecutionProvider::JsExecutionProvider(const JsExecutionProviderInfo& info)
    : IExecutionProvider{kJsExecutionProvider, true} {
}

// implement RegisterAllocator to test/validate sharing the CPU EP's allocator
void JsExecutionProvider::RegisterAllocator(AllocatorManager& allocator_manager) {

  printf("JsExecutionProvider::RegisterAllocator() \n");

  AllocatorCreationInfo cpuInputAllocatorCreationInfo([&](int) {
    return std::make_unique<js::JsCPUInputAllocator>();
  });
  InsertAllocator(CreateAllocator(cpuInputAllocatorCreationInfo));

  AllocatorCreationInfo cpuOutputAllocatorCreationInfo([&](int) {
    return std::make_unique<js::JsCPUOutputAllocator>();
  });
  InsertAllocator(CreateAllocator(cpuOutputAllocatorCreationInfo));

  // use_arena might have some issue, for this to work need to change
  // https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/framework/execution_frame.cc#L507
  AllocatorCreationInfo memory_info(
      [&](int) { return std::make_unique<js::JsCustomAllocator>(); }, 0, false);

  AllocatorPtr allocator = CreateAllocator(memory_info);
  InsertAllocator(allocator);
}

std::vector<std::unique_ptr<ComputeCapability>> JsExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph,
    const IKernelLookup& kernel_lookup) const {

  auto lookup = JsKernelLookup{kernel_lookup};
  auto list = IExecutionProvider::GetCapability(graph, lookup);
  printf("JsExecutionProvider::GetCapability() results:\n");

  for (size_t i = 0; i < list.size(); i++) {
    auto &nodes = list[i]->sub_graph->nodes;
    printf("  subgraph %zu: %zu node(s)\n", i, list[i]->sub_graph->nodes.size());
    for (size_t j = 0; j < nodes.size(); j++) {
      auto node_index = nodes[j];
      auto *node = graph.GetNode(node_index);
      auto *kernel_info = lookup.LookUpKernel(*node);

      (void)(node_index);
      (void)(node);
      (void)(kernel_info);
      printf("    node[%zu]: [%s][%s][%s]\n", node_index, node->Domain().c_str(), node->OpType().c_str(), node->Name().c_str());

      // if (node->OpType() == "Clip" && node->InputDefs().size() == 3) {
      //   printf("Clip node: [%s] %s, %s\n", node->Name().c_str(), node->InputDefs()[1]->Name().c_str(), node->InputDefs()[2]->Name().c_str());
      //   if (!graph.IsConstantInitializer(node->InputDefs()[1]->Name(), true) ||
      //       !graph.IsConstantInitializer(node->InputDefs()[2]->Name(), true)) {
      //     printf("--erasing\n");
      //     nodes.erase(nodes.begin() + j);
      //     j--;
      //     continue;
      //   }
      // }
    }
  }

  return list;
}

std::shared_ptr<KernelRegistry> JsExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> registry = js::RegisterKernels();
  return registry;
}

std::unique_ptr<onnxruntime::IDataTransfer> JsExecutionProvider::GetDataTransfer() const {
  return std::make_unique<js::DataTransfer>();
}

JsExecutionProvider::~JsExecutionProvider() {
}

}  // namespace onnxruntime