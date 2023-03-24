// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// The class helps to feed the Sequence as an inference input
    /// It takes NamedOnnxValue. The NamedOnnxValue can be a tensor, sequence or map.
    /// For recursive structures, create nested NamedOnnxValue instances.
    /// For example, a sequence instance would contain a list of NamedOnnxValue instances
    /// that in turn may represent tensors or other ONNX values.
    /// </summary>
    class ManagedOnnxType
    {
        DisposableList<IDisposable> _disposables;
        OrtValue _ortValue;

        internal ManagedOnnxType(NamedOnnxValue namedOnnxValue, NodeMetadata metadata)
        {
            if (namedOnnxValue.ValueType != metadata.OnnxValueType)
            {
                throw new OnnxRuntimeException(ErrorCode.RuntimeException,
                    $"NamedOnnxValue: {namedOnnxValue.Name} has value type: {namedOnnxValue.ValueType} expected: {metadata.OnnxValueType}");
            }

            var disposables = new DisposableList<IDisposable>();
            try
            {
                switch (namedOnnxValue.ValueType)
                {
                    case OnnxValueType.ONNX_TYPE_TENSOR:
                        _ortValue = CreateTensorMapping(namedOnnxValue, metadata, disposables);
                        break;
                    case OnnxValueType.ONNX_TYPE_SEQUENCE:
                        _ortValue = CreateSequence(namedOnnxValue, metadata, disposables);
                        break;
                }
            }
            catch (Exception)
            {
                disposables.Dispose();
                throw;
            }
            _disposables = disposables;
        }

        private OrtValue CreateSequence(NamedOnnxValue namedOnnxValue, NodeMetadata metadata, DisposableList<IDisposable> disposables)
        {
            OrtValue result = null;
            var elementMeta = metadata.AsSequenceMetadata().ElementMeta;
            var elementOnnxValue = elementMeta.OnnxValueType;
            var seqContainer = namedOnnxValue.AsEnumerable<NamedOnnxValue>();

            if(seqContainer is null)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                                               $"NamedOnnxValue: {namedOnnxValue.Name} sequence does not contain NamedOnnxValue elements");
            }

            var sequenceOrtValues = new List<OrtValue>();
            foreach (var element in seqContainer)
            {
                if(elementOnnxValue != element.ValueType)
                {
                    throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                        $"NamedOnnxValue: {namedOnnxValue.Name} sequence element expected to be {elementOnnxValue}, received {element.ValueType}");
                }

                if (element.ValueType == OnnxValueType.ONNX_TYPE_TENSOR)
                {
                    sequenceOrtValues.Add(CreateTensorMapping(element, elementMeta, disposables));
                }
                else if (element.ValueType == OnnxValueType.ONNX_TYPE_SEQUENCE)
                {
                    sequenceOrtValues.Add(CreateSequence(element, elementMeta, disposables));
                }
                //else if (element.ValueType == OnnxValueType.ONNX_TYPE_MAP)
                //{
                //    var map = new ManagedMap(element, element.AsMap<NamedOnnxValue>());
                //    _disposables.Add(map);
                //    _ortValues.Add(map);
                //}
                else
                {
                    throw new OnnxRuntimeException(ErrorCode.RuntimeException,
                                               $"NamedOnnxValue: {element.Name} is not a tensor, sequence or map");
                }
            }

            var ortValPtrs = sequenceOrtValues.ConvertAll(v => v.Handle).ToArray();

            return result;
        }

        private OrtValue CreateTensorMapping(NamedOnnxValue node, NodeMetadata elementMeta, DisposableList<IDisposable> disposables)
        {
            OrtValue result = null;
            switch (elementMeta.ElementDataType)
            {
                case TensorElementType.Float:
                    result = FromManagedTensor<float>(node, elementMeta, disposables);
                    break;
                case TensorElementType.Double:
                    result = FromManagedTensor<double>(node, elementMeta, disposables);
                    break;
                case TensorElementType.Int16:
                    result = FromManagedTensor<short>(node, elementMeta, disposables);
                    break;
                case TensorElementType.UInt16:
                    result = FromManagedTensor<ushort>(node, elementMeta, disposables);
                    break;
                case TensorElementType.Int32:
                    result = FromManagedTensor<int>(node, elementMeta, disposables);
                    break;
                case TensorElementType.UInt32:
                    result = FromManagedTensor<uint>(node, elementMeta, disposables);
                    break;
                case TensorElementType.Int64:
                    result = FromManagedTensor<long>(node, elementMeta, disposables);
                    break;
                case TensorElementType.UInt64:
                    result = FromManagedTensor<ulong>(node, elementMeta, disposables);
                    break;
                case TensorElementType.UInt8:
                    result = FromManagedTensor<byte>(node, elementMeta, disposables);
                    break;
                case TensorElementType.Int8:
                    result = FromManagedTensor<sbyte>(node, elementMeta, disposables);
                    break;
                case TensorElementType.Bool:
                    result = FromManagedTensor<bool>(node, elementMeta, disposables);
                    break;
                case TensorElementType.Float16:
                    result = FromManagedTensor<Float16>(node, elementMeta, disposables);
                    break;
                case TensorElementType.BFloat16:
                    result = FromManagedTensor<BFloat16>(node, elementMeta, disposables);
                    break;
                case TensorElementType.String:
                    result = FromManagedTensor<string>(node, elementMeta, disposables);
                    break;
                default:
                    throw new NotSupportedException("Tensor of element type: " + elementMeta.ElementDataType + " is not supported");
            }
            return result;
        }

        private OrtValue FromManagedTensor<T>(NamedOnnxValue namedOnnxValue, NodeMetadata metadata,
            DisposableList<IDisposable> disposables)
        {
            var ortValue = OrtValue.CreateFromTensorObject(namedOnnxValue.Value,
                out MemoryHandle? memoryHandle, out TensorElementType elementType);
            disposables.Add(ortValue);

            if (memoryHandle.HasValue)
            {
                disposables.Add(memoryHandle);
            }

            if (elementType != metadata.ElementDataType)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                    $"Tensor element data type discovered: {elementType} metadata expected: {metadata.ElementDataType}");
            }

            return ortValue;
        }

    }
}

