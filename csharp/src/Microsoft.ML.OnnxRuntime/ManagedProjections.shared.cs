// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// The class helps to feed the NamedOnnxValue as an inference input.
    /// it projects managed classes to OrtValues so they can be consumed
    /// by the native onnxruntime library.
    /// The NamedOnnxValue can be a tensor, sequence or map.
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

            int requiredCapacity = 32;
            var disposables = new DisposableList<IDisposable>(requiredCapacity);
            try
            {
                switch (namedOnnxValue.ValueType)
                {
                    case OnnxValueType.ONNX_TYPE_TENSOR:
                        _ortValue = CreateTensorProjection(namedOnnxValue, metadata, disposables);
                        break;
                    case OnnxValueType.ONNX_TYPE_SEQUENCE:
                        _ortValue = CreateSequenceProjection(namedOnnxValue, metadata, disposables);
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

        /// <summary>
        /// The function creates OrtValue objects for each element of the sequence
        /// and then creates an OrtValue for the whole sequence.
        /// </summary>
        /// <param name="namedOnnxValue">NamedOnnxValue containing a IEnumeralbe<NameOnnValue></param>
        /// <param name="metadata">sequence metadata</param>
        /// <param name="disposables">cleanup list</param>
        /// <returns></returns>
        /// <exception cref="OnnxRuntimeException"></exception>
        private OrtValue CreateSequenceProjection(NamedOnnxValue namedOnnxValue, NodeMetadata metadata, DisposableList<IDisposable> disposables)
        {
            OrtValue result = null;
            var elementMeta = metadata.AsSequenceMetadata().ElementMeta;
            var elementOnnxValue = elementMeta.OnnxValueType;
            var seqContainer = namedOnnxValue.AsEnumerable<NamedOnnxValue>();

            if (seqContainer is null)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                                               $"NamedOnnxValue: {namedOnnxValue.Name} sequence does not contain NamedOnnxValue elements");
            }

            // Record all the ortValues belonging to the sequence locally
            var sequenceOrtValues = new List<OrtValue>();
            foreach (var element in seqContainer)
            {
                if (elementOnnxValue != element.ValueType)
                {
                    throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                        $"NamedOnnxValue: {namedOnnxValue.Name} sequence element expected to be {elementOnnxValue}, received {element.ValueType}");
                }

                if (element.ValueType == OnnxValueType.ONNX_TYPE_TENSOR)
                {
                    sequenceOrtValues.Add(CreateTensorProjection(element, elementMeta, disposables));
                }
                else if (element.ValueType == OnnxValueType.ONNX_TYPE_SEQUENCE)
                {
                    sequenceOrtValues.Add(CreateSequenceProjection(element, elementMeta, disposables));
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

            IntPtr[] ortValHandles = new IntPtr[sequenceOrtValues.Count];
            for(int i = 0; i < sequenceOrtValues.Count; i++)
            {
                ortValHandles[i] = sequenceOrtValues[i].Handle;
            }

            using (var memHandle = new Memory<IntPtr>(ortValHandles).Pin())
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateValue(ortValHandles,
                    (UIntPtr)sequenceOrtValues.Count, (IntPtr)OnnxValueType.ONNX_TYPE_SEQUENCE, out IntPtr sequenceHandle));
                result = new OrtValue(sequenceHandle);
            }

            return result;
        }

        private OrtValue CreateMapProjection(NamedOnnxValue node, NodeMetadata elementMeta, DisposableList<IDisposable> disposables)
        {
            OrtValue result = null;
            var mapMeta = elementMeta.AsMapMetadata();
            Debug.Assert(mapMeta != null);
            // Let's figure out the representation of the map inside NamedOnnxValue\
            // and how we query in a non-generic code.

        }


        /// <summary>
        /// This pins memory that is contained within DenseTensor.
        /// </summary>
        /// <param name="node">NodeOnnxValue containing DenseTensor</param>
        /// <param name="elementMeta"></param>
        /// <param name="disposables">cleanup list</param>
        /// <returns></returns>
        /// <exception cref="OnnxRuntimeException"></exception>
        private OrtValue CreateTensorProjection(NamedOnnxValue node, NodeMetadata elementMeta, DisposableList<IDisposable> disposables)
        {
            var ortValue = OrtValue.CreateFromTensorObject(node.Value,
                out MemoryHandle? memoryHandle, out TensorElementType elementType);
            disposables.Add(ortValue);

            if (memoryHandle.HasValue)
            {
                disposables.Add(memoryHandle);
            }

            if (elementType != elementMeta.ElementDataType)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                    $"Tensor element data type discovered: {elementType} metadata expected: {elementMeta.ElementDataType}");
            }

            return ortValue;
        }
    }
}

