#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include "/sgl-workspace/sglang/python/sglang/kernels/jit/csrc/elementwise/activation.cuh"
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_activation, (ActivationKernel<bf16_t, true>::run_activation));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_activation_with_rounding, (ActivationKernel<bf16_t, true>::run_activation_with_rounding));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_activation_with_rounding_input_inplace, (ActivationKernel<bf16_t, true>::run_activation_with_rounding_input_inplace));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_activation_filtered, (ActivationKernel<bf16_t, true>::run_activation_filtered));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_unary_activation, (ActivationKernel<bf16_t, true>::run_unary_activation));
