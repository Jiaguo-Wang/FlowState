#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include "/sgl-workspace/sglang/python/sglang/kernels/jit/csrc/elementwise/kvcache.cuh"
TVM_FFI_DLL_EXPORT_TYPED_FUNC(store_cache, (StoreKVCacheKernel<2048, 2048, true>::run));
