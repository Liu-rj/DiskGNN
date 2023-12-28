#ifndef OFFGS_CUDA_TENSOR_OPS_H_
#define OFFGS_CUDA_TENSOR_OPS_H_

#include <torch/torch.h>

namespace offgs {
namespace cuda {
torch::Tensor IndexSearch(torch::Tensor origin_data, torch::Tensor keys);
}  // namespace cuda
}  // namespace gs

#endif