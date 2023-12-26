#ifndef OFFGS_CUDA_GATHER_H_
#define OFFGS_CUDA_GATHER_H_

#include <torch/torch.h>

namespace offgs {
namespace cuda {
void GatherInGPU(torch::Tensor& out, const torch::Tensor& out_idx,
                 const torch::Tensor& in_cpu, const torch::Tensor& in_gpu,
                 const torch::Tensor& in_idx, const torch::Tensor& map_table);
}  // namespace cuda
}  // namespace offgs

#endif