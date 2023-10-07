#ifndef OFFGS_GATHER_H_
#define OFFGS_GATHER_H_

#include <torch/torch.h>

namespace offgs {
torch::Tensor GatherMemMap(const torch::Tensor& features, const torch::Tensor& idx, int64_t feature_dim);
torch::Tensor GatherPRead(const std::string& feature_file, const torch::Tensor& idx, int64_t feature_dim);
}  // namespace offgs

#endif