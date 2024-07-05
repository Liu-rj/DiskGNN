#ifndef OFFGS_GATHER_H_
#define OFFGS_GATHER_H_

#include <torch/torch.h>

namespace offgs {
torch::Tensor GatherMemMap(const torch::Tensor& features,
                           const torch::Tensor& idx, int64_t feature_dim);

torch::Tensor GatherPRead(const std::string& feature_file,
                          const torch::Tensor& idx, int64_t feature_dim);

torch::Tensor GatherPReadDirect(const std::string& feature_file,
                                const torch::Tensor& idx, int64_t feature_dim);

torch::Tensor GatherIOUringDirect(const std::string& feature_file,
                                  const torch::Tensor& idx,
                                  int64_t feature_dim);

void GatherInMem(torch::Tensor& out, const torch::Tensor& out_idx,
                 const torch::Tensor& in, const torch::Tensor& in_idx,
                 const torch::Tensor& map_table);
}  // namespace offgs

#endif