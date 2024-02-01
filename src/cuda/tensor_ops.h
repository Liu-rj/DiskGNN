#ifndef OFFGS_CUDA_TENSOR_OPS_H_
#define OFFGS_CUDA_TENSOR_OPS_H_

#include <torch/torch.h>

namespace offgs {
namespace cuda {
torch::Tensor IndexSearch(torch::Tensor origin_data, torch::Tensor keys);
std::vector<torch::Tensor> SegmentedMinHash(torch::Tensor src,
                                            torch::Tensor dst,
                                            torch::Tensor values,
                                            int64_t num_src, int64_t num_dst,
                                            bool return_counts);
}  // namespace cuda
}  // namespace offgs

#endif