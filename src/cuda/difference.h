#ifndef OFFGS_CUDA_DIFFERENCE_H_
#define OFFGS_CUDA_DIFFERENCE_H_

#include <torch/torch.h>

namespace offgs {
namespace cuda {
std::tuple<torch::Tensor, torch::Tensor> BuildHashMap(const torch::Tensor& t2);

std::vector<torch::Tensor> QueryHashMap(const torch::Tensor& t1,
                                        const torch::Tensor& key_buffer,
                                        const torch::Tensor& value_buffer);

std::vector<torch::Tensor> Difference(const torch::Tensor& t1,
                                      const torch::Tensor& t2,
                                      bool return_intersect = false);
}  // namespace cuda
}  // namespace offgs

#endif