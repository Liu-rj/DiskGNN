#ifndef OFFGS_CUDA_DIFFERENCE_H_
#define OFFGS_CUDA_DIFFERENCE_H_

#include <torch/torch.h>

namespace offgs {
namespace cuda {
std::tuple<torch::Tensor, torch::Tensor> Difference(const torch::Tensor& t1,
                                                    const torch::Tensor& t2);
}
}  // namespace offgs

#endif