#ifndef OFFGS_CUDA_DIFFERENCE_H_
#define OFFGS_CUDA_DIFFERENCE_H_

#include <torch/torch.h>

namespace offgs {
namespace cuda {
torch::Tensor Difference(torch::Tensor t1, torch::Tensor t2);
}
}  // namespace offgs

#endif