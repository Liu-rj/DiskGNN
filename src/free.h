#ifndef OFFGS_FREE_H_
#define OFFGS_FREE_H_

#include <torch/torch.h>

namespace offgs {
void FreeTensor(torch::Tensor& tensor);
}  // namespace offgs

#endif