#include <torch/torch.h>

#include "free.h"

namespace offgs {
void FreeTensor(torch::Tensor& tensor) { free(tensor.data_ptr()); }
}  // namespace offgs