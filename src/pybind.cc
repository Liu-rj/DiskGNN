#include <torch/custom_class.h>
#include <torch/script.h>

#include "cuda/difference.h"

namespace offgs {

TORCH_LIBRARY(offgs, m) { m.def("_CAPI_Difference", &cuda::Difference); }

}  // namespace offgs