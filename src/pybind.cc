#include <torch/custom_class.h>
#include <torch/script.h>

#include "cuda/difference.h"
#include "gather.h"
#include "load.h"

namespace offgs {

TORCH_LIBRARY(offgs, m) {
  m.def("_CAPI_GatherMemMap", &GatherMemMap);
  m.def("_CAPI_GatherPRead", &GatherPRead);
  m.def("_CAPI_Difference", &cuda::Difference);
  m.def("_CAPI_LoadFeats", &LoadFeats);
  m.def("_CAPI_LoadFeats_ODirect", &LoadFeats_ODirect);
  m.def("_CAPI_LoadTensor", &LoadTensor);
}

}  // namespace offgs