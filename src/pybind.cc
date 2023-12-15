#include <torch/custom_class.h>
#include <torch/script.h>

#include "cuda/difference.h"
#include "gather.h"
#include "load.h"

namespace offgs {

TORCH_LIBRARY(offgs, m) {
  m.def("_CAPI_GatherMemMap", &GatherMemMap);
  m.def("_CAPI_GatherPRead", &GatherPRead);
  m.def("_CAPI_GatherPReadDirect", &GatherPReadDirect);
  m.def("_CAPI_GatherInMem", &GatherInMem);
  m.def("_CAPI_BuildHashMap", &cuda::BuildHashMap);
  m.def("_CAPI_QueryHashMap", &cuda::QueryHashMap);
  m.def("_CAPI_Difference", &cuda::Difference);
  m.def("_CAPI_LoadFeats", &LoadFeats);
  m.def("_CAPI_LoadFeats_Direct", &LoadFeats_Direct);
  m.def("_CAPI_LoadTensor", &LoadTensor);
}

}  // namespace offgs