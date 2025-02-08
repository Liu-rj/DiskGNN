#include <torch/custom_class.h>
#include <torch/script.h>

#include "cuda/difference.h"
#include "cuda/gather.h"
#include "cuda/tensor_ops.h"
#include "free.h"
#include "gather.h"
#include "load.h"
#include "save.h"

namespace offgs {
TORCH_LIBRARY(offgs, m) {
  m.def("_CAPI_GatherMemMap", &GatherMemMap);
  m.def("_CAPI_GatherPRead", &GatherPRead);
  m.def("_CAPI_GatherPReadDirect", &GatherPReadDirect);
  m.def("_CAPI_GatherIOUringDirect", &GatherIOUringDirect);
  m.def("_CAPI_GatherInMem", &GatherInMem);
  m.def("_CAPI_GatherInGPU", &cuda::GatherInGPU);
  m.def("_CAPI_GatherInGPU_MegaBatch", &cuda::GatherInGPU_MegaBatch);
  m.def("_CAPI_GatherInGPU_MergeMiniBatch", &cuda::GatherInGPU_MergeMiniBatch);
  m.def("_CAPI_BuildHashMap", &cuda::BuildHashMap);
  m.def("_CAPI_QueryHashMap", &cuda::QueryHashMap);
  m.def("_CAPI_Difference", &cuda::Difference);
  m.def("_CAPI_LoadFeats", &LoadFeats);
  m.def("_CAPI_LoadFeats_Direct", &LoadFeats_Direct);
  m.def("_CAPI_LoadFeats_Direct_lseek", &LoadFeats_Direct_lseek);
  m.def("_CAPI_LoadFeats_Direct_OMP", &LoadFeats_Direct_OMP);
  m.def("_CAPI_LoadDiskCache_Direct_OMP", &LoadDiskCache_Direct_OMP);
  m.def("_CAPI_LoadDiskCache_Direct_OMP_iouring",
        &LoadDiskCache_Direct_OMP_iouring);
  m.def("_CAPI_LoadTensor", &LoadTensor);
  m.def("_CAPI_SaveFeats", &SaveFeats);
  m.def("_CAPI_SaveFeatsAppend", &SaveFeatsAppend);
  m.def("_CAPI_IndexSearch", &cuda::IndexSearch);
  m.def("_CAPI_FreeTensor", &FreeTensor);
  m.def("_CAPI_SegmentedMinHash", &cuda::SegmentedMinHash);
  m.def("_CAPI_Init_iouring", &Init_iouring);
  m.def("_CAPI_Exit_iouring", &Exit_iouring);
}

}  // namespace offgs