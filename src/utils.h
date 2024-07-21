#ifndef OFFGS_UTILS_H_
#define OFFGS_UTILS_H_

#include <torch/torch.h>

namespace offgs {
#define ID_TYPE_SWITCH(SCALAR_TYPE, IdType, ...)                      \
  do {                                                                \
    if ((SCALAR_TYPE) == torch::kInt32) {                             \
      typedef int IdType;                                             \
      { __VA_ARGS__ }                                                 \
    } else if ((SCALAR_TYPE) == torch::kInt64) {                      \
      typedef int64_t IdType;                                         \
      { __VA_ARGS__ }                                                 \
    } else {                                                          \
      LOG(FATAL) << "Index type not recognized with " << SCALAR_TYPE; \
    }                                                                 \
  } while (0)

#define FLOAT_TYPE_SWITCH(SCALAR_TYPE, DType, ...)                   \
  do {                                                               \
    if ((SCALAR_TYPE) == torch::kFloat16) {                          \
      typedef at::Half DType;                                        \
      { __VA_ARGS__ }                                                \
    } else if ((SCALAR_TYPE) == torch::kFloat32) {                   \
      typedef float DType;                                           \
      { __VA_ARGS__ }                                                \
    } else if ((SCALAR_TYPE) == torch::kFloat64) {                   \
      typedef double DType;                                          \
      { __VA_ARGS__ }                                                \
    } else {                                                         \
      LOG(FATAL) << "Data type not recognized with " << SCALAR_TYPE; \
    }                                                                \
  } while (0)

#define ITEMSIZE_TO_FLOAT(ITEM_SIZE, DType, ...)                   \
  do {                                                             \
    if ((ITEM_SIZE) == 2) {                                        \
      typedef at::Half DType;                                      \
      { __VA_ARGS__ }                                              \
    } else if ((ITEM_SIZE) == 4) {                                 \
      typedef float DType;                                         \
      { __VA_ARGS__ }                                              \
    } else if ((ITEM_SIZE) == 8) {                                 \
      typedef double DType;                                        \
      { __VA_ARGS__ }                                              \
    } else {                                                       \
      LOG(FATAL) << "Data size not recognized with " << ITEM_SIZE; \
    }                                                              \
  } while (0)

template <typename T>
struct TorchTypeMap;

template <>
struct TorchTypeMap<at::Half> {
  static constexpr auto value = torch::kFloat16;
};

template <>
struct TorchTypeMap<float> {
  static constexpr auto value = torch::kFloat32;
};

template <>
struct TorchTypeMap<double> {
  static constexpr auto value = torch::kFloat64;
};
}  // namespace offgs

#endif