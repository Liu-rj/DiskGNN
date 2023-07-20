#include "atomic.h"
#include "cuda_common.h"
#include "difference.h"

namespace offgs {
namespace cuda {
inline __host__ __device__ int UpPower(int key) {
  int ret = 1 << static_cast<uint32_t>(std::log2(key) + 1);
  return ret;
}

__device__ inline uint32_t Hash32Shift(uint32_t key) {
  key = ~key + (key << 15);  // # key = (key << 15) - key - 1;
  key = key ^ (key >> 12);
  key = key + (key << 2);
  key = key ^ (key >> 4);
  key = key * 2057;  // key = (key + (key << 3)) + (key << 11);
  key = key ^ (key >> 16);
  return key;
}

__device__ inline uint64_t Hash64Shift(uint64_t key) {
  key = (~key) + (key << 21);             // key = (key << 21) - key - 1;
  key = key ^ (key >> 24);
  key = (key + (key << 3)) + (key << 8);  // key * 265
  key = key ^ (key >> 14);
  key = (key + (key << 2)) + (key << 4);  // key * 21
  key = key ^ (key >> 28);
  key = key + (key << 31);
  return key;
}

/**
 * @brief Used to judge whether a node is in a node set
 *
 * @tparam IdType
 */
template <typename IdType>
struct IdxQueryHashmap {
  __device__ inline IdxQueryHashmap(IdType* Kptr, IdType* Vptr, size_t numel)
      : kptr(Kptr), vptr(Vptr), capacity(numel){};

  __device__ inline void Insert(IdType key, IdType value) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);
    IdType prev = AtomicCAS(&kptr[pos], kEmptyKey, key);

    while (prev != key and prev != kEmptyKey) {
      pos = hash(pos + delta);
      delta += 1;
      prev = AtomicCAS(&kptr[pos], kEmptyKey, key);
    }

    vptr[pos] = value;
  }

  __device__ inline IdType Query(IdType key) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);

    while (true) {
      if (kptr[pos] == key) {
        return vptr[pos];
      }
      if (kptr[pos] == kEmptyKey) {
        return -1;
      }
      pos = hash(pos + delta);
      delta += 1;
    }

    return -1;
  }

  __device__ inline uint32_t hash(int32_t key) {
    return Hash32Shift(key) & (capacity - 1);
  }

  __device__ inline uint32_t hash(uint32_t key) {
    return Hash32Shift(key) & (capacity - 1);
  }

  __device__ inline uint32_t hash(int64_t key) {
    return static_cast<uint32_t>(Hash64Shift(key)) & (capacity - 1);
  }

  __device__ inline uint32_t hash(uint64_t key) {
    return static_cast<uint32_t>(Hash64Shift(key)) & (capacity - 1);
  }

  IdType kEmptyKey{-1};
  IdType* kptr;
  IdType* vptr;
  uint32_t capacity{0};
};

torch::Tensor Difference(torch::Tensor t1, torch::Tensor t2) {
  int64_t t2_size = t2.numel();
  int64_t dir_size = UpPower(t2_size) * 2;
  torch::Tensor key_buffer = torch::full(dir_size, -1, t2.options());
  torch::Tensor value_buffer = torch::full(dir_size, -1, t2.options());

  using it = thrust::counting_iterator<int64_t>;
  thrust::for_each(thrust::device, it(0), it(t2_size),
                   [key = t2.data_ptr<int64_t>(),
                    _key_buffer = key_buffer.data_ptr<int64_t>(),
                    _value_buffer = value_buffer.data_ptr<int64_t>(),
                    dir_size] __device__(int64_t i) {
                     IdxQueryHashmap<int64_t> hashmap(_key_buffer,
                                                      _value_buffer, dir_size);
                     hashmap.Insert(key[i], 1);
                   });

  torch::Tensor out_mask = torch::zeros_like(t1);

  thrust::for_each(
      thrust::device, it(0), it(t1.numel()),
      [key = t1.data_ptr<int64_t>(), out = out_mask.data_ptr<int64_t>(),
       _key_buffer = key_buffer.data_ptr<int64_t>(),
       _value_buffer = value_buffer.data_ptr<int64_t>(),
       dir_size] __device__(int64_t i) {
        IdxQueryHashmap<int64_t> hashmap(_key_buffer, _value_buffer, dir_size);
        int64_t value = hashmap.Query(key[i]);
        if (value == -1) {
          out[i] = 1;
        }
      });

  torch::Tensor select_index = torch::nonzero(out_mask).reshape(-1);
  return t1.index({select_index});
}
}  // namespace cuda
}  // namespace offgs