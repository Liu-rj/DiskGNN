#include "atomic.h"
#include "cuda_common.h"

#include "tensor_ops.h"

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
  key = (~key) + (key << 21);  // key = (key << 21) - key - 1;
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
struct NodeQueryHashmap {
  __device__ inline NodeQueryHashmap(IdType* Kptr, IdType* Vptr, size_t numel)
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

////////////////////// IndexHashMap Insert ///////////////////////////////
template <typename IdType>
std::tuple<torch::Tensor, torch::Tensor> _IndexHashMapInsert(
    torch::Tensor keys) {
  int num_elem = keys.numel();
  int dir_size = UpPower(num_elem) * 2;
  torch::Tensor key_buffer = torch::full(
      dir_size, -1, torch::dtype(keys.dtype()).device(torch::kCUDA));
  torch::Tensor value_buffer = torch::full(
      dir_size, -1, torch::dtype(keys.dtype()).device(torch::kCUDA));

  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(thrust::device, it(0), it(num_elem),
                   [key = keys.data_ptr<IdType>(),
                    _key_buffer = key_buffer.data_ptr<IdType>(),
                    _value_buffer = value_buffer.data_ptr<IdType>(),
                    dir_size] __device__(IdType i) {
                     NodeQueryHashmap<IdType> hashmap(_key_buffer,
                                                      _value_buffer, dir_size);
                     hashmap.Insert(key[i], i);
                   });

  return {key_buffer, value_buffer};
}

std::tuple<torch::Tensor, torch::Tensor> IndexHashMapInsertCUDA(
    torch::Tensor keys) {
  return _IndexHashMapInsert<int64_t>(keys);
};

////////////////////// IndexHashMap Serach ///////////////////////////////
template <typename IdType>
torch::Tensor _IndexHashMapSearch(torch::Tensor key_buffer,
                                  torch::Tensor value_buffer,
                                  torch::Tensor keys) {
  int num_elem = keys.numel();
  int dir_size = key_buffer.numel();
  torch::Tensor results = torch::empty_like(keys);

  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(thrust::device, it(0), it(num_elem),
                   [key = keys.data_ptr<IdType>(),
                    _key_buffer = key_buffer.data_ptr<IdType>(),
                    _value_buffer = value_buffer.data_ptr<IdType>(), dir_size,
                    out = results.data_ptr<IdType>()] __device__(IdType i) {
                     NodeQueryHashmap<IdType> hashmap(_key_buffer,
                                                      _value_buffer, dir_size);
                     out[i] = hashmap.Query(key[i]);
                   });

  return results;
}

torch::Tensor IndexHashMapSearchCUDA(torch::Tensor key_buffer,
                                     torch::Tensor value_buffer,
                                     torch::Tensor keys) {
  return _IndexHashMapSearch<int64_t>(key_buffer, value_buffer, keys);
};

torch::Tensor IndexSearch(torch::Tensor origin_data, torch::Tensor keys) {
  torch::Tensor key_buffer, value_buffer;

  std::tie(key_buffer, value_buffer) = IndexHashMapInsertCUDA(origin_data);
  torch::Tensor result = IndexHashMapSearchCUDA(key_buffer, value_buffer, keys);
  return result;
}

// Compute min value for all neighbors (dst) of each node (src) and return
// counts for each bucket.
std::vector<torch::Tensor> SegmentedMinHash(torch::Tensor src,
                                            torch::Tensor dst,
                                            torch::Tensor values,
                                            int64_t num_src, int64_t num_dst) {
  int64_t num_ele = src.numel();
  torch::Tensor result = torch::full({num_src}, INT64_MAX, values.options());
  torch::Tensor counts = torch::zeros({num_dst}, values.options());

  using it = thrust::counting_iterator<int64_t>;
  thrust::for_each(
      thrust::device, it(0), it(num_ele),
      [src = src.data_ptr<int64_t>(), dst = dst.data_ptr<int64_t>(),
       values = values.data_ptr<int64_t>(), result = result.data_ptr<int64_t>(),
       counts = counts.data_ptr<int64_t>()] __device__(int64_t i) {
        int64_t src_id = src[i];
        int64_t dst_id = dst[i];
        AtomicMin(&result[src_id], values[dst_id]);
      });

  using it = thrust::counting_iterator<int64_t>;
  thrust::for_each(thrust::device, it(0), it(num_src),
                   [result = result.data_ptr<int64_t>(),
                    counts = counts.data_ptr<int64_t>()] __device__(int64_t i) {
                     int64_t val = result[i];
                     if (val != INT64_MAX) AtomicAdd(&counts[val], 1);
                   });

  return {result, counts};
}
}  // namespace cuda
}  // namespace offgs