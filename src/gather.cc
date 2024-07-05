#include <fcntl.h>
#include <liburing.h>
#include <omp.h>
#include <pthread.h>
#include <unistd.h>

#include "gather.h"
#include "state.h"

#define ALIGNMENT 4096
#define RING_LEN 1024

namespace offgs {
torch::Tensor GatherMemMap(const torch::Tensor& features,
                           const torch::Tensor& idx, int64_t feature_dim) {
  // open file
  int64_t feature_size = feature_dim * sizeof(float);

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  torch::Tensor ret = torch::empty({idx.numel(), feature_dim}, options);

  auto features_data = features.data_ptr<float>();
  auto idx_data = idx.data_ptr<int64_t>();
  auto ret_data = ret.data_ptr<float>();

#pragma omp parallel for num_threads(64)
  for (int64_t i = 0; i < idx.numel(); i++) {
    memcpy(ret_data + feature_dim * i,
           features_data + idx_data[i] * feature_dim, feature_size);
  }

  return ret;
}

torch::Tensor GatherPRead(const std::string& feature_file,
                          const torch::Tensor& idx, int64_t feature_dim) {
  int feature_fd = open(feature_file.c_str(), O_RDONLY);
  int64_t feature_size = feature_dim * sizeof(float);

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  torch::Tensor ret = torch::empty({idx.numel(), feature_dim}, options);

  auto idx_data = idx.data_ptr<int64_t>();
  auto ret_data = ret.data_ptr<float>();

#pragma omp parallel for num_threads(64)
  for (int64_t i = 0; i < idx.numel(); i++) {
    if (pread(feature_fd, ret_data + feature_dim * i, feature_size,
              idx_data[i] * feature_dim) == -1) {
      fprintf(stderr, "ERROR: %s\n", strerror(errno));
    }
  }

  close(feature_fd);

  return ret;
}

torch::Tensor GatherPReadDirect(const std::string& feature_file,
                                const torch::Tensor& idx, int64_t feature_dim) {
  int feature_fd = open(feature_file.c_str(), O_RDONLY | O_DIRECT);
  int64_t feature_size = feature_dim * sizeof(float);
  int64_t num_idx = idx.numel();

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  torch::Tensor ret = torch::empty({idx.numel(), feature_dim}, options);

  float* read_buffer =
      (float*)aligned_alloc(ALIGNMENT, ALIGNMENT * 2 * num_idx);
  float* result_buffer = ret.data_ptr<float>();
  auto idx_data = idx.data_ptr<int64_t>();

#pragma omp parallel for num_threads(64)
  for (int64_t i = 0; i < idx.numel(); i++) {
    int64_t node_idx = idx_data[i];
    int64_t offset = node_idx * feature_size;
    int64_t aligned_offset = offset & (long)~(ALIGNMENT - 1);
    int64_t residual = offset - aligned_offset;

    int64_t read_size =
        (residual + feature_size > ALIGNMENT) ? ALIGNMENT * 2 : ALIGNMENT;

    if (pread(feature_fd, read_buffer + (ALIGNMENT * 2 * i) / sizeof(float),
              read_size, aligned_offset) == -1) {
      fprintf(stderr, "ERROR: %s\n", strerror(errno));
    }

    memcpy(result_buffer + feature_dim * i,
           read_buffer + (ALIGNMENT * 2 * i + residual) / sizeof(float),
           feature_size);
  }

  free(read_buffer);
  close(feature_fd);

  return ret;
}

torch::Tensor GatherIOUringDirect(const std::string& feature_file,
                                  const torch::Tensor& idx,
                                  int64_t feature_dim) {
  int feature_fd = open(feature_file.c_str(), O_RDONLY | O_DIRECT);
  int64_t feature_size = feature_dim * sizeof(float);
  int64_t num_idx = idx.numel();

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  torch::Tensor ret = torch::empty({idx.numel(), feature_dim}, options);

  float* read_buffer =
      (float*)aligned_alloc(ALIGNMENT, ALIGNMENT * 2 * num_idx);
  float* result_buffer = ret.data_ptr<float>();
  auto idx_data = idx.data_ptr<int64_t>();
  int num_read = idx.numel();

  auto* uring_state = IOUringState::Global();
  auto& ring_arr = uring_state->ring_arr;

#pragma omp parallel for num_threads(NUM_RING)
  for (int i = 0; i < num_read; i += RING_LEN) {
    int thd_id = omp_get_thread_num();
    auto& ring = ring_arr[thd_id];

    int r = std::min(num_read, i + RING_LEN);
    struct io_uring_sqe* sqe;
    for (int j = i; j < r; j++) {
      int64_t node_idx = idx_data[j];
      int64_t offset = node_idx * feature_size;
      int64_t aligned_offset = offset & (long)~(ALIGNMENT - 1);
      int64_t residual = offset - aligned_offset;

      int64_t read_size =
          (residual + feature_size > ALIGNMENT) ? ALIGNMENT * 2 : ALIGNMENT;

      sqe = io_uring_get_sqe(&ring);
      if (!sqe) {
        LOG(FATAL) << "Could not get SQE." << " i: " << i << " j: " << j;
      }
      io_uring_prep_read(sqe, feature_fd,
                         read_buffer + (ALIGNMENT * 2 * j) / sizeof(float),
                         read_size, aligned_offset);
      if ((j + 1) % 64 == 0 || j == r - 1) {
        io_uring_submit(&ring);
      }
    }

    int finish_num = 0;
    while (finish_num < r - i) {
      struct io_uring_cqe* cqe;
      int ret = io_uring_wait_cqe(&ring, &cqe);
      if (ret < 0) {
        LOG(FATAL) << "Error waiting for completion: " << strerror(-ret);
      }
      struct io_uring_cqe* cqes[RING_LEN];
      int cqecount = io_uring_peek_batch_cqe(&ring, cqes, RING_LEN);
      if (cqecount == -1) {
        LOG(FATAL) << "Error peeking batch data";
      }
      io_uring_cq_advance(&ring, cqecount);
      finish_num += cqecount;
    }
  }

  for (int i = 0; i < num_read; i++) {
    int64_t node_idx = idx_data[i];
    int64_t offset = node_idx * feature_size;
    int64_t aligned_offset = offset & (long)~(ALIGNMENT - 1);
    int64_t residual = offset - aligned_offset;

    memcpy(result_buffer + feature_dim * i,
           read_buffer + (ALIGNMENT * 2 * i + residual) / sizeof(float),
           feature_size);
  }

  free(read_buffer);
  close(feature_fd);

  return ret;
}

void GatherInMem(torch::Tensor& out, const torch::Tensor& out_idx,
                 const torch::Tensor& in, const torch::Tensor& in_idx,
                 const torch::Tensor& map_table) {
  assert(out_idx.numel() == in_idx.numel());
  auto num_idx = in_idx.numel();
  assert(out.sizes()[1] == in.sizes()[1]);
  auto feature_dim = out.sizes()[1];
  auto feature_size = feature_dim * sizeof(float);

  auto out_dataptr = out.data_ptr<float>();
  auto in_dataptr = in.data_ptr<float>();
  auto map_ptr = map_table.data_ptr<int32_t>();
  auto in_idx_ptr = in_idx.data_ptr<int64_t>();
  auto out_idx_ptr = out_idx.data_ptr<int64_t>();

#pragma omp parallel for num_threads(64)
  for (int64_t i = 0; i < num_idx; i++) {
    memcpy(out_dataptr + out_idx_ptr[i] * feature_dim,
           in_dataptr + map_ptr[in_idx_ptr[i]] * feature_dim, feature_size);
  }
}
}  // namespace offgs