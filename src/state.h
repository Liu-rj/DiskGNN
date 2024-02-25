#ifndef OFFGS_STATE_H_
#define OFFGS_STATE_H_

#include <liburing.h>
#include <array>
#include <queue>

#define NUM_RING 4

namespace offgs {
struct IOUringState {
  std::array<io_uring, NUM_RING> ring_arr;
  std::queue<io_uring> ring_queue;

  static IOUringState *Global() {
    static IOUringState state;
    return &state;
  }
};
}  // namespace offgs

#endif  // OFFGS_STATE_H