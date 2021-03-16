// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out
//
//==----------------- get_backend.cpp ------------------------==//
// This is a test of get_backend().
// Do not set SYCL_DEVICE_FILTER. We do not want the preferred
// backend.
//==----------------------------------------------------------==//

#include <CL/sycl.hpp>
#include <CL/sycl/backend_types.hpp>
#include <iostream>

using namespace cl::sycl;

class DUMMY {
public:
  void operator()(item<1>){};
};

bool check(backend be) {
  switch (be) {
  case backend::opencl:
  case backend::level_zero:
  case backend::cuda:
  case backend::host:
    return true;
  default:
    return false;
  }
  return false;
}

inline void return_fail() {
  std::cout << "Failed" << std::endl;
  exit(1);
}

int main() {
  for (const auto &plt : platform::get_platforms()) {
    if (!plt.is_host()) {
      if (check(plt.get_backend()) == false) {
        return_fail();
      }

      context c(plt);
      if (c.get_backend() != plt.get_backend()) {
        return_fail();
      }

      program prog(c);
      prog.compile_with_kernel_type<DUMMY>();
      prog.link("-cl-finite-math-only");
      if (prog.get_backend() != plt.get_backend()) {
        return_fail();
      }

      default_selector sel;
      queue q(c, sel);
      if (q.get_backend() != plt.get_backend()) {
        return_fail();
      }

      auto device = q.get_device();
      if (device.get_backend() != plt.get_backend()) {
        return_fail();
      }

      auto e = q.submit([&](handler &cgh) {
        cgh.single_task<DUMMY>([](){});
      });
      if (e.get_backend() != plt.get_backend()) {
        return_fail();
      }
    }
  }
  std::cout << "Passed" << std::endl;
  return 0;
}
