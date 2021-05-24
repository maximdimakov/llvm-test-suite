// RUN: %clangxx %debug_option -fsycl -fsycl-targets=%sycl_triple %s -O0 -o %t.out
// RUN: %clangxx %debug_option -fsycl -fsycl-targets=%sycl_triple %s -O1 -o %t.out
// RUN: %clangxx %debug_option -fsycl -fsycl-targets=%sycl_triple %s -O2 -o %t.out
// RUN: %clangxx %debug_option -fsycl -fsycl-targets=%sycl_triple %s -O3 -o %t.out
// RUN: %clangxx %debug_option -fsycl -fsycl-targets=%sycl_triple %s -Ofast -o %t.out
// RUN: %clangxx %debug_option -fsycl -fsycl-targets=%sycl_triple %s -Os -o %t.out
// RUN: %clangxx %debug_option -fsycl -fsycl-targets=%sycl_triple %s -Oz -o %t.out
// RUN: %clangxx %debug_option -fsycl -fsycl-targets=%sycl_triple %s -Og -o %t.out
// RUN: %clangxx %debug_option -fsycl -fsycl-targets=%sycl_triple %s -O -o %t.out

// NOTE: Tests that debugging information can be generated for all optimization
// levels.

#include <CL/sycl.hpp>

int main() {
  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class kernel_test>(100, [=](sycl::id<1> idx) {});
  });
  q.wait();

  return 0;
}