// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER SYCL_PI_TRACE=-1 SYCL_PROGRAM_LINK_OPTIONS="-cl-fast-relaxed-math" %t.out %CPU_CHECK_PLACEHOLDER --check-prefix=CHECK-IS-FINITE-MATH
// RUN: %GPU_RUN_PLACEHOLDER SYCL_PI_TRACE=-1 SYCL_PROGRAM_LINK_OPTIONS="-cl-fast-relaxed-math" %t.out %GPU_CHECK_PLACEHOLDER --check-prefix=CHECK-IS-FINITE-MATH
// RUN: %ACC_RUN_PLACEHOLDER SYCL_PI_TRACE=-1 SYCL_PROGRAM_LINK_OPTIONS="-cl-fast-relaxed-math" %t.out %ACC_CHECK_PLACEHOLDER --check-prefix=CHECK-IS-FINITE-MATH

#include <CL/sycl.hpp>
#include <iostream>
using namespace cl::sycl;
class DUMMY {
public:
  void operator()(item<1>){};
};

int main(void) {
  default_selector s;
  platform p(s);
  if (p.is_host()) {
    return 0;
  }
  context c(p);
  program pro(c);
  pro.compile_with_kernel_type<DUMMY>();
  pro.link("-cl-finite-math-only");
  assert(pro.get_state() == cl::sycl::program_state::linked && "fail to link program");
  // CHECK-IS-FINITE-MATH: -cl-fast-relaxed-math
  // CHECK-IS-FINITE-MATH-NOT: -cl-finite-math-only
  assert(pro.get_link_options() == "-cl-finite-math-only" && "program::get_link_options() output is wrong");
  return 0;
}