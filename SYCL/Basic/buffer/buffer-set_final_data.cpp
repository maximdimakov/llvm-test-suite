// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <CL/sycl/stl.hpp>
#include <vector>
#include <memory>
#include <iostream>

int main() {
  using T = bool;
  size_t size = 32;
  const size_t dims = 1;
  cl::sycl::range<dims> r(size);
  std::shared_ptr<T> data_shrd(new T[size], [](T *data) { delete[] data; });
  std::vector<T> data_vector;
  data_vector.reserve(size);

  cl::sycl::queue Queue;

  cl::sycl::mutex_class m;
  {
    cl::sycl::buffer<T, dims> buf_shrd(
        data_shrd, r,
        cl::sycl::property_list{cl::sycl::property::buffer::use_mutex(m)});
    m.lock();
    std::fill(data_shrd.get(), (data_shrd.get() + size), T());
    m.unlock();
    auto data_final = data_vector.begin();
    buf_shrd.set_final_data(data_final);
    buf_shrd.set_write_back(true);

    Queue.submit([&](cl::sycl::handler &cgh) {
      auto Accessor = buf_shrd.get_access<cl::sycl::access::mode::write>(cgh);
      cgh.parallel_for<class FillBuffer>(
          r, [=](cl::sycl::id<1> WIid) { Accessor[WIid] = false; });
    });
  } // Data is copied back

  for (size_t i = 0; i < size; i++) {
    if (data_vector[i] != true)
      assert(false);
  }
}