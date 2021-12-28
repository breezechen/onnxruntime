// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <memory_resource>
#include "core/framework/allocator.h"

#pragma warning(push)
#pragma warning(disable : 4127)
#include <absl/container/inlined_vector.h>
#include <absl/container/flat_hash_set.h>
#include <absl/container/flat_hash_map.h>
#pragma warning(pop)

namespace onnxruntime {

template <typename T, size_t N>
using InlinedVector = absl::InlinedVector<T, N>;

template <typename T>
using InlinedHashSet = absl::flat_hash_set<T>;

template <typename K, typename V>
using InlinedHashMap = absl::flat_hash_map<K, V>;

namespace pmr {
template <typename T, size_t N>
using InlinedVector = absl::InlinedVector<T, N, std::pmr::polymorphic_allocator<T>>;

template <typename T, typename Hash = absl::container_internal::hash_default_hash<T>, typename Eq = absl::container_internal::hash_default_eq<T>>
using InlinedHashSet = absl::flat_hash_set<T, Hash, Eq, std::pmr::polymorphic_allocator<T>>;

template <typename K, typename V,
          typename Hash = absl::container_internal::hash_default_hash<K>,
          typename Eq = absl::container_internal::hash_default_eq<K>>
    using InlinedHashMap = absl::flat_hash_map<K, V, Hash, Eq, std::pmr::polymorphic_allocator<std::pair<const K, V>>>;
}  // namespace pmr

#ifdef _MSC_VER
#define ORT_ALLOCA(s) _alloca(s)
constexpr size_t kOrtStackAllocationLimitBytes = 4 * 1024;
#elif defined(__GNUC__) || defined(__clang__)
#define ORT_ALLOCA(s) alloca(s)
constexpr size_t kOrtStackAllocationLimitBytes = 4 * 1024;
#else
// always on the heap
#define ORT_ALLOCA(s) nullptr
constexpr size_t kOrtStackAllocationLimitBytes = 0;
#endif

namespace ort_alloca_internal {
inline void* allocate_and_align(size_t size, size_t alignment, std::unique_ptr<uint8_t[]>& buf) {
  size_t to_allocate;
  bool result = IAllocator::CalcMemSizeForArrayWithAlignment(size, sizeof(uint8_t), alignment, &to_allocate);
  if (!result) {
    return nullptr;
  }
  buf = std::make_unique<uint8_t[]>(to_allocate);
  void* ptr = buf.get();
  return std::align(alignment, to_allocate, ptr, to_allocate);
}

inline void* allocate_and_align(const AllocatorPtr& allocator, size_t size, size_t alignment,
                                IAllocatorUniquePtr<void>& buf) {
  size_t to_allocate;
  bool result = IAllocator::CalcMemSizeForArrayWithAlignment(size, sizeof(uint8_t), alignment, &to_allocate);
  if (!result) {
    return nullptr;
  }
  buf = IAllocator::MakeUniquePtr<void>(allocator, to_allocate);
  void* ptr = buf.get();
  return std::align(alignment, to_allocate, ptr, to_allocate);
}

}  // namespace ort_alloca_internal

// Dynamically allocated size on the stack
#define OrtDeclareAllignedStackBuffer(buffer_ptr, size_in_bytes, alignment)                                        \
  std::unique_ptr<uint8_t[]> on_heap_##buffer_ptr;                                                                 \
  void* buffer_ptr = (size_in_bytes > kOrtStackAllocationLimitBytes)                                               \
                         ? ort_alloca_internal::allocate_and_align(size_in_bytes, alignment, on_heap_##buffer_ptr) \
                         : ORT_ALLOCA(size_in_bytes)

#define OrtDeclareAlignedStackOrAllocatedBuffer(alloc, buffer_ptr, size_in_bytes, alignment)                            \
  IAllocatorUniquePtr<void> guard_##buffer_ptr;                                                                         \
  void* buffer_ptr = (size_in_bytes > kOrtStackAllocationLimitBytes)                                                    \
                         ? ort_alloca_internal::allocate_and_align(alloc, size_in_bytes, alignment, guard_##buffer_ptr) \
                         : ORT_ALLOCA(size_in_bytes)

// This gives a set size stackbuffer
template <typename T, size_t N>
class SmallBuffer {
  T buffer_[N];

 public:
  T* Buffer() noexcept { return buffer_; }
  constexpr size_t size() const noexcept { return N; }
  constexpr size_t size_in_bytes() const noexcept { return sizeof(T) * N; }
};

class SmallBufferResource {
  std::pmr::monotonic_buffer_resource resource_;

 public:
  SmallBufferResource(void* ptr, size_t size_in_bytes)
      : resource_(ptr, size_in_bytes, std::pmr::get_default_resource()) {}
  SmallBufferResource(void* ptr, size_t size_in_bytes, std::pmr::memory_resource* upstream)
      : resource_(ptr, size_in_bytes, upstream) {}
  std::pmr::memory_resource* resource() noexcept { return &resource_; }
  std::pmr::memory_resource* upstream() const noexcept { return resource_.upstream_resource(); }
};

}  // namespace onnxruntime
