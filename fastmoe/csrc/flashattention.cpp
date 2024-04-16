// modified from https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/CPUBlas.h>
#include <mkl.h>
#include <c10/core/SymFloat.h>

#include <ATen/ATen.h>
#include <torch/extension.h>

namespace {                
#define FASTMOE_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                          \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__))

inline c10::SymFloat calculate_scale(
    const at::Tensor& query,
    c10::optional<double> scale) {
  const auto softmax_scale = scale.has_value()
      ? scale.value()
      : (c10::SymFloat(1.0) / (c10::SymFloat(query.sym_size(-1)).sqrt()));
  return c10::SymFloat(softmax_scale);
}


// from Aten/native/cpu/utils.h
template <typename T>
inline void _store(T* dst, at::vec::Vectorized<T> src) {
  src.store(dst);
}

inline void _store(at::BFloat16* dst, at::vec::Vectorized<float> src) {
  auto res = at::vec::convert_float_bfloat16(src, src);
  res.store(dst, at::vec::Vectorized<float>::size());
}

inline void _store(at::Half* dst, at::vec::Vectorized<float> src) {
  auto res = at::vec::convert_float_half(src, src);
  res.store(dst, at::vec::Vectorized<float>::size());
}

template <typename T>
inline T data_index_init(T offset) {
  return offset;
}

template <typename T, typename... Args>
inline T data_index_init(T offset, T& x, const T& X, Args&&... args) {
  offset = data_index_init(offset, std::forward<Args>(args)...);
  x = offset % X;
  return offset / X;
}

inline bool data_index_step() {
  return true;
}

template <typename T, typename... Args>
inline bool data_index_step(T& x, const T& X, Args&&... args) {
  if (data_index_step(std::forward<Args>(args)...)) {
    x = ((x + 1) == X) ? 0 : (x + 1);
    return x == 0;
  }
  return false;
}
// Aten/native/cpu/utils.h

// 1) out = exp(a - val)
// 2) val = sum(out)
template <typename T1, typename T2>
inline void _exp_reduce_sum_fusion_kernel(
    T1* a,
    const int& size,
    T2* out,
    T1& val) {
  auto vec_size = at::vec::Vectorized<T1>::size();
  auto vec_max = at::vec::Vectorized<T1>(val);
  T1 tmp_sum = 0;
  auto vec_tmp_sum = at::vec::Vectorized<T1>(tmp_sum);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<T1>::loadu(a + i);
    auto tmp1 = tmp0 - vec_max;
    // auto tmp2 = tmp1.exp_u20();
    auto tmp2 = tmp1.exp();
    vec_tmp_sum += tmp2;
    _store(out + i, tmp2);
  }
  tmp_sum = at::vec::vec_reduce_all<T1>(
      [](at::vec::Vectorized<T1>& x, at::vec::Vectorized<T1>& y) {
        return x + y;
      },
      vec_tmp_sum);
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 - val;
    auto tmp2 = exp(tmp1);
    tmp_sum += tmp2;
    out[i] = tmp2;
  }
  val = tmp_sum;
}

// 1) max(a)
template <typename scalar_t>
inline void _vec_max_kernel(
    const scalar_t* a,
    const int& size,
    scalar_t& max) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  scalar_t tmp_max = -std::numeric_limits<scalar_t>::infinity();
  auto vec_tmp_max = at::vec::Vectorized<scalar_t>(tmp_max);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(a + i);
    vec_tmp_max = at::vec::maximum(vec_tmp_max, tmp0);
  }
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    tmp_max = std::max(tmp_max, tmp0);
  }
  max = std::max(
      tmp_max,
      at::vec::vec_reduce_all<scalar_t>(
          [](at::vec::Vectorized<scalar_t>& x, at::vec::Vectorized<scalar_t>& y) {
            return at::vec::maximum(x, y);
          },
          vec_tmp_max));
}

template <typename scalar_t>
static inline scalar_t* conditional_data_ptr(scalar_t* ptr, scalar_t* ptr2) {
  TORCH_CHECK(ptr2 == nullptr);
  return ptr;
}

template <typename scalar_t,
          typename std::enable_if_t<std::is_reduced_floating_point_v<scalar_t>, int> = 0>
static inline scalar_t* conditional_data_ptr(float* ptr, scalar_t* ptr2) {
  return ptr2;
}

template <typename scalar_t>
inline void fill_stub(scalar_t* data, scalar_t val, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  Vec data_vec = Vec(val);
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    data_vec.store(data + d);
  }
  #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
  # pragma unroll
  #endif
  for (; d < size; d++) {
    data[d] = val;
  }
}

template <typename scalar_t, int64_t q_split_size, int64_t kv_split_size>
void cpu_flash_decode(
    const torch::Tensor& output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& seq_lens,
    const at::Tensor& start_loc,
    c10::optional<double> scale) {
  // Query -> (Batch x 1 x Num_Q_heads  x Dim_per_head)
  // Key -> ([kv_seq_len1, kv_seq_len2, ...] x Num_KV_heads  x Dim_per_head)
  // Value -> ([kv_seq_len1, kv_seq_len2, ...] x Num_KV_heads  x Dim_per_head)

  constexpr bool is_reduced_type = std::is_reduced_floating_point_v<scalar_t>;
  using accum_t = at::opmath_type<scalar_t>;
  using Vec = at::vec::Vectorized<accum_t>;
  accum_t scaling_factor = calculate_scale(query, scale).as_float_unchecked();

  // Sizes
  TORCH_CHECK((query.size(3) == value.size(2)) && (key.size(2) == value.size(2)),
        "token_attention_cpu: Q/K/V should have the same head size");
  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(1);
  int64_t num_head = query.size(2);
  int64_t headSize = query.size(3);
  int64_t num_kv_head = key.size(1);

  // Strides
  int64_t qStrideB = query.stride(0);
  int64_t qStrideM = query.stride(1);
  int64_t qStrideH = query.stride(2);
  // int64_t kStrideB = key.stride(0);
  // int64_t kStrideN = key.stride(1);
  // int64_t kStrideH = key.stride(2);
  // int64_t vStrideB = value.stride(0);
  // int64_t vStrideN = value.stride(1);
  // int64_t vStrideH = value.stride(2);
  int64_t kStrideN = key.stride(0);
  int64_t kStrideH = key.stride(1);
  int64_t vStrideN = value.stride(0);
  int64_t vStrideH = value.stride(1);
  int64_t oStrideB = output.stride(0);
  int64_t oStrideM = output.stride(1);
  int64_t oStrideH = output.stride(2);

  int64_t qSplitSize = q_split_size > qSize ? qSize : q_split_size;
  int64_t kvSplitSize = kv_split_size;
  int64_t qSlice = (qSize - 1) / qSplitSize + 1;
  int64_t num_thread = at::get_num_threads();
  int64_t kv_group_num = num_head / num_kv_head;

  const auto dtype = query.scalar_type();
  const auto accumulate_dtype = at::toOpMathType(dtype);

  // allocate per thread temp buf (accumulate type)
  int64_t size_per_thread =
      /* qk     */ qSplitSize * kvSplitSize +
      /* qk_max */ qSplitSize +
      /* qk_sum */ qSplitSize +
      /* dst    */ qSplitSize * headSize;

  at::Tensor buf = at::zeros({num_thread, size_per_thread}, query.options().dtype(accumulate_dtype));
  at::Tensor buf_reduced = at::zeros({num_thread, qSplitSize, is_reduced_type ? kvSplitSize : 0}, query.options());

  // Data ptrs
  const scalar_t* q_data = query.const_data_ptr<scalar_t>();
  const scalar_t* k_data = key.const_data_ptr<scalar_t>();
  const scalar_t* v_data = value.const_data_ptr<scalar_t>();
  const int64_t* seq_lens_data = seq_lens.data_ptr<int64_t>();
  const int64_t* start_loc_data = start_loc.data_ptr<int64_t>();
  scalar_t* out_data = output.data_ptr<scalar_t>();
  accum_t* buf_data = buf.data_ptr<accum_t>();
  scalar_t* buf_reduced_data = is_reduced_type ? buf_reduced.data_ptr<scalar_t>() : nullptr;

  at::parallel_for(0, batchSize * num_head * qSlice, 1, [&](int64_t begin, int64_t end) {
    int64_t i = 0, j = 0, k = 0;
    data_index_init(begin, i, batchSize, j, num_head, k, qSlice);
    int ompIdx = at::get_thread_num();
    accum_t* buf_ptr = buf_data + ompIdx * size_per_thread;
    accum_t* qk_data = buf_ptr;
    accum_t* qk_max_data = qk_data + qSplitSize * kvSplitSize;
    accum_t* qk_sum_data = qk_max_data + qSplitSize;
    accum_t* dst_data = qk_sum_data + qSplitSize;
    scalar_t* qk_reduced_data = is_reduced_type ? buf_reduced_data + ompIdx * qSplitSize * kvSplitSize : nullptr;

    for (const auto z : c10::irange(begin, end)) {
      (void)z; // Suppress unused variable
      int64_t m = k * qSplitSize;
      int64_t qBlockSize = std::min(qSplitSize, qSize - m);
      // Initialize max and sum
      fill_stub(qk_max_data,
          -std::numeric_limits<accum_t>::infinity(), qBlockSize);
      fill_stub(qk_sum_data,
          static_cast<accum_t>(0), qBlockSize);
      int64_t num_keys = seq_lens_data[i];
      int64_t start_pos = start_loc_data[i];
      int64_t j_kv = j / kv_group_num;
      for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
        int64_t kvBlockSize = std::min(kvSplitSize, num_keys - n);
        // Calculate scale * q @ k.T
        cblas_sgemv(
            CblasRowMajor,
            CblasNoTrans,
            kvBlockSize,
            headSize,
            scaling_factor,
            k_data + j_kv * kStrideH + (start_pos + n) * kStrideN,
            kStrideN,
            q_data + i * qStrideB + j * qStrideH,
            1,
            static_cast<accum_t>(0),
            qk_data,
            1);
        
        // Update coefficients with Softmax
        accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;
        _vec_max_kernel(
            qk_data,
            kvBlockSize,
            tmp_max);
        tmp_max = qk_max_data[0] > tmp_max ? qk_max_data[0] : tmp_max;
        tmp_sum = tmp_max;
        _exp_reduce_sum_fusion_kernel(
            qk_data, 
            kvBlockSize,
            conditional_data_ptr(qk_data, qk_reduced_data),
            tmp_sum);
        exp_tmp = std::exp(qk_max_data[0] - tmp_max);
        qk_sum_data[0] = tmp_sum + exp_tmp * qk_sum_data[0];
        qk_max_data[0] = tmp_max;

        // Calculate Softmax(q @ k.T) @ v
        cblas_sgemv(
            CblasRowMajor,
            CblasTrans,
            kvBlockSize,
            headSize,
            static_cast<accum_t>(1),
            v_data + j_kv * vStrideH + (start_pos + n) * vStrideN,
            vStrideN,
            conditional_data_ptr(qk_data, qk_reduced_data),
            1,
            n == 0 ? static_cast<accum_t>(0) : exp_tmp,
            dst_data,
            1);
      }
      // dst <- dst / sum[row]
      // reorder MHA output with strides)
      accum_t sum_reciprocal = 1 / qk_sum_data[0];
      at::vec::map<scalar_t>(
        [sum_reciprocal](Vec x) { return x * Vec(sum_reciprocal); },
        out_data + i * oStrideB + j * oStrideH + m * oStrideM,
        dst_data,
        headSize);
      // Move to the next query
      data_index_step(i, batchSize, j, num_head, k, qSlice);
    }
  });

}

template <typename scalar_t, int64_t q_split_size, int64_t kv_split_size>
void cpu_flash_decode_gqa(
    const torch::Tensor& output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& seq_lens,
    const at::Tensor& start_loc,
    c10::optional<double> scale) {
  // Query -> (Batch x 1 x Num_Q_heads  x Dim_per_head)
  // Key -> ([kv_seq_len1, kv_seq_len2, ...] x Num_KV_heads  x Dim_per_head)
  // Value -> ([kv_seq_len1, kv_seq_len2, ...] x Num_KV_heads  x Dim_per_head)

  constexpr bool is_reduced_type = std::is_reduced_floating_point_v<scalar_t>;
  using accum_t = at::opmath_type<scalar_t>;
  using Vec = at::vec::Vectorized<accum_t>;
  accum_t scaling_factor = calculate_scale(query, scale).as_float_unchecked();

  // Sizes
  TORCH_CHECK((query.size(3) == value.size(2)) && (key.size(2) == value.size(2)),
        "token_attention_cpu: Q/K/V should have the same head size");
  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(1);
  int64_t num_head = query.size(2);
  int64_t headSize = query.size(3);
  int64_t num_kv_head = key.size(1);

  // Strides
  int64_t qStrideB = query.stride(0);
  int64_t qStrideM = query.stride(1);
  int64_t qStrideH = query.stride(2);
  int64_t kStrideN = key.stride(0);
  int64_t kStrideH = key.stride(1);
  int64_t vStrideN = value.stride(0);
  int64_t vStrideH = value.stride(1);
  int64_t oStrideB = output.stride(0);
  int64_t oStrideM = output.stride(1);
  int64_t oStrideH = output.stride(2);

  int64_t qSplitSize = q_split_size > qSize ? qSize : q_split_size;
  int64_t kvSplitSize = kv_split_size;
  int64_t qSlice = (qSize - 1) / qSplitSize + 1;
  int64_t num_thread = at::get_num_threads();
  int64_t kv_group_num = num_head / num_kv_head;

  const auto dtype = query.scalar_type();
  const auto accumulate_dtype = at::toOpMathType(dtype);

  // allocate per thread temp buf (accumulate type)
  int64_t size_per_thread =
      /* qk     */ kv_group_num * kvSplitSize +
      /* qk_max */ kv_group_num +
      /* qk_sum */ kv_group_num +
      /* dst    */ kv_group_num * headSize;

  at::Tensor buf = at::zeros({num_thread, size_per_thread}, query.options().dtype(accumulate_dtype));
  at::Tensor buf_reduced = at::zeros({num_thread, kv_group_num, is_reduced_type ? kvSplitSize : 0}, query.options());

  // Data ptrs
  const scalar_t* q_data = query.const_data_ptr<scalar_t>();
  const scalar_t* k_data = key.const_data_ptr<scalar_t>();
  const scalar_t* v_data = value.const_data_ptr<scalar_t>();
  const int64_t* seq_lens_data = seq_lens.data_ptr<int64_t>();
  const int64_t* start_loc_data = start_loc.data_ptr<int64_t>();
  scalar_t* out_data = output.data_ptr<scalar_t>();
  accum_t* buf_data = buf.data_ptr<accum_t>();
  scalar_t* buf_reduced_data = is_reduced_type ? buf_reduced.data_ptr<scalar_t>() : nullptr;

  at::parallel_for(0, batchSize * num_kv_head, 1, [&](int64_t begin, int64_t end) {
    int64_t i = 0, j = 0, k = 0;
    data_index_init(begin, i, batchSize, j, num_kv_head, k, qSlice);
    int ompIdx = at::get_thread_num();
    accum_t* buf_ptr = buf_data + ompIdx * size_per_thread;
    accum_t* qk_data = buf_ptr;
    accum_t* qk_max_data = qk_data + kv_group_num * kvSplitSize;
    accum_t* qk_sum_data = qk_max_data + kv_group_num;
    accum_t* dst_data = qk_sum_data + kv_group_num;
    scalar_t* qk_reduced_data = is_reduced_type ? buf_reduced_data + ompIdx * kv_group_num * kvSplitSize : nullptr;

    for (const auto z : c10::irange(begin, end)) {
      (void)z; // Suppress unused variable
      int64_t m = j * kv_group_num;
      int64_t qBlockSize = 1;
      // Initialize max and sum
      fill_stub(qk_max_data,
          -std::numeric_limits<accum_t>::infinity(), kv_group_num);
      fill_stub(qk_sum_data,
          static_cast<accum_t>(0), kv_group_num);
      int64_t num_keys = seq_lens_data[i];
      int64_t start_pos = start_loc_data[i];
      for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
        int64_t kvBlockSize = std::min(kvSplitSize, num_keys - n);
        // Calculate scale * q @ k.T
        // query (kv_group_num, head_size), key (kvBlockSize, head_size), 
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            kv_group_num,
            kvBlockSize,
            headSize,
            scaling_factor,
            q_data + i * qStrideB + m * qStrideH,
            qStrideH,
            k_data + j * kStrideH + (start_pos + n) * kStrideN,
            kStrideN,
            static_cast<accum_t>(0),
            qk_data,
            kvBlockSize);
        
        // Update coefficients with Softmax
        accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;
        for (int64_t row = 0; row < kv_group_num; row++) {
          _vec_max_kernel(
              qk_data + row * kvBlockSize,
              kvBlockSize,
              tmp_max);
          tmp_max = qk_max_data[row] > tmp_max ? qk_max_data[row] : tmp_max;
          tmp_sum = tmp_max;
          _exp_reduce_sum_fusion_kernel(
              qk_data + row * kvBlockSize, 
              kvBlockSize,
              conditional_data_ptr(qk_data, qk_reduced_data)  + row * kvBlockSize,
              tmp_sum);
          exp_tmp = std::exp(qk_max_data[row] - tmp_max);
          qk_sum_data[row] = tmp_sum + exp_tmp * qk_sum_data[row];
          qk_max_data[row] = tmp_max;
          if (n > 0) {
            at::vec::map<accum_t>(
              [exp_tmp](Vec x) { return x * Vec(exp_tmp); },
              dst_data + row * headSize, dst_data + row * headSize, headSize);
          }
        }

        // Calculate Softmax(q @ k.T) @ v
        // qk (kv_group_num, kvBlockSize), v (kvBlockSize, head_size)
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            kv_group_num,
            headSize,
            kvBlockSize,
            static_cast<accum_t>(1),
            conditional_data_ptr(qk_data, qk_reduced_data),
            kvBlockSize,
            v_data + j * vStrideH + (start_pos + n) * vStrideN,
            vStrideN,
            n == 0 ? static_cast<accum_t>(0) : static_cast<accum_t>(1),
            dst_data,
            headSize);
      }
      // dst <- dst / sum[row]
      // reorder MHA output with strides)
      for (int64_t row = 0; row < kv_group_num; ++row) {
        accum_t sum_reciprocal = 1 / qk_sum_data[row];
        at::vec::map<scalar_t>(
          [sum_reciprocal](Vec x) { return x * Vec(sum_reciprocal); },
          out_data + i * oStrideB + (m + row) * oStrideH,
          dst_data + row * headSize,
          headSize);
      }
      // Move to the next query
      data_index_step(i, batchSize, j, num_kv_head, k, qSlice);
    }
  });

}


void flash_attention_kernel_impl(
    const torch::Tensor& output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& seq_lens,
    const at::Tensor& start_loc,
    c10::optional<double> scale) {
  auto q_head = query.size(2);
  auto kv_head = key.size(1);

  FASTMOE_DISPATCH_FLOATING_TYPES(query.scalar_type(), "cpu_flash_decode", [&] {
    if (q_head == kv_head) {
      cpu_flash_decode<scalar_t, 32, 1024>(
        output, query, key, value, seq_lens, start_loc, scale);
    } else {
      cpu_flash_decode_gqa<scalar_t, 32, 1024>(
        output, query, key, value, seq_lens, start_loc, scale);
    }
  });
}

// at::native::ALSO_REGISTER_AVX512_DISPATCH(flash_attention_kernel, &flash_attention_kernel_impl);

} // anonymous namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("token_attention_cpu", &flash_attention_kernel_impl, "Token Attention CPU");
}

