# ROCm Tensor Split TurboQuant KV Work Log

## 2026-05-28T00:25:39+01:00

- hypothesis: The target command is blocked by the documented high-level tensor-split plus quantized-KV guard before reaching HIP/RCCL, SET_ROWS, or Flash Attention.
- files inspected: `src/llama.cpp`, `src/llama-context.cpp`, `src/llama-graph.cpp`, `src/llama-kv-cache.cpp`, `ggml/src/ggml-backend-meta.cpp`, `ggml/src/ggml-hip/CMakeLists.txt`, `ggml/src/ggml-cuda/fattn.cu`, `ggml/src/ggml-cuda/fattn-vec.cuh`, `ggml/src/ggml-cuda/set-rows.cu`, `docs/multi-gpu.md`.
- patch summary: none; reproduced current behavior before editing.
- build command: existing `build-hip-rccl` binary from the branch was used.
- server command:
  ```sh
  GGML_CUDA_ALLREDUCE=nccl HIP_VISIBLE_DEVICES=0,1 ./build-hip-rccl/bin/llama-server \
    -hf unsloth/Qwen3.6-27B-MTP-GGUF:Q4_K_M \
    -ngl 99 \
    -c 32768 \
    -fa on \
    -ctk turbo4 \
    -ctv turbo4 \
    --spec-type draft-mtp \
    --spec-draft-n-max 2 \
    -sm tensor \
    -ts 1,1 \
    -np 1 \
    --no-mmproj \
    --port 8081
  ```
- benchmark curl command:
  ```sh
  curl http://localhost:8081/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "local-model",
      "messages": [
        {
          "role": "user",
          "content": "Write a commandline app that takes a dice type, d6 and outputs a roll, support multiple dice"
        }
      ],
      "max_tokens": 1000,
      "temperature": 0
    }' | jq '{
      content: .choices[0].message.content,
      usage: {
        prompt_tokens: .usage.prompt_tokens,
        completion_tokens: .usage.completion_tokens,
        total_tokens: .usage.total_tokens
      },
      performance: {
        prompt_tps: .timings.prompt_per_second,
        eval_tps: .timings.predicted_per_second,
        eval_ms: .timings.predicted_ms,
        draft_accepted_pct: (
          if (.timings.draft_n // 0) > 0
          then ((.timings.draft_n_accepted // 0) / .timings.draft_n * 100 | round)
          else 0
          end
        )
      }
    }'
  ```
- exact error or result: ROCm detected both RX 7800 XT devices. Model loaded, then context creation failed with `llama_init_from_model: simultaneous use of SPLIT_MODE_TENSOR and KV cache quantization not implemented`.
- next hypothesis: This first blocker is a conservative guard. Relax it only for TurboQuant KV cache types and let graph reservation/backend support expose the next real blocker.

## 2026-05-28T00:27:12+01:00

- hypothesis: After relaxing the high-level guard, the existing HIP TurboQuant KV update and Flash Attention paths may either run or expose the next tensor-split meta-backend rule that is missing.
- files inspected: `src/llama-context.cpp`, `ggml/src/ggml-backend-meta.cpp`, `ggml/src/ggml.c`, `ggml/src/ggml-cuda/turbo-wht.cu`.
- patch summary: `src/llama-context.cpp` now permits TurboQuant KV cache types (`turbo2`, `turbo3`, `turbo4`) with `SPLIT_MODE_TENSOR` while continuing to reject other quantized KV types.
- build command:
  ```sh
  cmake --build build-hip-rccl -j
  ```
- server command:
  ```sh
  GGML_CUDA_ALLREDUCE=nccl HIP_VISIBLE_DEVICES=0,1 ./build-hip-rccl/bin/llama-server \
    -hf unsloth/Qwen3.6-27B-MTP-GGUF:Q4_K_M \
    -ngl 99 \
    -c 32768 \
    -fa on \
    -ctk turbo4 \
    -ctv turbo4 \
    --spec-type draft-mtp \
    --spec-draft-n-max 2 \
    -sm tensor \
    -ts 1,1 \
    -np 1 \
    --no-mmproj \
    --port 8081
  ```
- benchmark curl command: not run; server aborted during warmup before accepting a request.
- exact error or result: Context creation, TurboQuant KV allocation, recurrent-state allocation, and scheduler reserve succeeded. Warmup aborted in the meta backend with `/home/tdeburca/git/model-learning/mtp-turboquant/ggml/src/ggml-backend-meta.cpp:981: ggml op not implemented: TURBO_WHT`.
- next hypothesis: `GGML_OP_TURBO_WHT` is an F32 shape-preserving transform over `src0`; tensor split can propagate the split state from `src0` as long as the WHT dimension is not split and the optional scale tensor is mirrored.

## 2026-05-28T00:36:00+01:00

- hypothesis: Adding a tensor-split meta rule for `GGML_OP_TURBO_WHT` should allow TurboQuant KV cache warmup and inference to run.
- files inspected: `ggml/src/ggml-backend-meta.cpp`, `ggml/src/ggml-cuda/fattn.cu`, `ggml/src/ggml-cuda/fattn-vec.cuh`, `ggml/src/ggml-hip/CMakeLists.txt`.
- patch summary: `ggml/src/ggml-backend-meta.cpp` now propagates `TURBO_WHT` split state from `src0`, rejecting split-axis-0 and non-mirrored scale tensors.
- build command:
  ```sh
  cmake --build build-hip-rccl -j
  ```
- server command:
  ```sh
  GGML_CUDA_ALLREDUCE=nccl HIP_VISIBLE_DEVICES=0,1 ./build-hip-rccl/bin/llama-server \
    -hf unsloth/Qwen3.6-27B-MTP-GGUF:Q4_K_M \
    -ngl 99 -c 32768 -fa on -ctk turbo4 -ctv turbo4 \
    --spec-type draft-mtp --spec-draft-n-max 2 \
    -sm tensor -ts 1,1 -np 1 --no-mmproj --port 8081
  ```
- benchmark curl command: exact requested request against `http://localhost:8081/v1/chat/completions`.
- exact error or result: server started successfully with target and MTP draft contexts using `turbo4/turbo4`. Two benchmark requests completed without crash. Results: run 1 prompt 31, completion 1000, prompt_tps 67.19, eval_tps 51.13, eval_ms 19557.06, draft acceptance 81%; run 2 prompt 31, completion 1000, prompt_tps 105.09, eval_tps 50.69, eval_ms 19727.66, draft acceptance 81%. `.choices[0].message.content` was empty, but raw response contained coherent dice-CLI planning text in `reasoning_content`. The f16/f16 tensor+RCCL baseline showed the same empty `content` behavior with this exact request: prompt 31, completion 1000, prompt_tps 78.22, eval_tps 51.81, eval_ms 19303.03, draft acceptance 81%.
- next hypothesis: The target is functionally enabled but does not beat the f16 baseline on this workload. Mixed `turbo4/f16` may show whether K compression alone helps, but it currently lacks Flash Attention VEC template coverage for `turbo4 K + f16 V`.

## 2026-05-28T00:42:00+01:00

- hypothesis: `-ctk turbo4 -ctv f16` should use the same TurboQuant K path plus normal V, but the Flash Attention VEC dispatch table may be missing the asymmetric template.
- files inspected: `ggml/src/ggml-cuda/fattn.cu`, `ggml/src/ggml-cuda/fattn-vec.cuh`, `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-turbo4_0.cu`, `ggml/src/ggml-cuda/CMakeLists.txt`, `ggml/src/ggml-hip/CMakeLists.txt`.
- patch summary: pending.
- build command:
  ```sh
  cmake --build build-hip-rccl -j
  ```
- server command:
  ```sh
  GGML_CUDA_ALLREDUCE=nccl HIP_VISIBLE_DEVICES=0,1 ./build-hip-rccl/bin/llama-server \
    -hf unsloth/Qwen3.6-27B-MTP-GGUF:Q4_K_M \
    -ngl 99 -c 32768 -fa on -ctk turbo4 -ctv f16 \
    --spec-type draft-mtp --spec-draft-n-max 2 \
    -sm tensor -ts 1,1 -np 1 --no-mmproj --port 8081
  ```
- benchmark curl command: not run; server aborted during warmup.
- exact error or result: warmup aborted at `ggml/src/ggml-cuda/fattn.cu:380: fatal error`, the fallback after all explicit VEC dispatch cases. The table has `f16/turbo4` but not `turbo4/f16`.
- next hypothesis: Add `turbo4_0-f16` VEC declarations, dispatch, and template instance to both CUDA and HIP CMake lists, then retest the mixed combination.

## 2026-05-28T00:52:00+01:00

- hypothesis: With `turbo4/f16` template coverage added, all TurboQuant tensor-split variants needed for the target and mixed comparisons should start and complete the fixed benchmark request.
- files inspected: `ggml/src/ggml-cuda/fattn.cu`, `ggml/src/ggml-cuda/fattn-vec.cuh`, `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo4_0-f16.cu`, `ggml/src/ggml-cuda/CMakeLists.txt`, `ggml/src/ggml-hip/CMakeLists.txt`.
- patch summary: added Flash Attention VEC coverage for `GGML_TYPE_TURBO4_0` K with `GGML_TYPE_F16` V and wired the new template instance into CUDA and HIP builds.
- build command:
  ```sh
  cmake --build build-hip-rccl -j
  ```
- server command:
  ```sh
  GGML_CUDA_ALLREDUCE=nccl HIP_VISIBLE_DEVICES=0,1 ./build-hip-rccl/bin/llama-server \
    -hf unsloth/Qwen3.6-27B-MTP-GGUF:Q4_K_M \
    -ngl 99 -c 32768 -fa on -ctk turbo4 -ctv f16 \
    --spec-type draft-mtp --spec-draft-n-max 2 \
    -sm tensor -ts 1,1 -np 1 --no-mmproj --port 8081
  ```
- benchmark curl command: exact requested request against `http://localhost:8081/v1/chat/completions`.
- exact error or result: `turbo4/f16` started and completed 1000 tokens without crash: prompt 31, completion 1000, prompt_tps 94.87, eval_tps 51.11, eval_ms 19567.08, draft acceptance 82%. `f16/turbo4` also completed: prompt 31, completion 1000, prompt_tps 94.25, eval_tps 50.72, eval_ms 19714.51, draft acceptance 80%. `turbo3/turbo3` completed: prompt 31, completion 1000, prompt_tps 89.74, eval_tps 50.34, eval_ms 19863.55, draft acceptance 81%.
- next hypothesis: The primary remaining limitation is performance, not basic HIP/RCCL correctness. TurboQuant KV saves memory under tensor split but the WHT/dequant path currently offsets that benefit for this short-context decode benchmark.

## Benchmark Table

| KV / split mode | command summary | prompt tokens | completion tokens | prompt_tps | eval_tps | eval_ms | draft_accepted_pct | correctness/stability notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| tensor + RCCL + f16/f16 + draft n=2 | `GGML_CUDA_ALLREDUCE=nccl ... -ctk f16 -ctv f16 -sm tensor -ts 1,1` | 31 | 1000 | 78.22 | 51.81 | 19303.03 | 81 | completed without crash; `content` empty but coherent dice-CLI planning text present in `reasoning_content` |
| row + turbo4/turbo4 + draft n=2 | prior local baseline: `HIP_VISIBLE_DEVICES=0,1 ./build/bin/llama-server ... -ctk turbo4 -ctv turbo4 -sm row -ts 1,1` | not recorded | 1000 | not recorded | 40.5 | not recorded | not recorded | prior user-provided result |
| tensor + RCCL + turbo4/turbo4 + draft n=2 | `GGML_CUDA_ALLREDUCE=nccl ... -ctk turbo4 -ctv turbo4 -sm tensor -ts 1,1` | 31 | 1000 | 67.19 | 51.13 | 19557.06 | 81 | target command completed without crash; `content` empty but coherent dice-CLI planning text present in `reasoning_content`; repeat eval_tps 50.69 |
| tensor + RCCL + turbo4/f16 + draft n=2 | `GGML_CUDA_ALLREDUCE=nccl ... -ctk turbo4 -ctv f16 -sm tensor -ts 1,1` | 31 | 1000 | 94.87 | 51.11 | 19567.08 | 82 | completed without crash after adding `turbo4_0-f16` FA VEC instance; same reasoning/content behavior |
| tensor + RCCL + f16/turbo4 + draft n=2 | `GGML_CUDA_ALLREDUCE=nccl ... -ctk f16 -ctv turbo4 -sm tensor -ts 1,1` | 31 | 1000 | 94.25 | 50.72 | 19714.51 | 80 | completed without crash; same reasoning/content behavior |
| tensor + RCCL + turbo3/turbo3 + draft n=2 | `GGML_CUDA_ALLREDUCE=nccl ... -ctk turbo3 -ctv turbo3 -sm tensor -ts 1,1` | 31 | 1000 | 89.74 | 50.34 | 19863.55 | 81 | completed without crash; same reasoning/content behavior |
| tensor + RCCL + q8_0/q8_0 + draft n=2 | `GGML_CUDA_ALLREDUCE=nccl ... -ctk q8_0 -ctv q8_0 -sm tensor -ts 1,1` | not run | not run | not run | not run | not run | not run | still blocked by the deliberately narrow guard because this patch only enables TurboQuant cache types under tensor split |

## 2026-05-28T00:55:13+01:00

- hypothesis: Focused backend tests can cover the two regressions fixed by this patch without requiring the full Qwen MTP benchmark: meta split-state propagation for `TURBO_WHT`, and Flash Attention dispatch support for `turbo4` K with `f16` V.
- files inspected: `tests/CMakeLists.txt`, `tests/test-backend-ops.cpp`, `tests/test-backend-meta.cpp`, `ggml/src/ggml-backend-meta.cpp`, `ggml/src/ggml-cuda/fattn.cu`.
- patch summary: added `test-backend-meta.cpp` with a minimal two-segment meta-device graph for `ggml_turbo_wht`; registered it in CMake; added a `FLASH_ATTN_EXT` eval case for `GGML_TYPE_TURBO4_0` K and `GGML_TYPE_F16` V.
- build command:
  ```sh
  cmake -B build-hip-rccl \
    -DGGML_HIP=ON \
    -DGGML_HIP_RCCL=ON \
    -DAMDGPU_TARGETS=gfx1101 \
    -DROCM_PATH=/opt/rocm-7.2.3 \
    -DCMAKE_BUILD_TYPE=Release

  cmake --build build-hip-rccl --target test-backend-meta test-backend-ops -j
  ```
- server command: not run for this test-only patch.
- benchmark curl command: not run for this test-only patch.
- exact error or result: `./build-hip-rccl/bin/test-backend-meta` passed. `./build-hip-rccl/bin/test-backend-ops support -o FLASH_ATTN_EXT -b ROCm0 -p 'type_K=turbo4,type_V=f16'` reported the new case supported. `./build-hip-rccl/bin/test-backend-ops test -o FLASH_ATTN_EXT -b ROCm0 -p 'type_K=turbo4,type_V=f16'` passed with `1/1 tests passed` and `3/3 backends passed`. Broad `ctest --test-dir build-hip-rccl -R '^test-backend-meta$|^test-backend-ops$' --output-on-failure` passed `test-backend-meta` but aborted in the full `test-backend-ops` sweep on an unrelated ROCm `GET_ROWS` unsupported type `tq3_1s`.
- next hypothesis: These focused tests are the right coverage for this first pass. A future broader test should add a scheduler-level tensor-split KV cache graph once the project has a small backend-independent fixture for split-mode KV cache updates.
