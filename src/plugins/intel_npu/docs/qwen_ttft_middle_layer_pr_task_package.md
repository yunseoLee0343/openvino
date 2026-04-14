[Problem Restatement]
- In the Intel x86_64 + OpenVINO Intel NPU stack, Qwen3-Next / Qwen 3.5 TTFT is dominated by a control-path chain: OpenVINO request preparation -> NPU plugin tensor/descriptor mutation -> Level Zero command list submission -> driver/firmware scheduling -> fence/event completion -> userspace wake -> token plumbing in LLM request code.
- This is not “just syscall cost” because latency accumulates across multiple boundary segments (userspace bookkeeping, queue object selection, command list close/execute, host wait, callback/token handling), each with its own tail behavior under contention and memory churn.
- This is not “just mapping detail” because address ownership and residency policy influence whether submit performs expensive per-request rebind/update (graph argument patching, memory import checks, command queue recreation) and whether completion waits for data that should have stayed resident.
- Likely bottleneck classes for Qwen first-token path:
  - **Submission Correctness**: over-frequent graph argument updates and queue/fence setup on prefill/generate transitions.
  - **Memory Policy**: KV/weights/state residency misses, transient control allocation churn, and remap/rebind pressure.
  - **Completion Latency**: fence/event wait + wakeup path and post-completion bookkeeping before first token is exposed.

[Key Translation from Original Theory]
- **SVC / vector entry -> OpenVINO submit chain (Submission Correctness)**
  - Practical equivalent: `LLMInferRequest::infer_*` and `ZeroInferRequest::infer_async` path entering `Pipeline::push()` and Level Zero `executeCommandList`, then later `Pipeline::pull()` for completion synchronization.
- **EL transition -> x86_64 boundary crossings (Completion Latency + Submission Correctness)**
  - Practical equivalent: userspace-to-driver submission boundary plus host wait wakeup boundary (`Fence::hostSynchronize` / `Event::hostSynchronize`), including scheduler wake/resume cost in WSL2 host/guest orchestration.
- **Page-table granule/depth -> host object -> device mapping topology (Memory Policy)**
  - Practical equivalent: tensor pointer -> `ZeroMem`/context allocation ID lookup/import path -> command list argument patching. Fragmentation or unstable allocation identity increases rebind frequency and mapping bookkeeping.
- **ioremap / early_ioremap -> control-plane mapping vs data-plane residency (Memory Policy)**
  - Practical equivalent: keep control descriptors/command metadata lifecycle distinct from large persistent tensors (weights/KV/state). Today, control and data updates can be interleaved via repeated `update_graph_arguments` and per-request setup logic.
- **acquire/release -> descriptor visibility contract (Submission Correctness)**
  - Practical equivalent: ensure host-side descriptor/tensor updates are completed before command list close/execute, and completion status reads happen after synchronization points (`hostSynchronize`) before token-visible actions.
- **interrupt completion -> completion fast path (Completion Latency)**
  - Practical equivalent: device completion -> fence/event signal -> plugin unblock -> LLM generate continuation and token materialization. Tail is shaped by lock scope and extra per-request post-processing before first token emission.

[OpenVINO-Relevant Code Search Targets]
- **submit wrapper path (Submission Correctness)**
  - `src/plugins/intel_npu/src/backend/src/zero_infer_request.cpp` (`infer_async`, `setup_pipeline`, `update_pipeline_if_memory_changed`).
- **infer request scheduling path (Completion Latency)**
  - `src/plugins/intel_npu/src/plugin/npuw/llm_infer_request.cpp` (prefill/generate orchestration and first-token sequencing).
- **command/descriptor build path (Submission Correctness)**
  - `src/plugins/intel_npu/src/backend/src/zero_pipeline.cpp` (command list construction, `appendGraphExecute`, `push`, `pull`, mutable argument updates).
- **control buffer allocation path (Memory Policy)**
  - `src/plugins/intel_npu/src/utils/src/zero/zero_mem_pool.cpp` and `zero_mem.cpp` (allocation/import/pool identity behavior).
- **completion callback path (Completion Latency)**
  - needs repo confirmation: async callback/wrapper layers around NPU request completion in plugin wrappers; start from `wrap_async_infer_request` callsites in `src/plugins/intel_npu/src/plugin/npuw/`.
- **event/fence wait path (Completion Latency)**
  - `src/plugins/intel_npu/src/backend/src/zero_pipeline.cpp` (`hostSynchronize` branches and reset path).
- **persistent weight/state allocation path (Memory Policy)**
  - `src/plugins/intel_npu/src/plugin/npuw/llm_infer_request.cpp` (KV cache sharing and variant reuse), `weights_bank.*`, and `zero_variable_state.cpp` (needs repo confirmation for full residency lifecycle).
- **driver boundary wrappers (Submission Correctness)**
  - needs repo confirmation: `src/plugins/intel_npu/src/compiler_adapter/src/ze_graph_ext_wrappers.cpp` and Level Zero wrappers for command submission semantics on MCDM stacks.

[Concrete Runtime Policies to Implement]
- **Policy 1: Submit-stage canonicalization cache for graph argument updates**
  - Class: **Submission Correctness**
  - Idea: cache last-bound tensor identity/shape/stride hash per argument index and skip `update_graph_arguments` when unchanged.
  - Why TTFT/correctness: reduces redundant descriptor patching before first prefill/generate submit; lowers control-path variance while preserving correctness via hash equality + fallback update.
  - Likely touchpoints: `zero_infer_request.cpp` + `zero_pipeline.cpp` mutable command list update calls.

- **Policy 2: Control-plane buffer/page pool split from large data buffers**
  - Class: **Memory Policy**
  - Idea: create a lightweight pool class (or tagged path) for short-lived control metadata and keep KV/weights/state in persistent allocations with minimal rebinding.
  - Why TTFT/correctness: avoids fragmented transient allocations interfering with persistent residency, reducing remap/rebind churn for first-token-critical turns.
  - Likely touchpoints: `zero_mem_pool.cpp`, `zero_tensor.cpp`, infer-request allocation paths.

- **Policy 3: Request-level boundary contraction for prefill/generate handoff**
  - Class: **Submission Correctness**
  - Idea: for Qwen path, precompute/canonicalize generate descriptor packets once per conversation and reuse across token steps unless shape/variant changes.
  - Why TTFT/correctness: fewer userspace->driver-equivalent submit preparations per step; lower p95/p99 submit latency.
  - Likely touchpoints: `llm_infer_request.cpp` (prepare/infer blocks), `zero_infer_request.cpp` pipeline update conditions.

- **Policy 4: First-token completion fast lane**
  - Class: **Completion Latency**
  - Idea: in first-token-critical phase, minimize post-`hostSynchronize` work before exposing token-ready outputs (defer non-critical stats/bookkeeping).
  - Why TTFT/correctness: shortens completion-to-visible-token interval without changing numerical path.
  - Likely touchpoints: `llm_infer_request.cpp` around prefill/generate boundaries and LM head handoff.

- **Policy 5: Queue/fence stability policy**
  - Class: **Completion Latency**
  - Idea: avoid command queue/fence churn when queue descriptor key is stable; explicitly track and report queue-version flips as anomaly.
  - Why TTFT/correctness: reduces tail jitter from reinitialization in `push()` and clarifies when driver/runtime renegotiation occurs.
  - Likely touchpoints: `zero_pipeline.cpp` `command_queue_version_changed` branch.

[Instrumentation Plan]
- Add request-scoped trace struct (enabled by plugin property/env) with monotonic timestamps and counters; aggregate histogram percentiles offline and optionally dump in debug logs.
- Required metrics:
  - `ttft_ms` (first token visible - user submit entry).
  - `submit_latency_ms` p50/p95/p99.
  - `completion_latency_ms` p50/p95/p99.
  - `boundary_crossings_per_request` (submit calls + waits + callback resumptions).
  - `control_translation_count` (argument canonicalization/validation/update operations).
  - `buffer_rebind_count` and `buffer_remap_count`.
  - `persistent_residency_hit` / `persistent_residency_miss` for KV/weights/state.
  - `submit_lock_hold_us` / `completion_lock_hold_us` for key mutexes.
- Stage breakdown (explicit probes):
  - `t0_userspace_submit_entry`
  - `t1_pre_submit_canonicalization_done`
  - `t2_descriptor_ready`
  - `t3_driver_handoff_called` (execute command list call boundary)
  - `t4_notify_or_doorbell_equivalent_done` (same call completion in user thread)
  - `t5_device_completion_visible` (`hostSynchronize` return)
  - `t6_wake_event_processed` (if async callback/event queue exists; needs repo confirmation)
  - `t7_userspace_callback_enter`
  - `t8_first_token_visible`
- Concrete insertion points:
  - `zero_pipeline.cpp`: around command list close/execute and `hostSynchronize`.
  - `zero_infer_request.cpp`: before/after `setup_pipeline`, `update_pipeline_if_memory_changed`, `push`, `pull`.
  - `llm_infer_request.cpp`: around prefill infer, first generate infer, LM head path, first-token publication.
  - `zero_mem_pool.cpp`: counters for import/allocate/reuse paths.

[Minimal Patch Candidates]
- **Candidate A (most mergeable): Add cross-layer instrumentation only**
  - Change summary: add opt-in per-request timing/counter probes and debug dump for submit/completion/TTFT stages.
  - Why small: no behavior change; additive diagnostics in existing paths.
  - Expected win: immediate p95/p99 visibility; enables evidence-driven follow-up patches.
  - Regression risk: minor overhead when enabled; negligible when disabled.
  - Proof benchmark: Qwen TTFT benchmark with instrumentation off/on overhead check (<1-2% when enabled, ~0 when disabled).

- **Candidate B: Skip redundant `update_graph_arguments` via canonicalization hash**
  - Change summary: memoize per-arg descriptor signature and bypass no-op mutable command list updates.
  - Why small: localized to update call sites and request state cache.
  - Expected measurable win: lower submit p95/p99 and reduced control translation count.
  - Regression risk: stale binding if signature misses a meaningful field; mitigate with strict hash input and debug asserts.
  - Proof benchmark: reduced `control_translation_count` and submit tail under repeated token steps.

- **Candidate C: Queue/fence churn guard + counterization**
  - Change summary: track queue descriptor changes and avoid unnecessary fence re-creation; expose churn metric.
  - Why small: isolated in `Pipeline::push` and fence initialization logic.
  - Expected measurable win: tighter completion tail where queue descriptor is stable.
  - Regression risk: incorrect reuse across incompatible queue desc; guarded by existing key equality.
  - Proof benchmark: decreased `queue_reinit_count` and improved completion p95/p99.

- **Candidate D (more ambitious but still bounded): First-token completion fast lane**
  - Change summary: reorder/defer non-critical post-completion bookkeeping for first token path.
  - Why still PR-sized: limited to LLM request first-token branch, no kernel/driver changes.
  - Expected measurable win: TTFT reduction, especially p95/p99.
  - Regression risk: profiling/report ordering changes; ensure correctness tests and token equivalence.
  - Proof benchmark: TTFT distribution shift with unchanged token outputs.

[Benchmark Design]
- Model/workload focus: Qwen3-Next / Qwen 3.5 style decode flow with prefill + iterative generate on Intel NPU plugin.
- Run matrix:
  - Cold start: first process run after model load (captures first-request residency/setup cost).
  - Warm steady-state: discard first N requests, then measure long run.
  - Prompt classes:
    - short prompt (TTFT-sensitive, small prefill).
    - medium prompt (prefill+generate balance).
    - first-token-stress prompt (forces heavier control metadata updates, e.g., dynamic attention mask/position changes).
  - Concurrency: 1, 2, 4 concurrent sessions (to expose queue/lock contention tails).
- Metrics collection method:
  - Plugin instrumentation logs -> structured CSV/JSON extraction per request.
  - Compute p50/p95/p99 for TTFT, submit stage total, completion stage total, and key sub-stage deltas.
  - Record counts: boundary crossings, translation count, rebind/remap, residency hit/miss, lock hold.
- Success criteria:
  - Primary: TTFT p95 and p99 improve (target directionally >=5% on warm path for optimization patches).
  - Secondary: submit/completion p95/p99 reduce with no correctness regressions.
  - Guardrail: mean latency may move less; tail improvement is required.
  - Distinguish benefits:
    - First-request benefit: cold TTFT/residency effects.
    - Steady-state benefit: warm p95/p99 submit/completion and token cadence.

[PR Claim]
- Added opt-in NPU request timeline instrumentation covering submit, device wait, callback, and first-token visibility stages.
- Added request-level counters for boundary crossings, descriptor translation churn, and buffer rebind/remap activity.
- Added residency hit/miss reporting for persistent KV/weights/state paths to expose avoidable remapping.
- Reduced redundant command-list argument updates on unchanged tensor bindings in iterative LLM generation (if Candidate B is included).
- Improved TTFT tail observability and provided benchmark scripts/config guidance focused on p95/p99 first-token latency.

[What to Deprioritize]
- Deep ARM exception-level theory and historical kernel entry discussions.
- Driver-internal redesign beyond plugin-visible submission/wait boundaries.
- Large architectural rewrites across all devices/backends before evidence from NPU-targeted instrumentation.
- Security/isolation deep-dives that do not change TTFT/submit/completion path now.
- Giant memory manager rewrites; prioritize policy toggles and counters first.

[Immediate Next Actions]
- Trace the `zero_infer_request` -> `pipeline.push/pull` path and count boundary crossings per request.
- Add timestamp probes for canonicalization, descriptor-ready, execute-call, synchronize-return, callback-enter, first-token-visible.
- Add counters for `update_graph_arguments` attempts vs effective updates.
- Add memory residency counters in `zero_mem_pool` import/reuse/allocation paths.
- Add lock hold-time probes around submit/completion critical mutex regions.
- Implement no-op update elision for unchanged argument bindings behind a debug property toggle.
- Run Qwen TTFT benchmark matrix (cold/warm, short/medium/stress prompts, concurrency 1/2/4) and export p95/p99.
- Prepare PR with evidence table: baseline vs patch metrics, plus correctness parity notes.
