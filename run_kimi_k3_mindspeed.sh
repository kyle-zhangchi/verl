#!/usr/bin/env bash
set -euo pipefail

# PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="/mnt/share/z00931161/kimi-k3/fsdpturbo-backend/"
run_mode="${KIMI_RUN_MODE:-single}"
start_time="${VERL_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
project_name="${VERL_PROJECT_NAME:-GRPO-Kimi-K3-A3B}"

case "${run_mode}" in
    single)
        nnodes=1
        npus_per_node=8
        visible_devices_default=0,1,2,3,4,5,6,7
        turbo_ep_size=8
        turbo_efsdp_size=1
        turbo_fsdp_size=8
        rollout_tp_size=8
        rollout_ep_size=8
        max_response_length="${VERL_MAX_RESPONSE_LENGTH:-512}"
        rollout_max_num_seqs="${VERL_VLLM_MAX_NUM_SEQS:-32}"
        rollout_max_num_batched_tokens="${VERL_VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
        train_batch_size=16
        rollout_n=4
        ppo_mini_batch_size=16
        actor_moe_chunk_tokens="${VERL_ACTOR_MOE_CHUNK_TOKENS:-512}"
        # model_path="${MODEL_PATH:-/mnt/share/z00804445/verl_fsdp_turbo/kimi_k3/model/K3-top16-pruned_bf16_countbench_128_n4_12layers-W4A8}"
        model_path="${MODEL_PATH:-/mnt/share/z00931161/weights/K3-top16-pruned_bf16_countbench_128_n4_12layers}"
        exp_name="${VERL_EXPERIMENT_NAME:-kimi_k3_1node_${start_time}}"
        total_training_steps="${VERL_TOTAL_TRAINING_STEPS:-10}"
        save_freq_default=-1
        profiler_log_level_default=WARN
        ;;
    4nodes)
        nnodes=4
        npus_per_node=8
        visible_devices_default=0,1,2,3,4,5,6,7
        turbo_ep_size=8
        turbo_efsdp_size=4
        turbo_fsdp_size=32
        rollout_tp_size=16
        rollout_ep_size=16
        max_response_length="${VERL_MAX_RESPONSE_LENGTH:-512}"
        rollout_max_num_seqs="${VERL_VLLM_MAX_NUM_SEQS:-32}"
        rollout_max_num_batched_tokens="${VERL_VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
        train_batch_size=16
        rollout_n=4
        ppo_mini_batch_size=16
        actor_moe_chunk_tokens="${VERL_ACTOR_MOE_CHUNK_TOKENS:-512}"
        model_path="${MODEL_PATH:-/mnt/share/w00803372/weights/K3-top16-pruned_bf16_countbench_128_n4}"
        exp_name="${VERL_EXPERIMENT_NAME:-kimi_k3_4node_${start_time}}"
        total_training_steps="${VERL_TOTAL_TRAINING_STEPS:-3}"
        save_freq_default=-1
        profiler_log_level_default=INFO
        ;;
    *)
        echo "Unsupported KIMI_RUN_MODE: ${run_mode}" >&2
        exit 2
        ;;
esac

save_freq="${VERL_SAVE_FREQ:-${save_freq_default}}"
rollout_enforce_eager="${VERL_VLLM_ENFORCE_EAGER:-False}"
max_actor_ckpt_to_keep="${VERL_MAX_ACTOR_CKPT_TO_KEEP:-2}"
resume_mode="${VERL_RESUME_MODE:-disable}"
resume_from_path="${VERL_RESUME_FROM_PATH:-}"
train_file="${TRAIN_FILE:-/mnt/share/w00848461/datasets/countbenchqa_lite/train_448.parquet}"
test_file="${TEST_FILE:-/mnt/share/w00848461/datasets/countbenchqa_lite/train.parquet}"

# verl_path="${PROJECT_ROOT}/verl"
# fsdp_turbo_path="${PROJECT_ROOT}/FSDPTurbo"
fsdp_turbo_path="/mnt/share/z00804445/verl_fsdp_turbo/kimi_k3/FSDPTurbo"
vllm_path="${PROJECT_ROOT}/vllm"
# vllm_path="/mnt/share/c00833002/projects/low-precision/kimi-k3/repos/vllm"
vllm_ascend_path="${PROJECT_ROOT}/vllm-ascend"
# vllm_ascend_path="/mnt/share/c00833002/projects/low-precision/kimi-k3/repos/vllm-ascend"
runtime_site="${PROJECT_ROOT}/.runtime-site"
# cann_python_site=/home/pkg/b131/cann-9.1.0/python/site-packages
# cann_python_site=/home/pkg/MAIN_0727/cann-9.2.0/python/site-packages
cann_python_site=/mnt/share/j00949742/conda_envs/cann/9.2.0b030/cann-9.2.0/python/site-packages

# The checkpoint's Python files are executable remote code. Keep actor and
# rollout on the reviewed FSDP-Turbo implementation for every launch.
cp -f "${fsdp_turbo_path}/fsdp_turbo/models/kimi/"*.py "${model_path}/" # 这个为啥要移动

export PYTHONPATH="${runtime_site}:${vllm_path}:${vllm_ascend_path}:${fsdp_turbo_path}:${fsdp_turbo_path}/examples:${cann_python_site}"
export FSDP_TURBO_ROOT="${fsdp_turbo_path}"
export VLLM_ASCEND_PATH="${vllm_ascend_path}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${visible_devices_default}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NON_MEGATRON="${NON_MEGATRON:-true}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV="${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-${profiler_log_level_default}}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-${profiler_log_level_default}}"
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-0}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING="${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-true}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ASCEND_ENABLE_FLASHCOMM="${VLLM_ASCEND_ENABLE_FLASHCOMM:-1}"
export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export MULTI_STREAM_MEMORY_REUSE="${MULTI_STREAM_MEMORY_REUSE:-1}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-data0.3001}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-data0.3001}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-auto}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-50000}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-17340}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_ASYNC_ERROR_HANDLING="${HCCL_ASYNC_ERROR_HANDLING:-0}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-256}"
export P2P_HCCL_BUFFSIZE="${P2P_HCCL_BUFFSIZE:-20}"
export VLLM_VERSION="${VLLM_VERSION:-0.26.0}"
export VERL_FSDP_TURBO_VISION_ATTN_IMPLEMENTATION="${VERL_FSDP_TURBO_VISION_ATTN_IMPLEMENTATION:-flash_attention_2}"
export KIMI_TRAIN_USE_PARTITION_INVARIANT_KDA_FORWARD="${KIMI_TRAIN_USE_PARTITION_INVARIANT_KDA_FORWARD:-1}"
export VERL_KIMI_MLP_CHUNK_TOKENS="${VERL_KIMI_MLP_CHUNK_TOKENS:-128}"
export VERL_FAIL_ON_NONFINITE_GRAD="${VERL_FAIL_ON_NONFINITE_GRAD:-1}"

local_ray_owned=0
cleanup_local_ray() {
    if (( local_ray_owned == 1 )); then
        echo "Stopping local Ray runtime"
        ray stop --force >/dev/null 2>&1 || true
    fi
}
trap cleanup_local_ray EXIT INT TERM

if [[ "${run_mode}" == single ]]; then
    ray stop --force >/dev/null 2>&1 || true
    node_suffix="$(awk -F- '{print $NF}' /proc/sys/kernel/hostname)"
    node_ip="${NODE_IP:-172.27.13.95}"
    ray start \
        --head \
        --port="${RAY_PORT:-6792}" \
        --node-ip-address="${node_ip}" \
        --dashboard-host="${node_ip}" \
        --dashboard-port="${RAY_DASHBOARD_PORT:-8262}" \
        --num-cpus="${RAY_NUM_CPUS:-64}" \
        --object-store-memory="${RAY_OBJECT_STORE_MEMORY:-68719476736}" \
        --resources="{\"NPU\": ${npus_per_node}}"
    local_ray_owned=1
fi

# rollout_data_dir="${VERL_ROLLOUT_DATA_DIR:-${PROJECT_ROOT}/rollout_dump/${start_time}}"
training_log_dir="/mnt/share/z00804445/verl_fsdp_turbo/kimi_k3/verl/logs_20261019"
export training_log_dir_curr=${training_log_dir}/$(date +"%Y%m%d_%H%M")
rollout_data_dir="${training_log_dir_curr}/rollout/"
mkdir -p "${training_log_dir_curr}" "${rollout_data_dir}"
training_log="${training_log_dir_curr}/${exp_name}.log"

max_prompt_length=1024
rollout_multimodal_token_margin="${VERL_VLLM_MULTIMODAL_TOKEN_MARGIN:-16384}"
rollout_max_model_len="${VERL_VLLM_MAX_MODEL_LEN:-$((max_prompt_length + max_response_length + rollout_multimodal_token_margin))}"
ppo_max_token_len=$(((max_prompt_length + max_response_length) / 2))

RAY_CONFIG=(
    "++ray_kwargs.ray_init.runtime_env.env_vars.ASCEND_RT_VISIBLE_DEVICES=\"${ASCEND_RT_VISIBLE_DEVICES}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=\"${PYTHONPATH}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.FSDP_TURBO_ROOT=\"${FSDP_TURBO_ROOT}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_PATH=\"${VLLM_ASCEND_PATH}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.OMP_NUM_THREADS=\"${OMP_NUM_THREADS}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.NON_MEGATRON=\"${NON_MEGATRON}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONHASHSEED=\"${PYTHONHASHSEED}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.TOKENIZERS_PARALLELISM=\"${TOKENIZERS_PARALLELISM}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL=\"${VLLM_LOGGING_LEVEL}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL=\"${VERL_LOGGING_LEVEL}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_DISABLE_COMPILE_CACHE=\"${VLLM_DISABLE_COMPILE_CACHE}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_BATCH_INVARIANT=\"${VLLM_BATCH_INVARIANT}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ALLOW_RUNTIME_LORA_UPDATING=\"${VLLM_ALLOW_RUNTIME_LORA_UPDATING}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1=\"${VLLM_USE_V1}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_ENABLE_FLASHCOMM=\"${VLLM_ASCEND_ENABLE_FLASHCOMM}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_ENABLE_NZ=\"${VLLM_ASCEND_ENABLE_NZ}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS=\"${CUDA_DEVICE_MAX_CONNECTIONS}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.MULTI_STREAM_MEMORY_REUSE=\"${MULTI_STREAM_MEMORY_REUSE}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.CPU_AFFINITY_CONF=\"${CPU_AFFINITY_CONF}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.HCCL_OP_EXPANSION_MODE=\"${HCCL_OP_EXPANSION_MODE}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.HCCL_SOCKET_IFNAME=\"${HCCL_SOCKET_IFNAME}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.GLOO_SOCKET_IFNAME=\"${GLOO_SOCKET_IFNAME}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.HCCL_HOST_SOCKET_PORT_RANGE=\"${HCCL_HOST_SOCKET_PORT_RANGE}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.HCCL_NPU_SOCKET_PORT_RANGE=\"${HCCL_NPU_SOCKET_PORT_RANGE}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.HCCL_IF_BASE_PORT=\"${HCCL_IF_BASE_PORT}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.HCCL_EXEC_TIMEOUT=\"${HCCL_EXEC_TIMEOUT}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.HCCL_CONNECT_TIMEOUT=\"${HCCL_CONNECT_TIMEOUT}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.HCCL_ASYNC_ERROR_HANDLING=\"${HCCL_ASYNC_ERROR_HANDLING}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.HCCL_BUFFSIZE=\"${HCCL_BUFFSIZE}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.P2P_HCCL_BUFFSIZE=\"${P2P_HCCL_BUFFSIZE}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_VERSION=\"${VLLM_VERSION}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FSDP_TURBO_VISION_ATTN_IMPLEMENTATION=\"${VERL_FSDP_TURBO_VISION_ATTN_IMPLEMENTATION}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.KIMI_TRAIN_USE_PARTITION_INVARIANT_KDA_FORWARD=\"${KIMI_TRAIN_USE_PARTITION_INVARIANT_KDA_FORWARD}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_KIMI_MLP_CHUNK_TOKENS=\"${VERL_KIMI_MLP_CHUNK_TOKENS}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FAIL_ON_NONFINITE_GRAD=\"${VERL_FAIL_ON_NONFINITE_GRAD}\""
)

enable_profiling="${VERL_ENABLE_PROFILING:-0}"
profile_step="${VERL_PROFILE_STEP:-3}"
PROFILER_CONFIG=(
    global_profiler.tool=null
    'global_profiler.steps=[]'
    actor_rollout_ref.actor.profiler.enable=False
    actor_rollout_ref.ref.profiler.enable=False
    actor_rollout_ref.rollout.profiler.enable=False
)
if [[ "${run_mode}" == 4nodes && "${enable_profiling}" == 1 ]]; then
    if ! [[ "${profile_step}" =~ ^[1-9][0-9]*$ ]]; then
        echo "VERL_PROFILE_STEP must be a positive integer, got: ${profile_step}" >&2
        exit 2
    fi
    if (( profile_step > total_training_steps )); then
        echo "VERL_PROFILE_STEP=${profile_step} exceeds total steps=${total_training_steps}" >&2
        exit 2
    fi

    # Profile the requested global training step on every training rank and rollout
    # replica. Keep parsing offline to avoid blocking the training processes.
    profile_save_path="${VERL_PROFILE_SAVE_PATH:-${PROJECT_ROOT}/profiles/${exp_name}_step${profile_step}_all_ranks}"
    mkdir -p "${profile_save_path}"
    for replica_rank in 0 1 2 3; do
        mkdir -p "${profile_save_path}/agent_loop_rollout_replica_${replica_rank}"
    done
    echo "Kimi profiling output: ${profile_save_path}"

    PROFILER_CONFIG=(
        global_profiler.tool=npu
        "global_profiler.steps=[${profile_step}]"
        global_profiler.profile_continuous_steps=False
        global_profiler.save_path="${profile_save_path}"

        actor_rollout_ref.actor.profiler.enable=True
        actor_rollout_ref.actor.profiler.tool=npu
        actor_rollout_ref.actor.profiler.all_ranks=True
        'actor_rollout_ref.actor.profiler.ranks=[]'
        actor_rollout_ref.actor.profiler.save_path="${profile_save_path}"
        actor_rollout_ref.actor.profiler.tool_config.npu.discrete=False
        'actor_rollout_ref.actor.profiler.tool_config.npu.contents=[npu,cpu,memory]'
        actor_rollout_ref.actor.profiler.tool_config.npu.level=level1
        actor_rollout_ref.actor.profiler.tool_config.npu.analysis=False

        actor_rollout_ref.ref.profiler.enable=True
        actor_rollout_ref.ref.profiler.tool=npu
        actor_rollout_ref.ref.profiler.all_ranks=True
        'actor_rollout_ref.ref.profiler.ranks=[]'
        actor_rollout_ref.ref.profiler.save_path="${profile_save_path}"
        actor_rollout_ref.ref.profiler.tool_config.npu.discrete=False
        'actor_rollout_ref.ref.profiler.tool_config.npu.contents=[npu,cpu,memory]'
        actor_rollout_ref.ref.profiler.tool_config.npu.level=level1
        actor_rollout_ref.ref.profiler.tool_config.npu.analysis=False

        actor_rollout_ref.rollout.profiler.enable=True
        # In Agent Loop mode, verl adapts this NPU config to vLLM-Ascend's
        # torch_npu profiler control plane.
        actor_rollout_ref.rollout.profiler.tool=npu
        actor_rollout_ref.rollout.profiler.all_ranks=True
        'actor_rollout_ref.rollout.profiler.ranks=[]'
        actor_rollout_ref.rollout.profiler.save_path="${profile_save_path}"
        actor_rollout_ref.rollout.profiler.tool_config.npu.discrete=True
        'actor_rollout_ref.rollout.profiler.tool_config.npu.contents=[npu,cpu,memory]'
        actor_rollout_ref.rollout.profiler.tool_config.npu.level=level1
        actor_rollout_ref.rollout.profiler.tool_config.npu.analysis=False
        actor_rollout_ref.rollout.profiler.tool_config.npu.profile_token_start=null
        actor_rollout_ref.rollout.profiler.tool_config.npu.profile_token_end=null
    )
else
    echo "Kimi profiling: disabled"
fi

DATA_CONFIG=(
    data.train_files="${train_file}"
    data.val_files="${test_file}"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.image_key=images
    data.shuffle=False
    data.validation_shuffle=False
    data.trust_remote_code=True
)

TRAINER_CONFIG=(
    trainer.critic_warmup=0
    'trainer.logger=["console"]'
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.n_gpus_per_node=${npus_per_node}
    trainer.nnodes=${nnodes}
    trainer.balance_batch=True
    trainer.resume_from_path="${resume_from_path}"
    trainer.resume_mode=${resume_mode}
    trainer.save_freq=${save_freq}
    trainer.default_local_dir="${PROJECT_ROOT}/checkpoints/${project_name}/${exp_name}"
    trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep}
    trainer.test_freq=-1
    trainer.val_before_train=False
    trainer.total_epochs=10
    trainer.total_training_steps=${total_training_steps}
    trainer.rollout_data_dir="${rollout_data_dir}"
)

MODEL_CONFIG=(
    actor_rollout_ref.model.path=${model_path}
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_gradient_checkpointing=False
)

ACTOR_CONFIG=(
    actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra']"
    actor_rollout_ref.actor.checkpoint.load_contents="['model','optimizer','extra']"
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
    actor_rollout_ref.actor.optim.weight_decay=0.01
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.optim.optimizer=AdamW
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len}
)

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp_size}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_tp_mode=data
    actor_rollout_ref.rollout.expert_parallel_size=${rollout_ep_size}
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.enable_rollout_routing_replay=True
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len}
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens}
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=${rollout_enforce_eager}
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
    actor_rollout_ref.rollout.load_format=dummy
    actor_rollout_ref.rollout.logprobs_mode=processed_logprobs
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
)

# Kimi MoE reload keeps the graph-captured runtime storage address stable, so
# graph mode can survive weight updates without recapture. Eager remains an
# explicit diagnostic fallback.
case "${rollout_enforce_eager,,}" in
    false|0|no|off)
        ROLLOUT_CONFIG+=(
            +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY
            '+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes=[1,2,4,8,16]'
            +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.max_cudagraph_capture_size=16
            # +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_sleep_mode_extra_cleanup=False
        )
        ;;
esac

FSDP_CONFIG=(
    actor_rollout_ref.actor.strategy=fsdp_turbo
    actor_rollout_ref.actor.fsdp_config.checkpoint_load_mode=streaming_local
    actor_rollout_ref.actor.fsdp_config.checkpoint_load_strict=True
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16
    +actor_rollout_ref.actor.fsdp_config.turbo_config.model.attn_implementation=eager
    +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.ignored_modules=[]
    '++actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.apply_modules={vision_tower.encoder.blocks.\{*\}:{},mm_projector:{},language_model.model.embed_tokens:{},language_model.model.layers.\{*\}.self_attn:{},language_model.model.layers.\{*\}.mlp:{},language_model.model.layers.\{*\}.block_sparse_moe:{},language_model.lm_head:{}}'
    actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.reduce_dtype=bf16
    actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.num_to_forward_prefetch=1
    actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.num_to_backward_prefetch=1
    '+actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.tp_plan.colwise_parallel=["*.q_proj","*.k_proj","*.v_proj"]'
    '+actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.tp_plan.rowwise_parallel=["*.o_proj"]'
    actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fully_shard_parallel_size=${turbo_fsdp_size}
    actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.ulysses_parallel_size=1
    actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.expert_parallel_size=${turbo_ep_size}
    "+actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.cp_plan.ulysses_function_patches=[{target_functions:['transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.eager_attention_forward'],type:full_attention},{target_functions:['transformers.loss.loss_utils.ForCausalLMLoss'],type:causal_lm_loss}]"
    '+actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.ep_plan.apply_modules=["language_model.model.layers.{*}.block_sparse_moe.experts"]'
    +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.ep_plan.dispatcher=custom_ep_forward_fused
    +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.ep_plan.fixed_router=False
    actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.expert_fully_shard_parallel_size=${turbo_efsdp_size}
    '+actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.ep_plan.apply_efsdp_modules=["language_model.model.layers.{*}.block_sparse_moe.experts"]'
    '+actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan=["language_model.model.layers.{*}","vision_tower.encoder.blocks.{*}"]'
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.fsdp_implementation=native
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.fsdp_plan.fsdp_implementation=native
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.fsdp_config.wrap_policy.min_num_params=0
    actor_rollout_ref.actor.fsdp_config.offload_policy=False
    actor_rollout_ref.ref.fsdp_config.offload_policy=False
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.ref.fsdp_config.param_offload=False
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.ref.use_torch_compile=False
    +actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute=True
)

FSDP_CONFIG+=(
    +actor_rollout_ref.actor.fsdp_config.turbo_config.memory.chunk_batch=False
    "+actor_rollout_ref.actor.fsdp_config.turbo_config.memory.chunk_batch_plan={chunk_mbs:${actor_moe_chunk_tokens},apply_modules:['language_model.model.layers.{*}.block_sparse_moe'],batch_dim:1,chunk_arg_indexs:[0],chunk_kwarg_names:[]}"
)

REF_CONFIG=(
    actor_rollout_ref.ref.strategy=fsdp_turbo
    actor_rollout_ref.ref.fsdp_config.checkpoint_load_mode=streaming_local
    actor_rollout_ref.ref.fsdp_config.checkpoint_load_strict=True
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.ref.fsdp_config.dtype=bfloat16
    +actor_rollout_ref.ref.fsdp_config.turbo_config.model.attn_implementation=eager
    +actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.fsdp_plan.ignored_modules=[]
    '++actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.fsdp_plan.apply_modules={vision_tower.encoder.blocks.\{*\}:{},mm_projector:{},language_model.model.embed_tokens:{},language_model.model.layers.\{*\}.self_attn:{},language_model.model.layers.\{*\}.mlp:{},language_model.model.layers.\{*\}.block_sparse_moe:{},language_model.lm_head:{}}'
    actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.fsdp_plan.reduce_dtype=bf16
    actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.fsdp_plan.num_to_forward_prefetch=0
    actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.fsdp_plan.num_to_backward_prefetch=0
    '+actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.tp_plan.colwise_parallel=["*.q_proj","*.k_proj","*.v_proj"]'
    '+actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.tp_plan.rowwise_parallel=["*.o_proj"]'
    actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.fully_shard_parallel_size=${turbo_fsdp_size}
    actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.ulysses_parallel_size=1
    actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.expert_parallel_size=${turbo_ep_size}
    "+actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.cp_plan.ulysses_function_patches=[{target_functions:['transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.eager_attention_forward'],type:full_attention},{target_functions:['transformers.loss.loss_utils.ForCausalLMLoss'],type:causal_lm_loss}]"
    '+actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.ep_plan.apply_modules=["language_model.model.layers.{*}.block_sparse_moe.experts"]'
    +actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.ep_plan.dispatcher=custom_ep_forward_fused
    actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.expert_fully_shard_parallel_size=${turbo_efsdp_size}
    '+actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.ep_plan.apply_efsdp_modules=["language_model.model.layers.{*}.block_sparse_moe.experts"]'
    '+actor_rollout_ref.ref.fsdp_config.turbo_config.memory.recompute_plan=["language_model.model.layers.{*}","vision_tower.encoder.blocks.{*}"]'
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len}
)



QUANT_APPLY_MODULES="['language_model.model.layers.{*}']"
QUANT_IGNORED_MODULES="['lm_head','*mlp.gate','*shared_experts*','*q_proj*','*k_proj*','*v_proj*','*f_a_proj*','*f_b_proj*','*b_proj*','*g_proj*','*mlp*','*self_attn*','*routed_expert*']"
QUANT_CONVERTERS="['quantize.linear.mx','quantize.moe.mx']"

W4A8_CONFIG=(
    +actor_rollout_ref.actor.fsdp_config.turbo_config.quantization.quantization_plan.quant_recipe=W4A8
    +actor_rollout_ref.actor.fsdp_config.turbo_config.quantization.quantization_plan.quant_format=W4A8
    +actor_rollout_ref.actor.fsdp_config.turbo_config.quantization.quantization_plan.converters=${QUANT_CONVERTERS}
    +actor_rollout_ref.actor.fsdp_config.turbo_config.quantization.quantization_plan.quant_apply_modules=${QUANT_APPLY_MODULES}
    +actor_rollout_ref.actor.fsdp_config.turbo_config.quantization.quantization_plan.quant_ignored_modules=${QUANT_IGNORED_MODULES}
    +actor_rollout_ref.actor.fsdp_config.turbo_config.quantization.quantization_plan.enable_fsdp_low_precision_all_gather=True
    +actor_rollout_ref.actor.fsdp_config.turbo_config.quantization.quantization_plan.fsdp_low_precision_all_gather_mode=on-demand
    +actor_rollout_ref.ref.fsdp_config.turbo_config.quantization.quantization_plan.quant_recipe=W4A8
    +actor_rollout_ref.ref.fsdp_config.turbo_config.quantization.quantization_plan.quant_format=W4A8
    +actor_rollout_ref.ref.fsdp_config.turbo_config.quantization.quantization_plan.converters=${QUANT_CONVERTERS}
    +actor_rollout_ref.ref.fsdp_config.turbo_config.quantization.quantization_plan.quant_apply_modules=${QUANT_APPLY_MODULES}
    +actor_rollout_ref.ref.fsdp_config.turbo_config.quantization.quantization_plan.quant_ignored_modules=${QUANT_IGNORED_MODULES}
    +actor_rollout_ref.ref.fsdp_config.turbo_config.quantization.quantization_plan.enable_fsdp_low_precision_all_gather=True
    +actor_rollout_ref.ref.fsdp_config.turbo_config.quantization.quantization_plan.fsdp_low_precision_all_gather_mode=on-demand
    actor_rollout_ref.rollout.quantization=ascend
)

echo "Kimi run: mode=${run_mode} experiment=${exp_name} model=${model_path} steps=${total_training_steps}"
echo "Kimi offload: actor_policy=False actor_param=False actor_optimizer=False ref_config_policy=False ref_forward_only_runtime_policy=True"
echo "Kimi profiling: enabled=${enable_profiling} step=${profile_step}"
echo "Kimi topology: nodes=${nnodes} NPU/node=${npus_per_node} FSDP=${turbo_fsdp_size} EP=${turbo_ep_size} eFSDP=${turbo_efsdp_size} rollout_TP=${rollout_tp_size}"
echo "Kimi sequence: prompt=${max_prompt_length} response=${max_response_length} multimodal_margin=${rollout_multimodal_token_margin} model_len=${rollout_max_model_len} actor_ref_tokens_per_gpu=${ppo_max_token_len}"
echo "Kimi batches: train=${train_batch_size} rollout_n=${rollout_n} effective=$((train_batch_size * rollout_n)) ppo_mini=${ppo_mini_batch_size} micro_per_gpu=1 max_num_seqs=${rollout_max_num_seqs} max_batched_tokens=${rollout_max_num_batched_tokens} moe_chunk=${actor_moe_chunk_tokens}"
echo "Kimi training log: ${training_log}"

# cd "${verl_path}"
python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_trainer.yaml \
    model_engine=dp \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=False \
    "${DATA_CONFIG[@]}" \
    "${MODEL_CONFIG[@]}" \
    "${ACTOR_CONFIG[@]}" \
    "${REF_CONFIG[@]}" \
    "${ROLLOUT_CONFIG[@]}" \
    "${FSDP_CONFIG[@]}" \
    "${TRAINER_CONFIG[@]}" \
    "${RAY_CONFIG[@]}" \
    "${PROFILER_CONFIG[@]}" \
    "$@" 2>&1 | tee "${training_log}"

    # "${W4A8_CONFIG[@]}" \