#!/bin/bash
# GLM-5.2 split across TWO machines (8 Vega II dies, 256 GB HBM total) via
# llama.cpp RPC. Validated 2026-08-26: decode 6.65 t/s with MTP speculation
# vs 4.56 single-machine (+46%), one-load numbers.
#
# Remote prerequisite (tomb, 192.168.2.4): rpc-server built at the SAME
# llama.cpp commit (RPC protocol is version-checked; worktree
# ~/projects/llama.cpp-rpc @ 60eeeb608, build-hip-rpc, GGML_RPC=ON):
#   ssh 192.168.2.4 '~/projects/llama.cpp-rpc/build-hip-rpc/bin/ggml-rpc-server \
#       -H 192.168.2.4 -p 50052 -t 16'
# The RPC protocol is unauthenticated - bind only to the private switched LAN.
#
#   ./chat-glm-5.2-rpc.sh             # interactive chat
#   ./chat-glm-5.2-rpc.sh server      # llama-server on :8091
#
# Placement (the "sandwich", computed from per-layer sizes, load-verified):
#   -dev ROCm0,ROCm1,RPC0..3,ROCm2,ROCm3 with -ts 31/7/7/7/7/7/7/7
#   RPC devices otherwise occupy the FIRST -ts slots; -dev reorders so that:
#     ROCm0 (local): layers 0-30 = 27 light + 4 expert + the 2.4 GiB compute
#                    buffer. The light layers' experts run on the LOCAL CPU
#                    (-ncmoe 27, blocks 0-26), so these layers must sit on a
#                    local die: putting them on a remote die interleaves
#                    remote attention with local-CPU experts - ~46 network
#                    crossings per token and a staging buffer blown to
#                    3.8 GiB (OOM). The sandwich crosses the wire exactly
#                    twice per token.
#     ROCm1, RPC0-3, ROCm2: 7 pure-GPU expert layers each (29.4 GiB).
#     ROCm3 (local): tail 73-78 + output - the MTP layer (blk.78) and the
#                    head stay local, so drafting and sampling never cross
#                    the network.
#   Expert residency: 52 of 76 expert layers in HBM (~69% of expert bytes),
#   vs 23 layers (31%) single-machine. Load ~5.6 min warm (~118 GiB over
#   10GbE near line rate).

MODEL=${MODEL:-$HOME/llms/glm-5.2-q3/UD-Q3_K_XL/GLM-5.2-UD-Q3_K_XL-00001-of-00009.gguf}
BIN_DIR=${BIN_DIR:-$HOME/projects/llama.cpp/build-hip/bin}
CTX=${CTX:-8192}
SPEC=${SPEC:-draft-mtp}
SPEC_N=${SPEC_N:-3}
RPC=${RPC:-192.168.2.4:50052}

ARGS=(
    -m "$MODEL"
    --rpc "$RPC"
    -dev ROCm0,ROCm1,RPC0,RPC1,RPC2,RPC3,ROCm2,ROCm3
    -ngl 99 -ncmoe 27 -ts 31/7/7/7/7/7/7/7
    -t 16 -c "$CTX" --fit off
    --jinja
)

if [ "$SPEC" != "none" ]; then
    ARGS+=(--spec-type "$SPEC" --spec-draft-n-max "$SPEC_N")
fi

if [ "$1" = "server" ]; then
    exec "$BIN_DIR/llama-server" "${ARGS[@]}" --host 127.0.0.1 --port 8091
else
    exec "$BIN_DIR/llama-cli" "${ARGS[@]}" -cnv "$@"
fi
