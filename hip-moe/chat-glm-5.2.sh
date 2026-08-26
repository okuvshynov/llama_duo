#!/bin/bash
# Interactive chat with GLM-5.2 (UD-Q3_K_XL, 319 GiB) on the four Vega II
# dies, with MTP speculative decoding from the model's own NextN layer.
#
#   ./chat-glm-5.2.sh             # interactive llama-cli chat
#   ./chat-glm-5.2.sh server      # llama-server: web UI + OpenAI API
#                                 #   http://127.0.0.1:8091
#   SPEC=none ./chat-glm-5.2.sh   # disable speculation
#
# Placement notes (capacity frontier computed from per-tensor sizes and
# validated 2026-08-25; smoke logs in session scratchpad):
#   79 blocks = 3 dense + 75 MoE + blk.78 (NextN/MTP, has its own experts).
#   Experts are 300 of 319 GiB (3984 MiB/layer; exceptions: blk.8 5376,
#   blk.75-77 4872, blk.78 4368).
#   -ncmoe 56 -ts 59/7/7/7  the frontier: 23 expert layers in HBM, 53 on CPU.
#                           ts counts 80 units (79 layers + output). die 0 =
#                           blocks 0..58 (56 light + 3 expert) + the 2.5 GiB
#                           compute buffer; dies 1-2 = 7 full layers each;
#                           die 3 = 73..78 (incl. the three 4872 MiB layers +
#                           MTP) + output. Loads with 2.5-3.6 GiB free per
#                           die; one more expert layer anywhere overflows a
#                           die. ncmoe counts BLOCK indices (blocks 0-2 are
#                           dense), and ts must align with the ncmoe boundary:
#                           a misaligned split OOMs the die-0 compute buffer.
#   --spec-type draft-mtp   drafts from blk.78's NextN head against the target
#                           model itself - no separate draft GGUF, shares the
#                           target KV (mem-shared mode). NOTE: blk.78 (4.6
#                           GiB) is only loaded when speculation is on, so
#                           SPEC=none frees 4.6 GiB on die 3.
#   --fit off               the fitter can only project, not fix, with -ngl
#                           user-set; everything is pinned by hand anyway.
#
# Measured (n=1 load each, 256-token generations): 3.89 t/s no-spec at the
# old conservative ncmoe 64; 4.46 t/s +MTP (acceptance 0.763); 4.56 t/s +MTP
# at ncmoe 56 (acceptance 0.716 that run - sampling noise, the placement gain
# is ~5-6%). MTP acc/pos ~(0.94, 0.76, 0.58); spec-draft-n-max not yet swept.

MODEL=${MODEL:-$HOME/llms/glm-5.2-q3/UD-Q3_K_XL/GLM-5.2-UD-Q3_K_XL-00001-of-00009.gguf}
BIN_DIR=${BIN_DIR:-$HOME/projects/llama.cpp/build-hip/bin}
CTX=${CTX:-8192}
SPEC=${SPEC:-draft-mtp}
SPEC_N=${SPEC_N:-3}

ARGS=(
    -m "$MODEL"
    -ngl 99 -ncmoe 56 -ts 59/7/7/7
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
