// tree-dump: offline DSpark draft-tree study harness.
//
// Generates a greedy reference sequence with the target model while feeding the
// DSpark drafter (common_speculative) exactly as online speculative decoding
// would — but drafts are never committed. At every position it calls
// common_speculative_draft() and dumps the drafter's full-vocab biased logits
// for all block slots, the 5 per-slot confidences, and the greedy draft chain.
//
// The dump plus the drafter's markov_w1/markov_w2 tensors (read from the GGUF
// in analyze.py) is enough to reconstruct the drafter's entire top-k draft
// tree at every position: base logits per slot are independent of the drafted
// tokens, branching enters only through the additive rank-256 Markov bias.
//
// Usage: same flags as llama-speculative-simple (target/draft models, -ngl,
// -ncmoe, -ts, --spec-type draft-dspark, ...), plus environment variables:
//   DSTREE_PROMPTS  comma-separated prompt files (all run in one model load)
//   DSTREE_OUT      output directory (default: results)
// -n controls tokens generated per prompt.
//
// Output per prompt: <out>/<prompt-stem>.dstree (binary, header below) and
// <out>/<prompt-stem>.meta.json (full token sequence for alignment).

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"
#include "llama-ext.h"

#include <cinttypes>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct dstree_header {
    char     magic[8];   // "DSPKTRE2"
    uint32_t n_vocab;
    uint32_t block_size;
};

// record layout (fixed size, little-endian, follows header back to back):
//   u32 step, u32 n_past, i32 anchor
//   i32 chain[block_size]   (drafter's own greedy chain; -1 pad)
//   f32 conf[block_size]
//   f32 logits[block_size][n_vocab]   (biased, i.e. what the sampler sees)
//   f32 base[block_size][n_vocab]     (pre-Markov "result_output" logits)
//
// base is captured with a cb_eval callback on the draft context: the DSV4
// dflash graph names the pre-Markov logits "result_output" and the Markov head
// only adds new nodes on top, so the callback sees the unbiased tensor the
// moment it is computed. biased - base == markov_w2 @ markov_w1[prev] per
// slot, which lets analyze.py validate its numpy Markov-bias reconstruction
// elementwise on every record.

struct base_capture {
    std::vector<float> data;
    int64_t ne0  = 0;
    int64_t ne1  = 0;
    bool    seen = false;
};

static bool cb_eval_capture_base(struct ggml_tensor * t, bool ask, void * ud) {
    auto * cap = (base_capture *) ud;
    const bool match = strcmp(t->name, "result_output") == 0;
    if (ask) {
        return match;
    }
    if (match) {
        cap->ne0 = t->ne[0];
        cap->ne1 = t->ne[1];
        cap->data.resize((size_t) ggml_nelements(t));
        ggml_backend_tensor_get(t, cap->data.data(), 0, ggml_nbytes(t));
        cap->seen = true;
    }
    return true;
}

static llama_token argmax_row(const float * l, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (l[i] > l[best]) {
            best = i;
        }
    }
    return best;
}

static std::string path_stem(const std::string & p) {
    size_t slash = p.find_last_of("/\\");
    std::string base = slash == std::string::npos ? p : p.substr(slash + 1);
    size_t dot = base.find_last_of('.');
    return dot == std::string::npos ? base : base.substr(0, dot);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    // the study wants the full trained block and no early exits; the biased
    // logits must reach the host in full (backend sampling keeps only top-10)
    params.speculative.draft.n_max            = 5; // clamped to dflash.block_size by the impl
    params.speculative.draft.n_min            = 0;
    params.speculative.draft.p_min            = 0.0f;
    params.speculative.draft.backend_sampling = false;

    const auto output_limits = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max         = output_limits.total;
    params.n_outputs_max_per_seq = output_limits.per_seq;

    llama_backend_init();
    llama_numa_init(params.numa);

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    llama_model   * model_tgt = llama_init_tgt->model();
    llama_context * ctx_tgt   = llama_init_tgt->context();

    if (model_tgt == nullptr || ctx_tgt == nullptr) {
        LOG_ERR("%s", "failed to load the target model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // load the draft model
    base_capture cap;

    common_speculative_init_result_ptr spec_init;
    {
        common_params params_dft = common_base_params_to_speculative(params);

        // capture the pre-Markov "result_output" logits of every draft decode
        params_dft.cb_eval           = cb_eval_capture_base;
        params_dft.cb_eval_user_data = &cap;

        spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);

        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = spec_init->context();
    }

    llama_model   * model_dft = spec_init->model();
    llama_context * ctx_dft   = params.speculative.draft.ctx_dft;

    if (model_dft == nullptr || ctx_dft == nullptr) {
        LOG_ERR("%s", "failed to load the draft model (-md required)\n");
        return 1;
    }

    const int32_t n_vocab    = llama_vocab_n_tokens(vocab);
    const int32_t n_embd_dec = llama_model_n_embd(model_dft);

    int32_t block_size = 0;
    {
        char buf[32] = {};
        if (llama_model_meta_val_str(model_dft, "dflash.block_size", buf, sizeof(buf)) >= 0) {
            block_size = std::atoi(buf);
        }
    }
    if (block_size <= 0) {
        LOG_ERR("%s", "draft model has no dflash.block_size - not a DSpark/DFlash drafter?\n");
        return 1;
    }

    LOG_INF("tree-dump: n_vocab=%d block_size=%d n_embd_dec=%d\n", n_vocab, block_size, n_embd_dec);

    const bool use_ckpt_dft = common_context_can_seq_rm(ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    LOG_INF("tree-dump: use_ckpt_dft=%d\n", use_ckpt_dft ? 1 : 0);

    struct common_speculative * spec = common_speculative_init(params.speculative, 1);
    if (spec == nullptr) {
        LOG_ERR("%s", "failed to initialize speculative decoding\n");
        return 1;
    }

    // prompt list
    std::vector<std::string> prompt_files;
    if (const char * env = getenv("DSTREE_PROMPTS")) {
        std::string s(env);
        size_t beg = 0;
        while (beg < s.size()) {
            size_t end = s.find(',', beg);
            if (end == std::string::npos) end = s.size();
            if (end > beg) prompt_files.push_back(s.substr(beg, end - beg));
            beg = end + 1;
        }
    }
    if (prompt_files.empty()) {
        LOG_ERR("%s", "set DSTREE_PROMPTS to a comma-separated list of prompt files\n");
        return 1;
    }

    std::string out_dir = "results";
    if (const char * env = getenv("DSTREE_OUT")) {
        out_dir = env;
    }
    std::string cmd = "mkdir -p '" + out_dir + "'";
    if (system(cmd.c_str()) != 0) {
        LOG_ERR("failed to create output dir %s\n", out_dir.c_str());
        return 1;
    }

    const int n_gen_max = params.n_predict > 0 ? params.n_predict : 512;

    const llama_seq_id seq_id = 0;

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    std::vector<float> conf_rec(block_size);
    std::vector<int32_t> chain_rec(block_size);

    for (const auto & pf : prompt_files) {
        std::ifstream f(pf);
        if (!f) {
            LOG_ERR("cannot read prompt file %s\n", pf.c_str());
            return 1;
        }
        std::string prompt((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

        // reset both contexts for the new prompt
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, -1, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, -1, -1);

        std::vector<llama_token> inp = common_tokenize(ctx_tgt, prompt, true, true);

        if (llama_n_ctx(ctx_tgt) < inp.size() + n_gen_max + block_size + 2) {
            LOG_ERR("prompt %s + generation does not fit the context (%zu + %d, ctx %u)\n",
                    pf.c_str(), inp.size(), n_gen_max, llama_n_ctx(ctx_tgt));
            return 1;
        }
        if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
            LOG_ERR("prompt %s exceeds the batch size (%zu, batch %u)\n", pf.c_str(), inp.size(), llama_n_batch(ctx_tgt));
            return 1;
        }

        const std::string stem = path_stem(pf);
        const std::string bin_path  = out_dir + "/" + stem + ".dstree";
        const std::string meta_path = out_dir + "/" + stem + ".meta.json";

        FILE * fout = fopen(bin_path.c_str(), "wb");
        if (!fout) {
            LOG_ERR("cannot open %s for writing\n", bin_path.c_str());
            return 1;
        }
        {
            dstree_header hdr;
            memcpy(hdr.magic, "DSPKTRE2", 8);
            hdr.n_vocab    = (uint32_t) n_vocab;
            hdr.block_size = (uint32_t) block_size;
            fwrite(&hdr, sizeof(hdr), 1, fout);
        }

        LOG_INF("\n=== prompt %s: %zu tokens, generating %d ===\n", pf.c_str(), inp.size(), n_gen_max);

        const auto t_start = ggml_time_us();

        // prefill all but the last prompt token (no outputs needed)
        {
            common_batch_clear(batch_tgt);
            for (size_t i = 0; i < inp.size() - 1; ++i) {
                common_batch_add(batch_tgt, inp[i], i, { seq_id }, false);
            }
            if (llama_decode(ctx_tgt, batch_tgt) != 0) {
                LOG_ERR("prefill decode failed for %s\n", pf.c_str());
                return 1;
            }
            if (!common_speculative_process(spec, batch_tgt)) {
                LOG_ERR("%s", "failed to process speculative prompt\n");
                return 1;
            }
        }

        llama_token id_last = inp.back();

        llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
        prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

        int n_past = (int) inp.size() - 1;

        common_speculative_begin(spec, seq_id, prompt_tgt);

        common_prompt_checkpoint ckpt;

        llama_tokens draft;

        int n_records = 0;
        int n_chain_mismatch = 0;

        for (int step = 0; step < n_gen_max; ++step) {
            // ---- draft at the current state: anchor id_last at position n_past ----
            ckpt.update_pos(
                    prompt_tgt.size(),
                    llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id));

            if (use_ckpt_dft) {
                ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            draft.clear();
            cap.seen = false;

            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting   = */ true,
                /* .n_max      = */ -1,
                /* .n_past     = */ n_past,
                /* .id_last    = */ id_last,
                /* .prompt     = */ &prompt_tgt,
                /* .result     = */ &draft,
            };
            common_speculative_draft(spec);

            if ((int) draft.size() != block_size) {
                LOG_ERR("step %d: draft returned %zu tokens, expected %d\n", step, draft.size(), block_size);
                return 1;
            }
            if (!cap.seen || cap.ne0 != n_vocab || cap.ne1 != block_size) {
                LOG_ERR("step %d: base-logits capture failed (seen=%d ne=[%" PRId64 ",%" PRId64 "])\n",
                        step, cap.seen ? 1 : 0, cap.ne0, cap.ne1);
                return 1;
            }

            // ---- dump: biased logits + confidences for all block slots ----
            {
                const float * conf_all = llama_get_embeddings_nextn(ctx_dft);

                for (int i = 0; i < block_size; ++i) {
                    chain_rec[i] = draft[i];
                    conf_rec[i]  = conf_all ? conf_all[(size_t) i * n_embd_dec] : -1.0f;
                }

                const uint32_t step_u = (uint32_t) step;
                const uint32_t past_u = (uint32_t) n_past;
                fwrite(&step_u,  sizeof(step_u),  1, fout);
                fwrite(&past_u,  sizeof(past_u),  1, fout);
                fwrite(&id_last, sizeof(id_last), 1, fout);
                fwrite(chain_rec.data(), sizeof(int32_t), block_size, fout);
                fwrite(conf_rec.data(),  sizeof(float),   block_size, fout);

                for (int i = 0; i < block_size; ++i) {
                    const float * row = llama_get_logits_ith(ctx_dft, i);
                    if (row == nullptr) {
                        LOG_ERR("step %d: no logits at draft slot %d\n", step, i);
                        return 1;
                    }
                    // cheap invariant: the chain token must be the argmax of the row
                    if (argmax_row(row, n_vocab) != draft[i]) {
                        n_chain_mismatch++;
                    }
                    fwrite(row, sizeof(float), n_vocab, fout);
                }
                fwrite(cap.data.data(), sizeof(float), (size_t) block_size * n_vocab, fout);
                n_records++;
            }

            // ---- restore the drafter KV (drop the noise block) ----
            if (use_ckpt_dft) {
                ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);

            // ---- decode the pending token on the target, greedy-pick the next ----
            common_batch_clear(batch_tgt);
            common_batch_add(batch_tgt, id_last, n_past, { seq_id }, true);

            if (llama_decode(ctx_tgt, batch_tgt) != 0) {
                LOG_ERR("step %d: target decode failed\n", step);
                return 1;
            }
            if (!common_speculative_process(spec, batch_tgt)) {
                LOG_ERR("step %d: failed to process speculative batch\n", step);
                return 1;
            }

            const float * logits_tgt = llama_get_logits_ith(ctx_tgt, 0);
            const llama_token next = argmax_row(logits_tgt, n_vocab);

            prompt_tgt.push_back(id_last);
            id_last = next;
            n_past++;

            LOG("%s", common_token_to_piece(ctx_tgt, next).c_str());

            if (llama_vocab_is_eog(vocab, next)) {
                LOG_INF("\n[eog at step %d]\n", step);
                break;
            }
        }

        fclose(fout);

        const auto t_end = ggml_time_us();

        // sidecar: the full token sequence (prompt + generated), for alignment.
        // token at index p sits at position p; a record with n_past=p is anchored
        // on tokens[p] and its true continuation is tokens[p+1 .. p+block_size].
        {
            FILE * fm = fopen(meta_path.c_str(), "w");
            if (!fm) {
                LOG_ERR("cannot open %s for writing\n", meta_path.c_str());
                return 1;
            }
            fprintf(fm, "{\n  \"prompt_file\": \"%s\",\n  \"n_prompt\": %zu,\n  \"n_records\": %d,\n  \"n_chain_mismatch\": %d,\n  \"tokens\": [",
                    pf.c_str(), inp.size(), n_records, n_chain_mismatch);
            for (size_t i = 0; i < prompt_tgt.size(); ++i) {
                fprintf(fm, "%s%d", i ? "," : "", prompt_tgt[i]);
            }
            fprintf(fm, ",%d]\n}\n", id_last);
            fclose(fm);
        }

        LOG_INF("\n%s: %d records (%d chain/argmax mismatches) in %.1f s -> %s\n",
                stem.c_str(), n_records, n_chain_mismatch, (t_end - t_start) / 1e6, bin_path.c_str());
    }

    llama_batch_free(batch_tgt);
    common_speculative_free(spec);
    llama_backend_free();

    return 0;
}
