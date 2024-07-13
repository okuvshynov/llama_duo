#include <cstdio>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <httplib.h>
#include <common.h> // llama.cpp
#include <llama.h>

#include "llm_formats.h"
#include "logging.h"

using llama_tokens = std::vector<llama_token>;

// this would be session context which would handle both priming
// queries and conversations. Each conversation is a session.
struct session_context
{
    // the input passed. Should include the history,
    // as it is not guaranteed that we'll have cache.
    // as we produce new tokens we append it here. This can be updated
    // from both processing and API calls.
    std::string  input_str;
    bool         input_updated = false;
    llama_tokens tokens; 
    
    // input we worked on. 
    llama_tokens input;
    // how many of the input above was processed
    size_t n_done;

    // output for the current message.
    // TODO: do we need to support some offset-based indexing for retries?
    std::queue<std::string> output;

    // input done - we are ready to generate output
    // output done - we have completed the turn for this session
    bool input_done  = false;
    bool output_done = false;

  private:
    // what do we do if we are generating output and got input update?
    friend class locked_session;
    std::mutex mutex;
};

class locked_session
{
  public:
    locked_session(session_context& session_ctx): session_ctx_(&session_ctx), lock_(session_ctx.mutex) {}
    
    locked_session(const locked_session &) = delete;
    locked_session& operator=(const locked_session&) = delete;

    locked_session(locked_session && other) noexcept
        : session_ctx_(other.session_ctx_), lock_(std::move(other.lock_)) {
        other.session_ctx_ = nullptr;
    }

    session_context* operator->()
    {
        return session_ctx_;
    }
  private:
    session_context * session_ctx_;
    std::unique_lock<std::mutex> lock_;
};

// just one session for now. This should become a manager with map
locked_session get_locked_session(uint64_t /* id */)
{
    static session_context session_ctx;
    return locked_session(session_ctx);
}

// llama itself should be generic enough to support duo/speculation/rpc-based remote runs
// it will work with one query at a time, and server would dispatch based on state
// we need to decouple session state/context management and actual llm evaluation
class llama 
{
  public:
    llama(gpt_params params)
    {
        llama_backend_init();
        llama_numa_init(params.numa);
        std::tie(model_, ctx_) = llama_init_from_gpt_params(params);
        ctx_sampling_ = llama_sampling_init(params.sparams);

        loop_thread_ = std::thread(&llama::loop, this);
    }

    ~llama()
    {
        loop_thread_.join();
        llama_sampling_free(ctx_sampling_);
        llama_free(ctx_);
        llama_free_model(model_);
        llama_backend_free();
    }

    void update_prompt(uint64_t session_id, std::string s, bool input_done)
    {
        auto session = get_locked_session(0ull);
        session->input_str  = s;
        session->input_done = input_done;
        session->input_updated = true;
        log::info("prompt updated");
    }

    // here we continously process the prompt
    void loop()
    {
        // TODO: configure batch size
        const int batch_size = 32;
        llama_batch batch = llama_batch_init(batch_size, 0, 1);
        bool sleep_please = false;

        while (true)
        {
            if (sleep_please)
            {
                sleep_please = false;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            {
                auto session = get_locked_session(0ull);
                if (session->input_updated)
                {
                    session->tokens = llama_tokenize(ctx_, session->input_str, true);
                    session->input_updated = false;
                }
                if (session->input != session->tokens)
                {
                    log::info("updating input");
                    size_t n_matched = std::min(session->tokens.size(), session->input.size());
                    for (size_t i = 0; i < n_matched; i++)
                    {
                        if (session->tokens[i] != session->input[i])
                        {
                            n_matched = i;
                            break;
                        }
                    }
                    session->input = session->tokens;
                    log::info("done: %zu, matched: %zu", session->n_done, n_matched);
                    if (session->n_done > n_matched)
                    {
                        session->n_done = n_matched;
                        llama_kv_cache_seq_rm(ctx_, 0, session->n_done, -1);
                    }
                }
                if (session->n_done >= session->input.size())
                {
                    bool do_sampling = session->input_done && ! session->output_done;
                    if (!do_sampling)
                    {
                        //log::info("waiting for next chunk");
                        sleep_please = true;
                        continue;
                    }
                    //log::info("sample next");
                    llama_token id = llama_sampling_sample(ctx_sampling_, ctx_, nullptr, batch.n_tokens - 1);
                    llama_sampling_accept(ctx_sampling_, ctx_, id, true);
                    auto out  = llama_token_to_piece(ctx_, id);
                    bool done = false;
                    // TODO: also compare with n_predict
                    done = llama_token_is_eog(model_, id);

                    session->output.push(out);
                    session->output_done = done;

                    if (done)
                    {
                        continue;
                    }

                    llama_batch_clear(batch);
                    llama_batch_add(batch, id, session->n_done, { 0 }, true);
                    session->n_done += 1;
                    log::info("decoding n_done = %zu", session->n_done);
                }
                else
                {
                    // processing input
                    //log::info("processing input");
                    size_t i;
                    llama_batch_clear(batch);
                    for (i = 0; i < batch_size && i + session->n_done < session->input.size(); i++)
                    {
                        size_t j = i + session->n_done;
                        llama_batch_add(batch, session->input[j], j, {0}, false);
                    }
                    if (i + session->n_done == session->input.size())
                    {
                        batch.logits[batch.n_tokens - 1] = true;
                    }
                    session->n_done += i;
                    log::info("priming n_done = %zu", session->n_done);
                }
            }
            // not locking session
            if (llama_decode(ctx_, batch) != 0)
            {
                log::error("llama_decode() failed");
            }
        }
    }

    bool next(std::string * s)
    {
        //log::info("next()");
        auto session = get_locked_session(0ull);
        *s = "";
        while (!session->output.empty())
        {
            *s += session->output.front();
            session->output.pop();
        }
        return !session->output_done;
    }

  private:
    llama_model * model_;
    llama_context * ctx_;
    llama_sampling_context * ctx_sampling_;
    std::thread loop_thread_;

    // for all operations on llama context
    std::mutex mutex_;
};

void serve(std::shared_ptr<llama> llm)
{
    using nlohmann::json;

    httplib::Server http_server;
    // TODO: configure this
    std::string addr = "0.0.0.0";
    int port = 5555;

    http_server.Post("/query", [&llm](const httplib::Request & req, httplib::Response & res)
    {
        try
        {
            log::info("got query");
            auto req_j = json::parse(req.body);
            std::string text = llama3_instruct_fmt_msg(req_j);
            bool complete = req_j["complete"];

            llm->update_prompt(/*session_id = */ 0ull, text, complete);
            if (!complete)
            {
                res.set_content("revcd\n", "application/json");
            }
            else
            {
                res.set_chunked_content_provider(
                    "application/json",
                    [&llm](size_t /* offset */, httplib::DataSink& sink) {
                        std::string next;
                        json res_j;
                        res_j["choices"] = json::array();
                        while (true)
                        {
                            bool do_next = llm->next(&next);
                            //log::info("next is %s", next.c_str());
                            if (!do_next)
                            {
                                sink.done();
                                // TODO: mark everything as done, reset generation.
                                break;
                            }
                            if (next.size() == 0)
                            {
                                // nothing generated yet, wait
                                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                                continue;
                            }

                            res_j["choices"][0]["delta"]["content"] = next;
                            std::string res_s = res_j.dump() + "\n";
                            //log::info("returning %s", res_s.c_str());
                            sink.write(res_s.data(), res_s.size());
                            break;
                        }
                        return true;
                    }
                );
            }
        }
        catch(const std::exception & e)
        {
            log::error("%s", e.what());
        }
    });

    log::info("starting server on %s:%d\n", addr.c_str(), port);
    http_server.listen(addr, port);
}

int main(int argc, char ** argv)
{
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false)
    {
        return 1;
    }

    auto llm = std::make_shared<llama>(params);

    serve(llm);

    return 0;
}
