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
    std::mutex mutex;
    // the input passed. Should include the history,
    // as it is not guaranteed that we'll have cache.
    // as we produce new tokens we append it here
    llama_tokens tokens; 

    // output for the current message.
    // TODO: do we need to support some offset-based indexing for retries?
    std::queue<std::string> output;

    // input done - we are ready to generate output
    // output done - we have completed the turn for this session
    bool input_done  = false;
    bool output_done = false;

    // what do we do if we are generating output and got input update?
};

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

    void update_prompt(std::string s, bool input_done)
    {
        log::info("updating prompt, new s=%zu", s.size());
        {
            std::lock_guard<std::mutex> _lock(session_ctx_.mutex);
            session_ctx_.tokens = llama_tokenize(ctx_, s, true);
            session_ctx_.input_done = input_done;
        }
    }

    // here we continously process the prompt
    void loop()
    {
        // TODO: configure batch size
        const int batch_size = 32;
        llama_batch batch = llama_batch_init(batch_size, 0, 1);
        llama_tokens input;
        // how many processed. Should this be part of session too?
        size_t n_done = 0;
        while (true)
        {
            //log::info("iter");
            {
                std::lock_guard<std::mutex> _lock(session_ctx_.mutex);
                //log::info("checking input");
                if (input != session_ctx_.tokens)
                {
                    log::info("updating input");
                    size_t n_matched = std::min(session_ctx_.tokens.size(), input.size());
                    for (size_t i = 0; i < n_matched; i++)
                    {
                        if (session_ctx_.tokens[i] != input[i])
                        {
                            n_matched = i;
                            break;
                        }
                    }
                    input = session_ctx_.tokens;
                    log::info("done: %zu, matched: %zu", n_done, n_matched);
                    if (n_done > n_matched)
                    {
                        n_done = n_matched;
                        llama_kv_cache_seq_rm(ctx_, 0, n_done, -1);
                    }
                }
                else
                {
                    //log::info("not updating input");
                }
            }
            if (n_done >= input.size())
            {
                //log::info("done current input");
                bool do_sampling = false;
                {
                    std::lock_guard<std::mutex> _lock(session_ctx_.mutex);
                    do_sampling = session_ctx_.input_done && !session_ctx_.output_done;
                }
                if (!do_sampling)
                {
                    //log::info("waiting for next chunk");
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
                else
                {
                    //log::info("sample next");
                    llama_token id = llama_sampling_sample(ctx_sampling_, ctx_, nullptr, batch.n_tokens - 1);
                    llama_sampling_accept(ctx_sampling_, ctx_, id, true);
                    auto out = llama_token_to_piece(ctx_, id);
                    bool done = false;
                    // TODO: also compare with n_predict
                    done = llama_token_is_eog(model_, id);

                    {
                        std::lock_guard<std::mutex> _lock(session_ctx_.mutex);
                        session_ctx_.output.push(out);
                        session_ctx_.output_done = done;
                    }

                    if (done)
                    {
                        continue;
                    }
                    llama_batch_clear(batch);
                    llama_batch_add(batch, id, n_done, { 0 }, true);
                    n_done += 1;
                    if (llama_decode(ctx_, batch) != 0)
                    {
                        log::error("llama_decode() failed");
                    }
                }
            }
            else
            {
                // processing input
                //log::info("processing input");
                size_t i;
                llama_batch_clear(batch);
                for (i = 0; i < batch_size && i + n_done < input.size(); i++)
                {
                    llama_batch_add(batch, input[i + n_done], i + n_done, {0}, false);
                }
                if (i + n_done == input.size())
                {
                    batch.logits[batch.n_tokens - 1] = true;
                }
                if (llama_decode(ctx_, batch) != 0)
                {
                    log::error("llama_decode() failed");
                }
                n_done += i;
                log::info("n_done = %zu", n_done);
            }
        }
    }

    bool next(std::string * s)
    {
        //log::info("next()");
        std::lock_guard<std::mutex> _lock(session_ctx_.mutex);
        *s = "";
        while (!session_ctx_.output.empty())
        {
            *s += session_ctx_.output.front();
            session_ctx_.output.pop();
        }
        return !session_ctx_.output_done;
    }

  private:
    llama_model * model_;
    llama_context * ctx_;
    llama_sampling_context * ctx_sampling_;
    std::thread loop_thread_;

    std::mutex mutex_;

    session_context session_ctx_;
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
            auto req_j = json::parse(req.body);
            std::string text = llama3_instruct_fmt_msg(req_j);
            bool complete = req_j["complete"];

            llm->update_prompt(text, complete);
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
