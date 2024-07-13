#include <cstdio>
#include <condition_variable>
#include <iostream>
#include <mutex>
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

        gen_done_ = false;
        do_sampling_ = false;
        output_returned_ = 0;
    }

    ~llama()
    {
        loop_thread_.join();
        llama_sampling_free(ctx_sampling_);
        llama_free(ctx_);
        llama_free_model(model_);
        llama_backend_free();
    }

    void update_prompt(std::string s, bool start_sampling)
    {
        log::info("updating prompt, new s=%zu", s.size());
        {
            std::lock_guard<std::mutex> _lock(mutex_);
            llama_tokens input = llama_tokenize(ctx_, s, true);
            input_ = input;
            do_sampling_ = start_sampling;
        }
    }

    // here we continously process the prompt
    void loop()
    {
        // TODO: configure batch size
        const int batch_size = 32;
        llama_batch batch = llama_batch_init(batch_size, 0, 1);
        llama_tokens input;
        // how many processed 
        size_t n_done = 0;
        while (true)
        {
            //log::info("iter");
            {
                std::lock_guard<std::mutex> _lock(mutex_);
                //log::info("checking input");
                if (input != input_)
                {
                    log::info("updating input");
                    size_t n_matched = std::min(input_.size(), input.size());
                    for (size_t i = 0; i < n_matched; i++)
                    {
                        if (input_[i] != input[i])
                        {
                            n_matched = i;
                            break;
                        }
                    }
                    input = input_;
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
                    std::lock_guard<std::mutex> _lock(mutex_);
                    do_sampling = do_sampling_ && !gen_done_;
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
                        std::lock_guard<std::mutex> _lock(mutex_);
                        output_ += out;
                        gen_done_ = done;
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
        std::lock_guard<std::mutex> _lock(mutex_);
        *s = output_.substr(output_returned_);
        output_returned_ = output_.size();
        return !gen_done_;
    }

  private:
    llama_model * model_;
    llama_context * ctx_;
    llama_sampling_context * ctx_sampling_;
    std::thread loop_thread_;

    std::mutex mutex_;
    llama_tokens input_;
    bool gen_done_, do_sampling_;
    std::string output_;
    size_t output_returned_;
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
