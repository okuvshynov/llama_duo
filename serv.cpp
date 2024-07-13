#include <cstdio>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <httplib.h>

// llama.cpp
#include <common.h>
#include <llama.h>

/*
 * Need to support API similar to that of normal llama.cpp server
 *
 * request = {
 *    "max_tokens": 1024,
 *    "messages" : [
 *      {"role": "user", "content": "How are you?"},
 *      {"role": "assistant", "content": ",,,,"},
 *    ],
 *    "is_draft" : True ## this is streaming-specific
 * }
 *
 * we'll need to
 * * format it with llama3 template.
 * * handle is_draft correctly
 *
 * response = {
 *    "choices" : [{"delta": {"content": "blabla"}}]
 * }
 *
 * how to handle multiple queries/sequences? seems like we need to use sequence_ids?
 */

using llama_tokens = std::vector<llama_token>;

class log
{
  public:
    static void fatal(const char* format, ...)
    {
        va_list args;
        va_start(args, format);
        inst().write(FATAL, format, args);
        va_end(args);
    }

    static void error(const char* format, ...)
    {
        va_list args;
        va_start(args, format);
        inst().write(ERROR, format, args);
        va_end(args);
    }

    static void warn(const char* format, ...)
    {
        va_list args;
        va_start(args, format);
        inst().write(WARN, format, args);
        va_end(args);
    }

    static void info(const char* format, ...)
    {
        va_list args;
        va_start(args, format);
        inst().write(INFO, format, args);
        va_end(args);
    }

    static void debug(const char* format, ...)
    {
        va_list args;
        va_start(args, format);
        inst().write(DEBUG, format, args);
        va_end(args);
    }

  private:
    enum log_level
    {
        FATAL = 0,
        ERROR = 1,
        WARN  = 2,
        INFO  = 3,
        DEBUG = 4
    };
    static log& inst()
    {
        static log l;
        return l;
    }
    void write(log_level ll, const char* format, va_list args)
    {
        using namespace std::chrono;
        static std::string log_level_str[] = {"F", "E", "W", "I", "D"};

        auto now        = system_clock::now();
        auto now_time_t = system_clock::to_time_t(now);
        auto now_us     = duration_cast<microseconds>(now.time_since_epoch()) % 1000000;
        
        std::tm local;
        localtime_r(&now_time_t, &local);
        std::fprintf
        (
            stderr, 
            "%s%04d-%02d-%02d %02d:%02d:%02d.%06lld ",
            log_level_str[ll].c_str(),
            local.tm_year + 1900, local.tm_mon + 1, local.tm_mday,
            local.tm_hour, local.tm_min, local.tm_sec, now_us.count()
        );
        std::vfprintf(stderr, format, args);
        std::fprintf(stderr, "\n");
    }
};

std::string llama3_instruct_fmt_msg(const nlohmann::json & j)
{
    std::ostringstream oss;
    oss << "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n";
    oss << j.value("system", "") << "<|eot_id|>\n";

    for (const auto& msg: j["messages"])
    {
        oss 
            << "<|start_header_id|>"
            << msg["role"].get<std::string>()
            << "<|end_header_id|>\n\n"
            << msg["content"].get<std::string>() << "<|eot_id|>";
    }

    oss << "<|start_header_id|>assistant<|end_header_id|>";
    return oss.str();
}

// single thread/single query at first
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
                // log::info("done current input");
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
                    // sample next
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
                        while (true)
                        {
                            bool do_next = llm->next(&next);
                            if (!do_next)
                            {
                                sink.done();
                                break;
                            }
                            if (next.size() == 0)
                            {
                                // nothing generated yet, wait
                                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                                continue;
                            }

                            sink.write(next.data(), next.size());
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
