#include <nlohmann/json.hpp>
#include <httplib.h>

// llama.cpp
#include <common.h>
#include <llama.h>

/*

Test example:
curl -s -S -X POST "http://127.0.0.1:5555/query" -H 'Content-Type: application/json' -d '{"text": "t", "offset": 10, "complete": true}'
*/

using llama_tokens = std::vector<llama_token>;

// single thread/single query at first
class llama 
{
  public:
    llama(gpt_params params)
    {
        llama_backend_init();
        llama_numa_init(params.numa);
        std::tie(model_, ctx_) = llama_init_from_gpt_params(params);

    }

    ~llama()
    {
        llama_free(ctx_);
        llama_free_model(model_);
        llama_backend_free();

    }

    void update_prompt(std::string s)
    {
        llama_tokens input = llama_tokenize(ctx_, s, true);
        // now we need to compare it to input_ we work on and reset cache if needed

    }



    bool next(std::string * s)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        static std::string val = "";
        val += "A";
        nlohmann::json res_j;
        res_j["text"] = val + " ";
        *s = res_j.dump() + "\n"; 
        return val.size() < 5;
    }

  private:
    llama_model * model_;
    llama_context * ctx_;
    llama_tokens input_;
};

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

void serve(std::shared_ptr<llama> llm)
{
    using nlohmann::json;

    httplib::Server http_server;
    std::string addr = "0.0.0.0";
    int port = 5555;

    std::string curr = "";

    http_server.Post("/query", [&curr, &llm](const httplib::Request & req, httplib::Response & res)
    {
        try
        {
            auto req_j = json::parse(req.body);
            json res_j;

            size_t offset = req_j["offset"];
            std::string text = req_j["text"];
            bool complete = req_j["complete"];

            curr.resize(offset + text.size(), ' ');
            curr.replace(offset, text.size(), text);

            llm->update_prompt(curr);
            if (!complete)
            {
                res_j["text"] = curr;
                res.set_content(res_j.dump(), "application/json");
            }
            else
            {
                res.set_chunked_content_provider(
                    "application/json",
                    [&llm](size_t offset, httplib::DataSink& sink) {
                        std::string next;
                        if (llm->next(&next))
                        {
                            std::cout << "writing " << next << std::endl;
                            sink.write(next.data(), next.size());
                        }
                        else
                        {
                            sink.done();
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
