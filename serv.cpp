#include <nlohmann/json.hpp>
#include <httplib.h>

void update_prompt(std::string s)
{
}

void generate()
{
}

struct value_parser
{
    template<typename value_t>
    static void parse(const char * value, value_t & field)
    {
        std::istringstream iss(value);
        iss >> field;
    }
};

template<>
void value_parser::parse<std::string>(const char * value, std::string & field)
{
    field = value;
}

template<typename config_t>
struct parser
{
    int parse_options(int argc, char ** argv, config_t & conf)
    {
        for (int i = 1; i < argc; i++)
        {
            std::string key(argv[i]);
            auto it = setters_.find(key);
            if (it != setters_.end())
            {
                if (++i < argc)
                {
                    it->second(argv[i], conf);
                }
                else
                {
                    fprintf(stderr, "No argument value provided for %s\n", argv[i - 1]);
                    return 1;
                }
            }
            else
            {
                fprintf(stderr, "Unknown argument %s\n", argv[i]);
                return 1;
            }
        }
        return 0;
    }

    template<typename T>
    void add_option(const std::string& key, T config_t::* field)
    {
        setters_[key] = [field](const char * value, config_t & conf)
        {
            value_parser::parse(value, conf.*field);
        };
    }

    template<typename T>
    void add_option(const std::initializer_list<std::string>& keys, T config_t::* field)
    {
        for (const auto& key : keys)
        {
            add_option(key, field);
        }
    }

  private:
    std::map<std::string, std::function<void(const char*, config_t&)>> setters_;
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

struct config
{
    std::string addr;          // to listen on. "0.0.0.0" is default. 
    int32_t     port;          // 5555 is default.
};

config gen_config(int argc, char ** argv)
{
    config res = 
    {
        /* addr = */ "0.0.0.0",
        /* port = */ 5555
    };
    parser<config> p;
    // server options
    p.add_option({"--addr"},       &config::addr);
    p.add_option({"--port", "-p"}, &config::port);
    if (0 != p.parse_options(argc, argv, res))
    {
        exit(-1);
    }

    return res;
}

void serve(const config & conf)
{
    using nlohmann::json;
    httplib::Server http_server;

    std::string curr = "";

    http_server.Post("/query", [&curr](const httplib::Request & req, httplib::Response & res)
    {
        try
        {
            auto req_j = json::parse(req.body);
            json res_j;

            size_t offset = req_j["offset"];
            std::string text = req_j["text"];
            bool complete = req_j["complete"];

            if (offset + text.size() > curr.size())
            {
                curr.resize(offset + text.size(), ' ');
            }

            // Insert or override the text at the specified offset
            curr.replace(offset, text.size(), text);

            update_prompt(curr);
            if (!complete)
            {
                res_j["text"] = curr;
                res.set_content(res_j.dump(), "application/json");
            }
            else
            {
                generate();
                res.set_chunked_content_provider(
                    "application/json",
                    [](size_t offset, httplib::DataSink& sink) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(500));

                        std::vector<std::string> chunks = {
                            "{\"part\": 1, \"data\": \"Chunk 1\"}",
                            "{\"part\": 2, \"data\": \"Chunk 2\"}",
                            "{\"part\": 3, \"data\": \"Chunk 3\"}",
                            "{\"part\": 4, \"data\": \"Chunk 4\"}"
                        };

                        if (offset < chunks.size()) {
                            sink.write(chunks[offset].data(), chunks[offset].size());
                            return true; // There's more data to send
                        } else {
                            sink.done();
                            return false; // No more data
                        }
                    }
                );
            }
        }
        catch(const std::exception & e)
        {
            log::error("%s", e.what());
        }
    });

    log::info("starting server on %s:%d\n", conf.addr.c_str(), conf.port);
    http_server.listen(conf.addr, conf.port);
}

int main(int argc, char ** argv)
{
    auto conf = gen_config(argc, argv);

    serve(conf);


    return 0;
}
