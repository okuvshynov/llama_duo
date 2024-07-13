#pragma once

#include <cstdlib>

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

