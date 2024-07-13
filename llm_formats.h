#pragma once

#include <string>
#include <sstream>

#include <nlohmann/json.hpp>

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

    oss << "<|start_header_id|>assistant<|end_header_id|>\n\n";
    return oss.str();
}


