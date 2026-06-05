#include "llama_bridge.h"

#include "json-schema-to-grammar.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

static char * bridge_strdup(const std::string & s) {
    char * out = static_cast<char *>(std::malloc(s.size() + 1));
    if (!out) {
        return nullptr;
    }
    std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
}

extern "C" bridge_schema_grammar_result bridge_json_schema_to_grammar(const char * schema_json) {
    bridge_schema_grammar_result result = {};
    if (!schema_json) {
        result.error = bridge_strdup("schema JSON is required");
        return result;
    }
    try {
        auto schema = nlohmann::ordered_json::parse(schema_json);
        result.grammar = bridge_strdup(json_schema_to_grammar(schema, true));
        if (!result.grammar) {
            result.error = bridge_strdup("failed to allocate schema grammar");
        }
        return result;
    } catch (const std::exception & e) {
        result.error = bridge_strdup(e.what());
        return result;
    } catch (...) {
        result.error = bridge_strdup("unknown schema grammar conversion error");
        return result;
    }
}

extern "C" void bridge_schema_grammar_result_free(bridge_schema_grammar_result result) {
    std::free(result.grammar);
    std::free(result.error);
}
