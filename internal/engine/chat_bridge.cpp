#include "llama_bridge.h"

#include "chat.h"
#include "common.h"
#include "sampling.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <cctype>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static char * chat_bridge_strdup(const std::string & value) {
    char * out = static_cast<char *>(std::malloc(value.size() + 1));
    if (out != nullptr) {
        std::memcpy(out, value.c_str(), value.size() + 1);
    }
    return out;
}

extern "C" bridge_chat_render_result bridge_chat_render_native(
        const struct llama_model * model,
        const char * template_override,
        const char * messages_json,
        const char * tools_json,
        int32_t tool_choice,
        bool parallel_tool_calls,
        bool enable_thinking) {
    bridge_chat_render_result result = {};
    if (model == nullptr || messages_json == nullptr) {
        result.error = chat_bridge_strdup("model and messages are required");
        return result;
    }

    try {
        const auto messages = json::parse(messages_json);
        const auto tools = tools_json != nullptr && tools_json[0] != '\0'
            ? json::parse(tools_json)
            : json::array();

        auto templates = common_chat_templates_init(model, template_override != nullptr ? template_override : "");
        common_chat_templates_inputs inputs;
        inputs.messages = common_chat_msgs_parse_oaicompat(messages);
        inputs.tools = common_chat_tools_parse_oaicompat(tools);
        inputs.tool_choice = static_cast<common_chat_tool_choice>(tool_choice);
        inputs.parallel_tool_calls = parallel_tool_calls;
        inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        inputs.enable_thinking = enable_thinking;
        inputs.add_generation_prompt = true;
        inputs.use_jinja = true;

        const auto params = common_chat_templates_apply(templates.get(), inputs);
        const auto caps = common_chat_templates_get_caps(templates.get());

        json stops = params.additional_stops;
        json triggers = json::array();
        for (const auto & trigger : params.grammar_triggers) {
            triggers.push_back({
                {"type", static_cast<int>(trigger.type)},
                {"value", trigger.value},
                {"token", trigger.token},
            });
        }
        json capabilities = json::object();
        for (const auto & [name, supported] : caps) {
            capabilities[name] = supported;
        }

        result.prompt = chat_bridge_strdup(params.prompt);
        result.grammar = chat_bridge_strdup(params.grammar);
        result.parser = chat_bridge_strdup(params.parser);
        result.generation_prompt = chat_bridge_strdup(params.generation_prompt);
        result.additional_stops_json = chat_bridge_strdup(stops.dump());
        result.grammar_triggers_json = chat_bridge_strdup(triggers.dump());
        result.capabilities_json = chat_bridge_strdup(capabilities.dump());
        result.format = static_cast<int32_t>(params.format);
        result.grammar_lazy = params.grammar_lazy;
        result.supports_thinking = params.supports_thinking;
        return result;
    } catch (const std::exception & error) {
        result.error = chat_bridge_strdup(error.what());
        return result;
    } catch (...) {
        result.error = chat_bridge_strdup("unknown native chat rendering error");
        return result;
    }
}

extern "C" void bridge_chat_render_result_free(bridge_chat_render_result result) {
    std::free(result.prompt);
    std::free(result.grammar);
    std::free(result.parser);
    std::free(result.generation_prompt);
    std::free(result.additional_stops_json);
    std::free(result.grammar_triggers_json);
    std::free(result.capabilities_json);
    std::free(result.error);
}

extern "C" bridge_chat_parse_result bridge_chat_parse_native(
        const char * response,
        const char * parser,
        const char * generation_prompt,
        int32_t format) {
    bridge_chat_parse_result result = {};
    if (response == nullptr) {
        result.error = chat_bridge_strdup("response is required");
        return result;
    }
    try {
        common_chat_parser_params params;
        params.format = static_cast<common_chat_format>(format);
        params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        params.reasoning_in_content = false;
        params.parse_tool_calls = true;
        params.generation_prompt = generation_prompt != nullptr ? generation_prompt : "";
        if (parser != nullptr && parser[0] != '\0') {
            params.parser.load(parser);
        }
        auto message = common_chat_parse(response, false, params);
        result.message_json = chat_bridge_strdup(message.to_json_oaicompat().dump());
        return result;
    } catch (const std::exception & error) {
        result.error = chat_bridge_strdup(error.what());
        return result;
    } catch (...) {
        result.error = chat_bridge_strdup("unknown native chat parsing error");
        return result;
    }
}

extern "C" void bridge_chat_parse_result_free(bridge_chat_parse_result result) {
    std::free(result.message_json);
    std::free(result.error);
}

extern "C" llama_sampler * bridge_chat_grammar_sampler_init(
        const llama_vocab * vocab,
        const char * grammar,
        bool grammar_lazy,
        const char * grammar_triggers_json,
        const char * generation_prompt,
        char ** error) {
    if (error != nullptr) {
        *error = nullptr;
    }
    if (vocab == nullptr || grammar == nullptr || grammar[0] == '\0') {
        if (error != nullptr) {
            *error = chat_bridge_strdup("vocab and grammar are required");
        }
        return nullptr;
    }
    try {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        if (grammar_triggers_json != nullptr && grammar_triggers_json[0] != '\0') {
            for (const auto & trigger : json::parse(grammar_triggers_json)) {
                const auto type = static_cast<common_grammar_trigger_type>(trigger.value("type", 0));
                const auto value = trigger.value("value", std::string());
                switch (type) {
                    case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                        trigger_patterns.push_back(regex_escape(value));
                        break;
                    case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                        trigger_patterns.push_back(value);
                        break;
                    case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL: {
                        std::string anchored = "^$";
                        if (!value.empty()) {
                            anchored = (value.front() != '^' ? "^" : "") + value + (value.back() != '$' ? "$" : "");
                        }
                        trigger_patterns.push_back(std::move(anchored));
                        break;
                    }
                    case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                        trigger_tokens.push_back(trigger.value("token", LLAMA_TOKEN_NULL));
                        break;
                    default:
                        throw std::runtime_error("unknown native grammar trigger type");
                }
            }
        }

        std::vector<const char *> patterns;
        patterns.reserve(trigger_patterns.size());
        for (const auto & pattern : trigger_patterns) {
            patterns.push_back(pattern.c_str());
        }

        llama_sampler * sampler = grammar_lazy
            ? llama_sampler_init_grammar_lazy_patterns(vocab, grammar, "root",
                patterns.data(), patterns.size(), trigger_tokens.data(), trigger_tokens.size())
            : llama_sampler_init_grammar(vocab, grammar, "root");
        if (sampler == nullptr) {
            throw std::runtime_error("invalid native chat grammar");
        }

        if (!grammar_lazy && generation_prompt != nullptr && generation_prompt[0] != '\0') {
            auto tokens = common_tokenize(vocab, generation_prompt, false, true);
            if (!tokens.empty()) {
                const auto first_piece = common_token_to_piece(vocab, tokens.front(), true);
                if (!first_piece.empty() && std::isspace(static_cast<unsigned char>(first_piece.front())) &&
                    !std::isspace(static_cast<unsigned char>(generation_prompt[0]))) {
                    tokens.erase(tokens.begin());
                }
            }
            for (const auto token : tokens) {
                llama_sampler_accept(sampler, token);
            }
        }
        return sampler;
    } catch (const std::exception & cause) {
        if (error != nullptr) {
            *error = chat_bridge_strdup(cause.what());
        }
        return nullptr;
    }
}

extern "C" void bridge_string_free(char * value) {
    std::free(value);
}

extern "C" common_sampler * bridge_common_sampler_init(
        const llama_model * model,
        const char * options_json,
        const char * grammar,
        bool grammar_lazy,
        const char * grammar_triggers_json,
        const char * generation_prompt,
        char ** error) {
    if (error != nullptr) {
        *error = nullptr;
    }
    try {
        const auto options = options_json != nullptr && options_json[0] != '\0'
            ? json::parse(options_json)
            : json::object();
        common_params_sampling params;
        params.seed = options.value("seed", LLAMA_DEFAULT_SEED);
        params.top_k = options.value("top_k", 40);
        params.top_p = options.value("top_p", 0.9f);
        params.min_p = options.value("min_p", 0.05f);
        params.typ_p = options.value("typical_p", 1.0f);
        params.temp = options.value("temperature", 0.7f);
        params.penalty_repeat = options.value("repeat_penalty", 1.0f);
        params.penalty_last_n = options.value("repeat_last_n", 64);
        params.penalty_freq = options.value("frequency_penalty", 0.0f);
        params.penalty_present = options.value("presence_penalty", 0.0f);
        params.mirostat = options.value("mirostat", 0);
        params.mirostat_tau = options.value("mirostat_tau", 5.0f);
        params.mirostat_eta = options.value("mirostat_eta", 0.1f);
        params.grammar = {COMMON_GRAMMAR_TYPE_TOOL_CALLS, grammar != nullptr ? grammar : ""};
        params.grammar_lazy = grammar_lazy;
        params.generation_prompt = generation_prompt != nullptr ? generation_prompt : "";

        if (grammar_triggers_json != nullptr && grammar_triggers_json[0] != '\0') {
            for (const auto & trigger : json::parse(grammar_triggers_json)) {
                common_grammar_trigger parsed;
                parsed.type = static_cast<common_grammar_trigger_type>(trigger.value("type", 0));
                parsed.value = trigger.value("value", std::string());
                parsed.token = trigger.value("token", LLAMA_TOKEN_NULL);
                params.grammar_triggers.push_back(std::move(parsed));
            }
        }
        auto * sampler = common_sampler_init(model, params);
        if (sampler == nullptr) {
            throw std::runtime_error("failed to initialize llama.cpp common sampler");
        }
        return sampler;
    } catch (const std::exception & cause) {
        if (error != nullptr) {
            *error = chat_bridge_strdup(cause.what());
        }
        return nullptr;
    }
}

extern "C" llama_token bridge_common_sampler_sample(common_sampler * sampler, llama_context * ctx, int32_t idx) {
    return common_sampler_sample(sampler, ctx, idx, true);
}

extern "C" void bridge_common_sampler_accept(common_sampler * sampler, llama_token token) {
    common_sampler_accept(sampler, token, true);
}

extern "C" void bridge_common_sampler_free(common_sampler * sampler) {
    common_sampler_free(sampler);
}
