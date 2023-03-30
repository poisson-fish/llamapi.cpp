#include "common.h"
#include "llama.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#include <signal.h>
#endif

#if defined (_WIN32)
#pragma comment(lib,"kernel32.lib")
extern "C" __declspec(dllimport) void* __stdcall GetStdHandle(unsigned long nStdHandle);
extern "C" __declspec(dllimport) int __stdcall GetConsoleMode(void* hConsoleHandle, unsigned long* lpMode);
extern "C" __declspec(dllimport) int __stdcall SetConsoleMode(void* hConsoleHandle, unsigned long dwMode);
extern "C" __declspec(dllimport) int __stdcall SetConsoleCP(unsigned int wCodePageID);
extern "C" __declspec(dllimport) int __stdcall SetConsoleOutputCP(unsigned int wCodePageID);
#endif

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_BOLD          "\x1b[1m"

/* Keep track of current color of output, and emit ANSI code if it changes. */
enum console_state {
    CONSOLE_STATE_DEFAULT=0,
    CONSOLE_STATE_PROMPT,
    CONSOLE_STATE_USER_INPUT
};

static console_state con_st = CONSOLE_STATE_DEFAULT;
static bool con_use_color = false;

void set_console_state(console_state new_st) {
    if (!con_use_color) return;
    // only emit color code if state changed
    if (new_st != con_st) {
        con_st = new_st;
        switch(con_st) {
        case CONSOLE_STATE_DEFAULT:
            printf(ANSI_COLOR_RESET);
            return;
        case CONSOLE_STATE_PROMPT:
            printf(ANSI_COLOR_YELLOW);
            return;
        case CONSOLE_STATE_USER_INPUT:
            printf(ANSI_BOLD ANSI_COLOR_GREEN);
            return;
        }
    }
}

static bool is_interacting = false;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    set_console_state(CONSOLE_STATE_DEFAULT);
    printf("\n"); // this also force flush stdout.
    if (signo == SIGINT) {
        if (!is_interacting) {
            is_interacting=true;
        } else {
            _exit(130);
        }
    }
}
#endif

#if defined (_WIN32)
void win32_console_init(void) {
    unsigned long dwMode = 0;
    void* hConOut = GetStdHandle((unsigned long)-11); // STD_OUTPUT_HANDLE (-11)
    if (!hConOut || hConOut == (void*)-1 || !GetConsoleMode(hConOut, &dwMode)) {
        hConOut = GetStdHandle((unsigned long)-12); // STD_ERROR_HANDLE (-12)
        if (hConOut && (hConOut == (void*)-1 || !GetConsoleMode(hConOut, &dwMode))) {
            hConOut = 0;
        }
    }
    if (hConOut) {
        // Enable ANSI colors on Windows 10+
        if (con_use_color && !(dwMode & 0x4)) {
            SetConsoleMode(hConOut, dwMode | 0x4); // ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x4)
        }
        // Set console output codepage to UTF8
        SetConsoleOutputCP(65001); // CP_UTF8
    }
    void* hConIn = GetStdHandle((unsigned long)-10); // STD_INPUT_HANDLE (-10)
    if (hConIn && hConIn != (void*)-1 && GetConsoleMode(hConIn, &dwMode)) {
        // Set console input codepage to UTF8
        SetConsoleCP(65001); // CP_UTF8
    }
}
#endif
#include <fstream>
#include <regex>
#include "httplib.h"
//#include "lib/jsoncpp/include/json/json.h"
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#include "base64.h"
#include <cstdio>
#include <shared_mutex>

struct ModelData {
    llama_context* ctx;
    gpt_params params;
};

std::string queryModel(ModelData& data) {
    // Add a space in front of the first character to match OG llama tokenizer behavior
   // params.prompt.insert(0, 1, ' ');

   // tokenize the prompt
    auto embd_inp = ::llama_tokenize(data.ctx, data.params.prompt, true);
    std::string result = "";
    const int n_ctx = llama_n_ctx(data.ctx);

    if ((int)embd_inp.size() > n_ctx - 4) {
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int)embd_inp.size(), n_ctx - 4);
        return "<prompt too long>";
    }

    data.params.n_keep = std::min(data.params.n_keep, (int)embd_inp.size());


    // determine newline token
    auto llama_token_newline = ::llama_tokenize(data.ctx, "\n", false);

    if (data.params.verbose_prompt) {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: prompt: '%s'\n", __func__, data.params.prompt.c_str());
        fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int)embd_inp.size(); i++) {
            fprintf(stderr, "%6d -> '%s'\n", embd_inp[i], llama_token_to_str(data.ctx, embd_inp[i]));
        }
        if (data.params.n_keep > 0) {
            fprintf(stderr, "%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < data.params.n_keep; i++) {
                fprintf(stderr, "%s", llama_token_to_str(data.ctx, embd_inp[i]));
            }
            fprintf(stderr, "'\n");
        }
        fprintf(stderr, "\n");
    }

    fprintf(stderr, "sampling: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n", data.params.temp, data.params.top_k, data.params.top_p, data.params.repeat_last_n, data.params.repeat_penalty);
    fprintf(stderr, "generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, data.params.n_batch, data.params.n_predict, data.params.n_keep);
    fprintf(stderr, "\n\n");

    // TODO: replace with ring-buffer
    std::vector<llama_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    bool input_noecho = false;

    int n_past = 0;
    int n_remain = data.params.n_predict;
    int n_consumed = 0;

    std::vector<llama_token> embd;

    while (n_remain != 0) {
        // predict
        if (embd.size() > 0) {
            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
            if (n_past + (int)embd.size() > n_ctx) {
                const int n_left = n_past - data.params.n_keep;

                n_past = data.params.n_keep;

                // insert n_left/2 tokens at the start of embd from last_n_tokens
                embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left / 2 - embd.size(), last_n_tokens.end() - embd.size());

                //printf("\n---\n");
                //printf("resetting: '");
                //for (int i = 0; i < (int) embd.size(); i++) {
                //    printf("%s", llama_token_to_str(ctx, embd[i]));
                //}
                //printf("'\n");
                //printf("\n---\n");
            }

            if (llama_eval(data.ctx, embd.data(), embd.size(), n_past, data.params.n_threads)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return "unknown error";
            }
        }

        n_past += embd.size();
        embd.clear();

        if ((int)embd_inp.size() <= n_consumed) {
            // out of user input, sample next token
            const float top_k = data.params.top_k;
            const float top_p = data.params.top_p;
            const float temp = data.params.temp;
            const float repeat_penalty = data.params.repeat_penalty;

            llama_token id = 0;

            {
                auto logits = llama_get_logits(data.ctx);

                if (data.params.ignore_eos) {
                    logits[llama_token_eos()] = 0;
                }

                id = llama_sample_top_p_top_k(data.ctx,
                    last_n_tokens.data() + n_ctx - data.params.repeat_last_n,
                    data.params.repeat_last_n, top_k, top_p, temp, repeat_penalty);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
            }

            // replace end of text token with newline token when in interactive mode
            if (id == llama_token_eos() && data.params.interactive && !data.params.instruct) {
                id = llama_token_newline.front();
            }

            // add it to the context
            embd.push_back(id);

            // echo this to console
            input_noecho = false;

            // decrement remaining sampling budget
            --n_remain;
        }
        else {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int)embd_inp.size() > n_consumed) {
                input_noecho = true;
                embd.push_back(embd_inp[n_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int)embd.size() >= data.params.n_batch) {
                    break;
                }
            }
        }

        // display text
        if (!input_noecho) {
            for (auto id : embd) {
                const auto token = llama_token_to_str(data.ctx, id);
                result += token;
                printf("%s", token);
            }
            fflush(stdout);
        }

        // end of text token
        if (embd.back() == llama_token_eos()) {
            if (data.params.instruct) {
                is_interacting = true;
            }
            else {
                return result;
                fprintf(stderr, " [end of text]\n");
                break;
            }
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        if (n_remain <= 0 && data.params.n_predict != -1) {
            n_remain = data.params.n_predict;
            fprintf(stderr, " [end of text]\n");
            return result;
        }
    }

}




int main(int argc, char ** argv) {
    gpt_params params;
    
    params.temp = 0.6f;
    params.top_p = 0.98f;
    params.n_ctx = 220;
    params.model = "models/30B/ggml-model-q4_0.bin";
    params.n_threads = 18;
    params.repeat_last_n = 64;
    params.repeat_penalty = 1.15;
    params.ignore_eos = false;
    params.top_k = 40;
    params.n_batch = 32;
    params.n_predict = 75;
    params.use_mlock = false;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }
    


#if defined (_WIN32)
    win32_console_init();
#endif

    if (params.perplexity) {
        printf("\n************\n");
        printf("%s: please use the 'perplexity' tool for perplexity calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.embedding) {
        printf("\n************\n");
        printf("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.n_ctx > 2048) {
        fprintf(stderr, "%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
                "expect poor results\n", __func__, params.n_ctx);
    }

    if (params.seed <= 0) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

//    params.prompt = R"(// this function checks if the number n is prime
//bool is_prime(int n) {)";

    llama_context * ctx;

    // load the model
    {
        auto lparams = llama_context_default_params();

        lparams.n_ctx      = params.n_ctx;
        lparams.n_parts    = params.n_parts;
        lparams.seed       = params.seed;
        lparams.f16_kv     = params.memory_f16;
        lparams.use_mlock  = params.use_mlock;
        

        ctx = llama_init_from_file(params.model.c_str(), lparams);

        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
            return 1;
        }
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    // determine the maximum memory usage needed to do inference for the given n_batch and n_predict parameters
    // uncomment the "used_mem" line in llama.cpp to see the results
    if (params.mem_test) {
        {
            const std::vector<llama_token> tmp(params.n_batch, 0);
            llama_eval(ctx, tmp.data(), tmp.size(), 0, params.n_threads);
        }

        {
            const std::vector<llama_token> tmp = { 0, };
            llama_eval(ctx, tmp.data(), tmp.size(), params.n_predict - 1, params.n_threads);
        }

        llama_print_timings(ctx);
        llama_free(ctx);

        return 0;
    }

    ModelData model_state = ModelData {
        ctx,
        params
    };
    //model state mutex
    std::shared_mutex model_state_mutex;
   
    // HTTP
    httplib::Server svr;
    svr.set_keep_alive_max_count(2); // Default is 5
    svr.set_keep_alive_timeout(6000000);  // Default is 5
    svr.set_read_timeout(600000, 600000); // 600 seconds
    svr.set_write_timeout(6000000, 600000); // 600 seconds
    svr.set_idle_interval(6000000, 600000000); // 100 milliseconds
    svr.set_tcp_nodelay(true);
    svr.set_exception_handler([](const auto& req, auto& res, std::exception_ptr ep) {
        auto fmt = "<h1>Error 500</h1><p>%s</p>";
        char buf[BUFSIZ];
        try {
            std::rethrow_exception(ep);
        }
        catch (std::exception& e) {
            snprintf(buf, sizeof(buf), fmt, e.what());
        }
        catch (...) { // See the following NOTE
            snprintf(buf, sizeof(buf), fmt, "Unknown Exception");
        }
        res.set_content(buf, "text/html");
        res.status = 500;
        });

    svr.Post("/generate", [&](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::shared_mutex> guard(model_state_mutex);

        const auto request_body = req.body;
        json root = json::parse(request_body);
        std::string errors;

        if (!(root.contains("prompt") && root.contains("user"))) {
            //fprintf(stderr, "Json lacked the required fields user  and prompt.");
            res.set_content("{\"error\":\"JSON input malformed\"}", "application/json");
            return;
        }

        //fprintf(stderr, std::format("Request: {}\n", root.asString()).c_str());

        const auto promptInput = root["prompt"].get<std::string>();
        const auto userInput = root["user"].get<std::string>();
        const auto unBase64dPrompt = std::string(base64::decode(promptInput.c_str(),promptInput.length()));
        const auto unBase64dUser = std::string(base64::decode(userInput.c_str(),promptInput.length()));
        fprintf(stderr, "\nprompt: %s\n", unBase64dPrompt.c_str());
        fprintf(stderr, "user: %s\n", unBase64dUser.c_str());

        model_state.params.prompt = std::regex_replace(
            std::regex_replace(model_state.params.soft_prompt, std::regex("PROMPT"), unBase64dPrompt)
            , std::regex("USER"), unBase64dUser);

        json result;
        const std::string queryResult = queryModel(model_state);
        result["result"] = queryResult;
        res.body = result.dump();

        });

    fprintf(stderr, "API listening on 0.0.0.0:8080\n");
    svr.listen("0.0.0.0", 8080);

#if defined (_WIN32)
    signal(SIGINT, SIG_DFL);
#endif

    llama_print_timings(ctx);
    llama_free(ctx);

    set_console_state(CONSOLE_STATE_DEFAULT);

    return 0;
}
