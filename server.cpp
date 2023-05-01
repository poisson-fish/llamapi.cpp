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

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <signal.h>
#endif

#if defined(_WIN32)
#pragma comment(lib, "kernel32.lib")
extern "C" __declspec(dllimport) void *__stdcall GetStdHandle(
    unsigned long nStdHandle);
extern "C" __declspec(dllimport) int __stdcall GetConsoleMode(
    void *hConsoleHandle, unsigned long *lpMode);
extern "C" __declspec(dllimport) int __stdcall SetConsoleMode(
    void *hConsoleHandle, unsigned long dwMode);
extern "C" __declspec(dllimport) int __stdcall SetConsoleCP(
    unsigned int wCodePageID);
extern "C" __declspec(dllimport) int __stdcall SetConsoleOutputCP(
    unsigned int wCodePageID);
#endif

static bool is_interacting = false;

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) ||          \
    defined(_WIN32)
void sigint_handler(int signo) {
  printf("\n"); // this also force flush stdout.
  if (signo == SIGINT) {
    if (!is_interacting) {
      is_interacting = true;
    } else {
      _exit(130);
    }
  }
}
#endif

#if defined(_WIN32)
void win32_console_init(void) {
  unsigned long dwMode = 0;
  void *hConOut = GetStdHandle((unsigned long)-11); // STD_OUTPUT_HANDLE (-11)
  if (!hConOut || hConOut == (void *)-1 || !GetConsoleMode(hConOut, &dwMode)) {
    hConOut = GetStdHandle((unsigned long)-12); // STD_ERROR_HANDLE (-12)
    if (hConOut &&
        (hConOut == (void *)-1 || !GetConsoleMode(hConOut, &dwMode))) {
      hConOut = 0;
    }
  }
  if (hConOut) {
    // Enable ANSI colors on Windows 10+
    if (con_use_color && !(dwMode & 0x4)) {
      SetConsoleMode(hConOut,
                     dwMode | 0x4); // ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x4)
    }
    // Set console output codepage to UTF8
    SetConsoleOutputCP(65001); // CP_UTF8
  }
  void *hConIn = GetStdHandle((unsigned long)-10); // STD_INPUT_HANDLE (-10)
  if (hConIn && hConIn != (void *)-1 && GetConsoleMode(hConIn, &dwMode)) {
    // Set console input codepage to UTF8
    SetConsoleCP(65001); // CP_UTF8
  }
}
#endif
#include "httplib.h"
#include <fstream>
#include <regex>

// #include "lib/jsoncpp/include/json/json.h"
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#include "base64.h"
#include <cstdio>
#include <shared_mutex>

struct ModelData {
  llama_context *ctx;
  gpt_params params;
};

std::string queryModel(ModelData &data) {

  // Add a space in front of the first character to match OG llama tokenizer
  // behavior
  data.params.prompt.insert(0, 1, ' ');

  std::string path_session = data.params.path_session;
  std::vector<llama_token> session_tokens;

  if (!path_session.empty()) {
    fprintf(stderr, "%s: attempting to load saved session from %s..\n",
            __func__, path_session.c_str());

    // REVIEW - fopen to check for existing session
    FILE *fp = std::fopen(path_session.c_str(), "rb");
    if (fp != NULL) {
      std::fclose(fp);

      session_tokens.resize(data.params.n_ctx);
      size_t n_token_count_out = 0;
      const size_t n_session_bytes = llama_load_session_file(
          data.ctx, path_session.c_str(), session_tokens.data(),
          session_tokens.capacity(), &n_token_count_out);
      session_tokens.resize(n_token_count_out);

      if (n_session_bytes > 0) {
        fprintf(stderr, "%s: loaded %zu bytes of session data!\n", __func__,
                n_session_bytes);
      } else {
        fprintf(stderr, "%s: could not load session file, will recreate\n",
                __func__);
      }
    } else {
      fprintf(stderr, "%s: session file does not exist, will create\n",
              __func__);
    }
  }

  // tokenize the prompt
  auto embd_inp = ::llama_tokenize(data.ctx, data.params.prompt, true);

  const int n_ctx = llama_n_ctx(data.ctx);

  if ((int)embd_inp.size() > n_ctx - 4) {
    fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n",
            __func__, (int)embd_inp.size(), n_ctx - 4);
    return "<prompt too long>";
  }

  // debug message about similarity of saved session, if applicable
  size_t n_matching_session_tokens = 0;
  if (session_tokens.size()) {
    for (llama_token id : session_tokens) {
      if (n_matching_session_tokens >= embd_inp.size() ||
          id != embd_inp[n_matching_session_tokens]) {
        break;
      }
      n_matching_session_tokens++;
    }
    if (n_matching_session_tokens >= embd_inp.size()) {
      fprintf(stderr, "%s: session file has exact match for prompt!\n",
              __func__);
    } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
      fprintf(stderr,
              "%s: warning: session file has low similarity to prompt (%zu / "
              "%zu tokens); will mostly be reevaluated\n",
              __func__, n_matching_session_tokens, embd_inp.size());
    } else {
      fprintf(stderr, "%s: session file matches %zu / %zu tokens of prompt\n",
              __func__, n_matching_session_tokens, embd_inp.size());
    }
  }

  // number of tokens to keep when resetting context
  if (data.params.n_keep < 0 || data.params.n_keep > (int)embd_inp.size() ||
      data.params.instruct) {
    data.params.n_keep = (int)embd_inp.size();
  }

  // prefix & suffix for instruct mode
  const auto inp_pfx =
      ::llama_tokenize(data.ctx, "\n\n### Instruction:\n\n", true);
  const auto inp_sfx =
      ::llama_tokenize(data.ctx, "\n\n### Response:\n\n", false);

  // in instruct mode, we inject a prefix and a suffix to each input by the user
  if (data.params.instruct) {
    data.params.interactive_first = true;
    data.params.antiprompt.push_back("### Instruction:\n\n");
  }

  // enable interactive mode if reverse prompt or interactive start is specified
  if (data.params.antiprompt.size() != 0 || data.params.interactive_first) {
    data.params.interactive = true;
  }

  // determine newline token
  auto llama_token_newline = ::llama_tokenize(data.ctx, "\n", false);

  if (data.params.verbose_prompt) {
    fprintf(stderr, "\n");
    fprintf(stderr, "%s: prompt: '%s'\n", __func__, data.params.prompt.c_str());
    fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__,
            embd_inp.size());
    for (int i = 0; i < (int)embd_inp.size(); i++) {
      fprintf(stderr, "%6d -> '%s'\n", embd_inp[i],
              llama_token_to_str(data.ctx, embd_inp[i]));
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

  if (data.params.interactive) {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = sigint_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
#elif defined(_WIN32)
    signal(SIGINT, sigint_handler);
#endif

    fprintf(stderr, "%s: interactive mode on.\n", __func__);

    if (data.params.antiprompt.size()) {
      for (auto antiprompt : data.params.antiprompt) {
        fprintf(stderr, "Reverse prompt: '%s'\n", antiprompt.c_str());
      }
    }

    if (!data.params.input_prefix.empty()) {
      fprintf(stderr, "Input prefix: '%s'\n", data.params.input_prefix.c_str());
    }
  }
  fprintf(stderr,
          "sampling: repeat_last_n = %d, repeat_penalty = %f, presence_penalty "
          "= %f, frequency_penalty = %f, top_k = %d, tfs_z = %f, top_p = %f, "
          "typical_p = %f, temp = %f, mirostat = %d, mirostat_lr = %f, "
          "mirostat_ent = %f\n",
          data.params.repeat_last_n, data.params.repeat_penalty,
          data.params.presence_penalty, data.params.frequency_penalty,
          data.params.top_k, data.params.tfs_z, data.params.top_p,
          data.params.typical_p, data.params.temp, data.params.mirostat,
          data.params.mirostat_eta, data.params.mirostat_tau);
  fprintf(stderr,
          "generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n",
          n_ctx, data.params.n_batch, data.params.n_predict,
          data.params.n_keep);
  fprintf(stderr, "\n\n");

  // TODO: replace with ring-buffer
  std::vector<llama_token> last_n_tokens(n_ctx);
  std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

  if (data.params.interactive) {
    fprintf(
        stderr,
        "== Running in interactive mode. ==\n"
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) ||          \
    defined(_WIN32)
        " - Press Ctrl+C to interject at any time.\n"
#endif
        " - Press Return to return control to LLaMa.\n"
        " - If you want to submit another line, end your input in '\\'.\n\n");
    is_interacting = data.params.interactive_first;
  }

  bool is_antiprompt = false;
  bool input_noecho = false;

  // HACK - because session saving incurs a non-negligible delay, for now skip
  // re-saving session if we loaded a session with at least 75% similarity. It's
  // currently just used to speed up the initial prompt so it doesn't need to be
  // an exact match.
  bool need_to_save_session =
      !path_session.empty() &&
      n_matching_session_tokens < (embd_inp.size() * 3 / 4);

  int n_past = 0;
  int n_remain = data.params.n_predict;
  int n_consumed = 0;
  int n_session_consumed = 0;
  std::string result = "";
  std::vector<llama_token> embd;

  while (n_remain != 0 || data.params.interactive) {
    // predict
    if (embd.size() > 0) {
      // infinite text generation via context swapping
      // if we run out of context:
      // - take the n_keep first tokens from the original prompt (via n_past)
      // - take half of the last (n_ctx - n_keep) tokens and recompute the
      // logits in batches
      if (n_past + (int)embd.size() > n_ctx) {
        const int n_left = n_past - data.params.n_keep;

        n_past = data.params.n_keep;

        // insert n_left/2 tokens at the start of embd from last_n_tokens
        embd.insert(embd.begin(),
                    last_n_tokens.begin() + n_ctx - n_left / 2 - embd.size(),
                    last_n_tokens.end() - embd.size());

        // REVIEW - stop saving session if we run out of context
        path_session = "";

        // printf("\n---\n");
        // printf("resetting: '");
        // for (int i = 0; i < (int) embd.size(); i++) {
        //     printf("%s", llama_token_to_str(ctx, embd[i]));
        // }
        // printf("'\n");
        // printf("\n---\n");
      }

      // try to reuse a matching prefix from the loaded session instead of
      // re-eval (via n_past) REVIEW
      if (n_session_consumed < (int)session_tokens.size()) {
        size_t i = 0;
        for (; i < embd.size(); i++) {
          if (embd[i] != session_tokens[n_session_consumed]) {
            session_tokens.resize(n_session_consumed);
            break;
          }

          n_past++;
          n_session_consumed++;

          if (n_session_consumed >= (int)session_tokens.size()) {
            break;
          }
        }
        if (i > 0) {
          embd.erase(embd.begin(), embd.begin() + i);
        }
      }

      // evaluate tokens in batches
      // embd is typically prepared beforehand to fit within a batch, but not
      // always
      for (int i = 0; i < (int)embd.size(); i += data.params.n_batch) {
        int n_eval = (int)embd.size() - i;
        if (n_eval > data.params.n_batch) {
          n_eval = data.params.n_batch;
        }
        if (llama_eval(data.ctx, &embd[i], n_eval, n_past,
                       data.params.n_threads)) {
          fprintf(stderr, "%s : failed to eval\n", __func__);
          return "";
        }
        n_past += n_eval;
      }

      if (embd.size() > 0 && !path_session.empty()) {
        session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
        n_session_consumed = session_tokens.size();
      }
    }

    embd.clear();

    if ((int)embd_inp.size() <= n_consumed && !is_interacting) {
      // out of user input, sample next token
      const float temp = data.params.temp;
      const int32_t top_k =
          data.params.top_k <= 0 ? llama_n_vocab(data.ctx) : data.params.top_k;
      const float top_p = data.params.top_p;
      const float tfs_z = data.params.tfs_z;
      const float typical_p = data.params.typical_p;
      const int32_t repeat_last_n =
          data.params.repeat_last_n < 0 ? n_ctx : data.params.repeat_last_n;
      const float repeat_penalty = data.params.repeat_penalty;
      const float alpha_presence = data.params.presence_penalty;
      const float alpha_frequency = data.params.frequency_penalty;
      const int mirostat = data.params.mirostat;
      const float mirostat_tau = data.params.mirostat_tau;
      const float mirostat_eta = data.params.mirostat_eta;
      const bool penalize_nl = data.params.penalize_nl;

      // optionally save the session on first sample (for faster prompt loading
      // next time)
      if (!path_session.empty() && need_to_save_session) {
        need_to_save_session = false;
        llama_save_session_file(data.ctx, path_session.c_str(),
                                session_tokens.data(), session_tokens.size());
      }

      llama_token id = 0;

      {
        auto logits = llama_get_logits(data.ctx);
        auto n_vocab = llama_n_vocab(data.ctx);

        // Apply params.logit_bias map
        for (auto it = data.params.logit_bias.begin();
             it != data.params.logit_bias.end(); it++) {
          logits[it->first] += it->second;
        }

        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
          candidates.emplace_back(
              llama_token_data{token_id, logits[token_id], 0.0f});
        }

        llama_token_data_array candidates_p = {candidates.data(),
                                               candidates.size(), false};

        // Apply penalties
        float nl_logit = logits[llama_token_nl()];
        auto last_n_repeat =
            std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
        llama_sample_repetition_penalty(
            data.ctx, &candidates_p,
            last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
            last_n_repeat, repeat_penalty);
        llama_sample_frequency_and_presence_penalties(
            data.ctx, &candidates_p,
            last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
            last_n_repeat, alpha_frequency, alpha_presence);
        if (!penalize_nl) {
          logits[llama_token_nl()] = nl_logit;
        }

        if (temp <= 0) {
          // Greedy sampling
          id = llama_sample_token_greedy(data.ctx, &candidates_p);
        } else {
          if (mirostat == 1) {
            static float mirostat_mu = 2.0f * mirostat_tau;
            const int mirostat_m = 100;
            llama_sample_temperature(data.ctx, &candidates_p, temp);
            id = llama_sample_token_mirostat(data.ctx, &candidates_p,
                                             mirostat_tau, mirostat_eta,
                                             mirostat_m, &mirostat_mu);
          } else if (mirostat == 2) {
            static float mirostat_mu = 2.0f * mirostat_tau;
            llama_sample_temperature(data.ctx, &candidates_p, temp);
            id = llama_sample_token_mirostat_v2(data.ctx, &candidates_p,
                                                mirostat_tau, mirostat_eta,
                                                &mirostat_mu);
          } else {
            // Temperature sampling
            llama_sample_top_k(data.ctx, &candidates_p, top_k);
            llama_sample_tail_free(data.ctx, &candidates_p, tfs_z);
            llama_sample_typical(data.ctx, &candidates_p, typical_p);
            llama_sample_top_p(data.ctx, &candidates_p, top_p);
            llama_sample_temperature(data.ctx, &candidates_p, temp);
            id = llama_sample_token(data.ctx, &candidates_p);
          }
        }
        // printf("`%d`", candidates_p.size);

        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(id);
      }

      // replace end of text token with newline token when in interactive mode
      /*if (id == llama_token_eos() && data.params.interactive &&
      !data.params.instruct) { id = llama_token_newline.front(); if
      (data.params.antiprompt.size() != 0) {
          // tokenize and inject first reverse prompt
          const auto first_antiprompt =
              ::llama_tokenize(data.ctx, data.params.antiprompt.front(), false);
          embd_inp.insert(embd_inp.end(), first_antiprompt.begin(),
                          first_antiprompt.end());
        }
      }*/

      // add it to the context
      embd.push_back(id);

      // echo this to console
      input_noecho = false;

      // decrement remaining sampling budget
      --n_remain;
    } else {
      // some user input remains from prompt or interaction, forward it to
      // processing
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

    // in interactive mode, and not currently processing queued inputs;
    // check if we should prompt the user for more
    if (/*data.params.interactive && */ (int)embd_inp.size() <= n_consumed) {

      // check for reverse prompt
      if (data.params.antiprompt.size()) {
        std::string last_output;
        for (auto id : last_n_tokens) {
          last_output += llama_token_to_str(data.ctx, id);
        }

        is_antiprompt = false;
        // Check if each of the reverse prompts appears at the end of the
        // output.
        for (std::string &antiprompt : data.params.antiprompt) {
          if (last_output.find(antiprompt.c_str(),
                               last_output.length() - antiprompt.length(),
                               antiprompt.length()) != std::string::npos) {
            is_interacting = true;
            is_antiprompt = true;
            llama_print_timings(data.ctx);
            fflush(stdout);
            return result.substr(0, result.find(antiprompt.c_str()));
          }
        }
      }

      if (n_past > 0 && is_interacting) {
        // potentially set color to indicate we are taking user input
        // set_console_color(data.con_st, CONSOLE_COLOR_USER_INPUT);

#if defined(_WIN32)
        // Windows: must reactivate sigint handler after each signal
        signal(SIGINT, sigint_handler);
#endif

        if (data.params.instruct) {
          printf("\n> ");
        }

        std::string buffer;
        if (!data.params.input_prefix.empty()) {
          buffer += data.params.input_prefix;
          printf("%s", buffer.c_str());
        }

        std::string line;
        bool another_line = true;
        do {
#if defined(_WIN32)
          std::wstring wline;
          if (!std::getline(std::wcin, wline)) {
            // input stream is bad or EOF received
            return 0;
          }
          win32_utf8_encode(wline, line);
#else
          if (!std::getline(std::cin, line)) {
            // input stream is bad or EOF received
            return 0;
          }
#endif
          if (line.empty() || line.back() != '\\') {
            another_line = false;
          } else {
            line.pop_back(); // Remove the continue character
          }
          buffer += line + '\n'; // Append the line to the result
        } while (another_line);

        // done taking input, reset color
        // set_console_color(con_st, CONSOLE_COLOR_DEFAULT);

        // Add tokens to embd only if the input buffer is non-empty
        // Entering a empty line lets the user pass control back
        if (buffer.length() > 1) {

          // instruct mode: insert instruction prefix
          if (data.params.instruct && !is_antiprompt) {
            n_consumed = embd_inp.size();
            embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());
          }

          auto line_inp = ::llama_tokenize(data.ctx, buffer, false);
          embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

          // instruct mode: insert response suffix
          if (data.params.instruct) {
            embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
          }

          n_remain -= line_inp.size();
        }

        input_noecho = true; // do not echo this again
      }

      if (n_past > 0) {
        is_interacting = false;
      }
    }

    // end of text token
    if (!embd.empty() && embd.back() == llama_token_eos()) {
      if (data.params.instruct) {
        is_interacting = true;
      } else {
        fprintf(stderr, " [end of text]\n");
        llama_print_timings(data.ctx);
        return result;
        break;
      }
    }

    // In interactive mode, respect the maximum number of tokens and drop back
    // to user input when reached.
    if (data.params.interactive && n_remain <= 0 &&
        data.params.n_predict != -1) {
      n_remain = data.params.n_predict;
      is_interacting = true;
    }
  }
  return result;
}

int main(int argc, char **argv) {
  gpt_params params;

  params.temp = 0.6f;
  params.top_p = 0.98f;
  params.n_ctx = 220;
  params.model = "models/30B/ggml-model-q4_0.bin";
  params.n_threads = 18;
  params.repeat_last_n = 64;
  params.repeat_penalty = 1.15;
  params.top_k = 40;
  params.n_batch = 32;
  params.n_predict = 75;
  params.use_mlock = false;
  params.interactive = false;
  params.instruct = false;

  if (gpt_params_parse(argc, argv, params) == false) {
    return 1;
  }

#if defined(_WIN32)
  win32_console_init();
#endif

  if (params.perplexity) {
    printf("\n************\n");
    printf("%s: please use the 'perplexity' tool for perplexity calculations\n",
           __func__);
    printf("************\n\n");

    return 0;
  }

  if (params.embedding) {
    printf("\n************\n");
    printf("%s: please use the 'embedding' tool for embedding calculations\n",
           __func__);
    printf("************\n\n");

    return 0;
  }

  if (params.n_ctx > 2048) {
    fprintf(stderr,
            "%s: warning: model does not support context sizes greater than "
            "2048 tokens (%d specified);"
            "expect poor results\n",
            __func__, params.n_ctx);
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
  // bool is_prime(int n) {)";

  llama_context *ctx;

  // load the model
  {
    auto lparams = llama_context_default_params();

    lparams.n_ctx = params.n_ctx;
    lparams.n_parts = params.n_parts;
    lparams.seed = params.seed;
    lparams.f16_kv = params.memory_f16;
    lparams.use_mmap = params.use_mmap;
    lparams.use_mlock = params.use_mlock;

    ctx = llama_init_from_file(params.model.c_str(), lparams);

    if (ctx == NULL) {
      fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__,
              params.model.c_str());
      return 1;
    }
  }

  if (!params.lora_adapter.empty()) {
    int err = llama_apply_lora_from_file(
        ctx, params.lora_adapter.c_str(),
        params.lora_base.empty() ? NULL : params.lora_base.c_str(),
        params.n_threads);
    if (err != 0) {
      fprintf(stderr, "%s: error: failed to apply lora adapter\n", __func__);
      return 1;
    }
  }

  // print system information
  {
    fprintf(stderr, "\n");
    fprintf(stderr, "system_info: n_threads = %d / %d | %s\n", params.n_threads,
            std::thread::hardware_concurrency(), llama_print_system_info());
  }

  // determine the maximum memory usage needed to do inference for the given
  // n_batch and n_predict parameters uncomment the "used_mem" line in llama.cpp
  // to see the results
  if (params.mem_test) {
    {
      const std::vector<llama_token> tmp(params.n_batch, 0);
      llama_eval(ctx, tmp.data(), tmp.size(), 0, params.n_threads);
    }

    {
      const std::vector<llama_token> tmp = {
          0,
      };
      llama_eval(ctx, tmp.data(), tmp.size(), params.n_predict - 1,
                 params.n_threads);
    }

    llama_print_timings(ctx);
    llama_free(ctx);

    return 0;
  }

  ModelData model_state = ModelData{ctx, params};
  // model state mutex
  std::shared_mutex model_state_mutex;

  // HTTP
  httplib::Server svr;
  svr.set_keep_alive_max_count(2);           // Default is 5
  svr.set_keep_alive_timeout(6000000);       // Default is 5
  svr.set_read_timeout(600000, 600000);      // 600 seconds
  svr.set_write_timeout(6000000, 600000);    // 600 seconds
  svr.set_idle_interval(6000000, 600000000); // 100 milliseconds
  svr.set_tcp_nodelay(true);
  svr.set_exception_handler(
      [](const auto &req, auto &res, std::exception_ptr ep) {
        auto fmt = "<h1>Error 500</h1><p>%s</p>";
        char buf[BUFSIZ];
        try {
          std::rethrow_exception(ep);
        } catch (std::exception &e) {
          snprintf(buf, sizeof(buf), fmt, e.what());
        } catch (...) { // See the following NOTE
          snprintf(buf, sizeof(buf), fmt, "Unknown Exception");
        }
        res.set_content(buf, "text/html");
        res.status = 500;
      });

  svr.Post("/generate", [&](const httplib::Request &req,
                            httplib::Response &res) {
    std::lock_guard<std::shared_mutex> guard(model_state_mutex);

    const auto request_body = req.body;
    json root = json::parse(request_body);
    std::string errors;

    if (!(root.contains("prompt") && root.contains("user") &&
          root.contains("contextData"))) {
      // fprintf(stderr, "Json lacked the required fields user  and prompt.");
      res.set_content("{\"error\":\"JSON input malformed\"}",
                      "application/json");
      return;
    }

    // fprintf(stderr, std::format("Request: {}\n", root.asString()).c_str());

    const auto promptInput = root["prompt"].get<std::string>();
    const auto userInput = root["user"].get<std::string>();
    const auto unBase64dPrompt =
        std::string(base64::decode(promptInput.c_str(), promptInput.length()));
    const auto unBase64dUser =
        std::string(base64::decode(userInput.c_str(), userInput.length()));

    const auto ctxData = root["contextData"];
    const auto chatHistory = ctxData["ChatHistory"].get<std::string>();
    const auto unBase64dChatHistory =
        std::string(base64::decode(chatHistory.c_str(), chatHistory.length()));
    const auto dateStr = ctxData["DateString"].get<std::string>();
    const auto timeStr = ctxData["TimeString"].get<std::string>();
    const auto timeZoneStr = ctxData["TimeZoneString"].get<std::string>();

    fprintf(stderr, "\nprompt: %s", unBase64dPrompt.c_str());
    fprintf(stderr, "\nuser: %s", unBase64dUser.c_str());
    fprintf(stderr, "\ndate: %s", dateStr.c_str());
    fprintf(stderr, "\ntime: %s", timeStr.c_str());
    fprintf(stderr, "\ntimezone: %s", timeZoneStr.c_str());
    fprintf(stderr, "\nhistorystr: \n%s\n", unBase64dChatHistory.c_str());

    std::string soft_prompt =
        std::regex_replace(model_state.params.soft_prompt,
                           std::regex("<PROMPT>"), unBase64dPrompt);
    soft_prompt =
        std::regex_replace(soft_prompt, std::regex("<USER>"), unBase64dUser);
    soft_prompt = std::regex_replace(soft_prompt, std::regex("<HISTORY>"),
                                     unBase64dChatHistory);
    soft_prompt =
        std::regex_replace(soft_prompt, std::regex("<DATE>"), dateStr);
    soft_prompt =
        std::regex_replace(soft_prompt, std::regex("<TIME>"), timeStr);
    soft_prompt =
        std::regex_replace(soft_prompt, std::regex("<TIMEZONE>"), timeZoneStr);

    model_state.params.prompt = soft_prompt;
    fprintf(stderr, "\nprompt: \n%s\n", soft_prompt.c_str());

    json result;
    const std::string queryResult = queryModel(model_state);
    result["result"] = queryResult;
    res.body = result.dump();
  });

  fprintf(stderr, "API listening on 0.0.0.0:8080\n");
  svr.listen("0.0.0.0", 8080);

#if defined(_WIN32)
  signal(SIGINT, SIG_DFL);
#endif

  llama_print_timings(ctx);
  llama_free(ctx);

  return 0;
}
