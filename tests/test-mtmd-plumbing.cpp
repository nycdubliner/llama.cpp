#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

static std::string read_file(const std::string & path) {
    std::ifstream file(path);
    if (!file.good()) {
        std::fprintf(stderr, "failed to open %s\n", path.c_str());
        std::exit(1);
    }

    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

static bool expect(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
    }
    return ok;
}

int main(int argc, char ** argv) {
    bool ok = true;

    ok &= expect(argc == 2, "expected repo root argument");
    if (!ok) {
        return 1;
    }

    const std::string root = argv[1];
    const std::string mtmd_h = read_file(root + "/tools/mtmd/mtmd.h");
    const std::string clip_h = read_file(root + "/tools/mtmd/clip.h");
    const std::string mtmd_cpp = read_file(root + "/tools/mtmd/mtmd.cpp");
    const std::string mtmd_image = read_file(root + "/tools/mtmd/mtmd-image.cpp");
    const std::string clip_cpp = read_file(root + "/tools/mtmd/clip.cpp");
    const std::string mtmd_helper = read_file(root + "/tools/mtmd/mtmd-helper.cpp");
    const std::string server_context = read_file(root + "/tools/server/server-context.cpp");
    const std::string mtmd_cli = read_file(root + "/tools/mtmd/mtmd-cli.cpp");
    const std::string mtmd_debug = read_file(root + "/tools/mtmd/debug/mtmd-debug.cpp");

    ok &= expect(mtmd_h.find("int decoder_n_ubatch") != std::string::npos,
        "mtmd context params must carry the decoder physical ubatch");
    ok &= expect(clip_h.find("int decoder_n_ubatch") != std::string::npos,
        "clip context params must receive the decoder physical ubatch");
    ok &= expect(mtmd_cpp.find("/* decoder_n_ubatch */ 0") != std::string::npos,
        "mtmd default params must leave decoder ubatch unspecified for API callers without a decoder context");
    ok &= expect(mtmd_h.find("struct mtmd_decode_requirements") != std::string::npos &&
                 mtmd_h.find("min_decoder_batch_tokens") != std::string::npos,
        "mtmd must expose a lightweight mmproj decode-requirements query");
    ok &= expect(clip_h.find("struct clip_decode_requirements") != std::string::npos &&
                 clip_h.find("clip_get_decode_requirements") != std::string::npos,
        "clip must expose metadata-only decode requirements for mtmd");
    ok &= expect(mtmd_cpp.find("mtmd_get_decode_requirements_from_file") != std::string::npos,
        "mtmd must wrap clip decode-requirement metadata queries");

    ok &= expect(server_context.find("mparams.decoder_n_ubatch = llama_n_ubatch(ctx)") != std::string::npos ||
                 server_context.find("mparams.decoder_n_ubatch = llama_n_ubatch(ctx_tgt)") != std::string::npos ||
                 server_context.find("mparams.decoder_n_ubatch = ctx_tgt ? llama_n_ubatch(ctx_tgt) : params_base.n_ubatch") != std::string::npos,
        "server mmproj setup must pass the text decoder ubatch to mtmd");
    ok &= expect(server_context.find("has_mmproj && params_base.fit_params && params_base.mmproj_use_gpu && llama_supports_gpu_offload() && !params_base.mmproj_gpu_swap") != std::string::npos,
        "server mmproj memory measurement for fit must only run when mmproj is GPU-offloaded");
    ok &= expect(mtmd_cli.find("mparams.decoder_n_ubatch = llama_n_ubatch(lctx)") != std::string::npos,
        "mtmd CLI setup must pass the text decoder ubatch to mtmd");
    ok &= expect(mtmd_debug.find("mparams.decoder_n_ubatch = llama_n_ubatch(llama_init->context())") != std::string::npos,
        "mtmd debug setup must pass the text decoder ubatch to mtmd");

    ok &= expect(clip_cpp.find("clip_set_limit_image_tokens_for_non_causal_decode(hparams, 252, 280, decoder_n_ubatch)") != std::string::npos,
        "Gemma4 image token limits must be capped to decoder ubatch for non-causal image decode");
    ok &= expect(clip_cpp.find("clip_set_fixed_image_size_for_non_causal_decode") == std::string::npos,
        "Gemma3 fixed-size image decode must preserve full resolution instead of shrinking image_size");
    ok &= expect(clip_cpp.find("PROJECTOR_TYPE_GEMMA3") != std::string::npos &&
                 clip_cpp.find("requirements.min_decoder_batch_tokens = clip_default_fixed_image_tokens(model.hparams)") != std::string::npos,
        "Gemma3 fixed-size decode requirements must report the full image token count");
    ok &= expect(clip_cpp.find("loader.load_hparams(ctx_vision->model, CLIP_MODALITY_VISION, ctx_params.decoder_n_ubatch)") != std::string::npos,
        "clip initialization must pass decoder ubatch into vision hparam loading");
    ok &= expect(mtmd_cpp.find("/* decoder_n_ubatch */ ctx_params.decoder_n_ubatch") != std::string::npos,
        "mtmd must forward decoder ubatch to clip initialization");

    const size_t decode_req_pos = server_context.find("mtmd_get_decode_requirements_from_file(mmproj_path.c_str())");
    const size_t llama_init_pos = server_context.find("llama_init = common_init_from_params(params_base)");
    ok &= expect(decode_req_pos != std::string::npos && llama_init_pos != std::string::npos && decode_req_pos < llama_init_pos,
        "server must inspect mmproj decode requirements before creating the text context");
    ok &= expect(server_context.find("params_base.n_batch = std::max(params_base.n_batch, mmproj_decode_req.min_decoder_batch_tokens)") != std::string::npos &&
                 server_context.find("params_base.n_ubatch = std::max(params_base.n_ubatch, mmproj_decode_req.min_decoder_batch_tokens)") != std::string::npos,
        "server must raise batch and ubatch before context creation for full non-causal image chunks");

    ok &= expect(mtmd_helper.find("const int32_t n_ubatch = llama_n_ubatch(lctx)") != std::string::npos,
        "mtmd image decode must inspect the decoder physical ubatch");
    ok &= expect(mtmd_helper.find("non-causal attention requires the full") != std::string::npos,
        "mtmd image decode must fail gracefully when a non-causal chunk exceeds ubatch");
    ok &= expect(mtmd_helper.find("lower --image-max-tokens") != std::string::npos,
        "mtmd image decode error must tell users how to avoid the non-causal ubatch limit");
    ok &= expect(mtmd_image.find("Min-pixel upscaling can overshoot max_pixels after alignment.") != std::string::npos,
        "dynamic image resize must re-clamp to max_pixels after min-pixel alignment");

    return ok ? 0 : 1;
}
