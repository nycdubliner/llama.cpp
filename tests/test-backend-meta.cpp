#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>

#include <cstdio>
#include <cstring>

static ggml_backend_dev_t get_cpu_device() {
    ggml_backend_load_all();

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            return dev;
        }
    }

    return nullptr;
}

static ggml_backend_meta_split_state split_state_for_test(const ggml_tensor * tensor, void *) {
    ggml_backend_meta_split_state state = {};

    if (std::strcmp(tensor->name, "x") == 0) {
        state.axis       = GGML_BACKEND_SPLIT_AXIS_1;
        state.n_segments = 1;
        state.ne[0]      = tensor->ne[1] / 2;
        state.ne[1]      = tensor->ne[1] - state.ne[0];
        return state;
    }

    state.axis       = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    state.n_segments = 1;
    return state;
}

static void test_turbo_wht_split_state() {
    ggml_backend_dev_t cpu_dev = get_cpu_device();
    GGML_ASSERT(cpu_dev != nullptr);

    ggml_backend_dev_t devs[2] = { cpu_dev, cpu_dev };
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(devs, 2, split_state_for_test, nullptr);
    GGML_ASSERT(meta_dev != nullptr);

    ggml_backend_buffer_type_t meta_buft = ggml_backend_dev_buffer_type(meta_dev);
    GGML_ASSERT(meta_buft != nullptr);

    ggml_init_params params = {
        /*.mem_size   =*/ 64 * ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx(ggml_init(params));
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 128, 4, 2);
    ggml_set_name(x, "x");

    ggml_tensor * scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 128);
    ggml_set_name(scale, "scale");

    ggml_backend_buffer_ptr static_buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), meta_buft));
    GGML_ASSERT(static_buf != nullptr);
    ggml_backend_buffer_set_usage(static_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    GGML_ASSERT(x->buffer != nullptr);
    GGML_ASSERT(scale->buffer != nullptr);

    ggml_tensor * y = ggml_turbo_wht(ctx.get(), x, /*direction=*/ 1, /*group_size=*/ 128, scale);
    ggml_set_name(y, "y");
    ggml_set_output(y);

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, y);

    ggml_gallocr_ptr galloc(ggml_gallocr_new(meta_buft));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc.get(), graph));
    GGML_ASSERT(y->buffer != nullptr);
}

int main() {
    ggml_log_set(nullptr, nullptr);

    test_turbo_wht_split_state();

    std::printf("OK\n");
    return 0;
}

