#include "include/vkfft_cuda.cuh"

VkFFTConfiguration* new_config(const pfUINT fftdim, const pfUINT buffer_dim, const pfUINT* size, const pfUINT* omit_dims, const pfUINT num_batches,
                               const pfUINT coalesced_memory, const pfUINT aimThreads, const pfUINT numSharedBanks,
                               const bool forward, const bool use_double_precision, const bool inplace) {
    VkFFTConfiguration* const config = new VkFFTConfiguration({});

    // FFT dimension config
    config->FFTdim = fftdim;
    for (size_t i = 0; i < VKFFT_MAX_FFT_DIMENSIONS; ++i) {
        config->size[i] = size[i]; // We guarantee in Julia that size is of size VKFFT_MAX_FFT_DIMENSIONS and is not a nullptr
        config->omitDimension[i] = omit_dims[i]; // We guarantee in Julia that omit_dims is of size VKFFT_MAX_FFT_DIMENSIONS and is not a nullptr
    }

    // Batching config
    config->numberBatches = num_batches;
    if (num_batches < 1) {
        config->numberBatches = 1;
    }

    // Device config
    CUdevice* const device = new CUdevice; // FIXME: Use a smart pointer?
    if (cuCtxGetDevice(device) != CUDA_SUCCESS) {
        // TODO: Let the user know about the error
        delete device;
        return nullptr;
    }
    config->device = device;

    // Buffer allocation (the buffer is the work area for the FFT)
    config->doublePrecision = use_double_precision;
    pfUINT* const buffer_size = new pfUINT(1); // FIXME: Use a smart pointer?
    for (size_t i = 0; i < buffer_dim; ++i) { // Product of all the non-omitted dimensions
        if (omit_dims[i]) continue;
        *buffer_size *= size[i];
    }
    *buffer_size *= config->numberBatches;
    *buffer_size *= 2 * (use_double_precision ? sizeof(double) : sizeof(float)); // *2 because we need to store both real and imaginary parts.
    config->bufferSize = buffer_size;
    void** const buffer_ptr = new void*; // FIXME: Use a smart pointer?
    *buffer_ptr = reinterpret_cast<void*>(1); // A dummy value, it just can't be nullptr
    config->buffer = buffer_ptr;

    if (!inplace) {
        config->isInputFormatted = true;

        void** const output_buffer_ptr = new void*; // FIXME: Use a smart pointer?
        *output_buffer_ptr = reinterpret_cast<void*>(2); // Another dummy value, must be different from buffer_ptr
        config->buffer = output_buffer_ptr;
        config->inputBuffer = buffer_ptr;

        config->inputBufferSize = buffer_size;
    }

    // Optional optimization parameters
    if (coalesced_memory) config->coalescedMemory = coalesced_memory;
    if (aimThreads) config->aimThreads = aimThreads;
    if (numSharedBanks) config->numSharedBanks = numSharedBanks;

    // The julia bindings create separate plans for forward and backward transforms, so we don't need to generate both kernels here
    config->makeForwardPlanOnly = forward;
    config->makeInversePlanOnly = !forward;

    return config;
}

void delete_config(VkFFTConfiguration* config) {
    if (config == nullptr) {
        return;
    }
    delete config->device;
    delete config->bufferSize;
    delete config->buffer;

    if ((config->outputBuffer != nullptr) && (config->buffer != config->outputBuffer)) {
        delete config->outputBuffer;
    }
    if ((config->inputBuffer != nullptr) && (config->buffer != config->inputBuffer) && (config->outputBuffer != config->inputBuffer)) {
        delete config->inputBuffer;
    }
    if ((config->inputBufferSize != nullptr) && (config->inputBufferSize != config->bufferSize)) {
        delete config->inputBufferSize;
    }
    if ((config->outputBufferSize != nullptr) && (config->outputBufferSize != config->bufferSize) && (config->outputBufferSize != config->inputBufferSize)) {
        delete config->outputBufferSize;
    }

    delete config;
}

VkFFTApplication* new_app(const VkFFTConfiguration* const config, VkFFTResult* const res) {
    if (config == nullptr) {
        return nullptr;
    }
    VkFFTApplication* const app = new VkFFTApplication({});
    *res = initializeVkFFT(app, *config);

    if (res == nullptr) {
        // TODO: This really should be unreachable
        // TODO: Let the user know about the error
        delete app;
        return nullptr;
    }

    if (*res != VKFFT_SUCCESS) {
        // TODO: Let the user know about the error
        delete app;
        return nullptr;
    }

    return app;
}

void delete_app(VkFFTApplication* app) {
    if (app == nullptr) {
        return;
    }
    deleteVkFFT(app);
    delete app;
}

VkFFTResult fft(VkFFTApplication* app, void* input_buffer, void* output_buffer, int direction) {
    *(app->configuration.buffer) = output_buffer;
    *(app->configuration.inputBuffer) = input_buffer;
    *(app->configuration.outputBuffer) = output_buffer;

    VkFFTLaunchParams params = {};
    params.buffer = app->configuration.buffer;
    params.inputBuffer = app->configuration.inputBuffer;
    params.outputBuffer = app->configuration.outputBuffer;

    return VkFFTAppend(app, direction, &params);
}