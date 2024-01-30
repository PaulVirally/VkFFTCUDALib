#include "vkFFT.h"

extern "C" VkFFTConfiguration* new_config(const pfUINT fftdim, const pfUINT buffer_dim, const pfUINT* size, const pfUINT* omit_dims, const pfUINT num_batches, const pfUINT coalesced_memory, const pfUINT aimThreads, const pfUINT numSharedBanks, const bool forward, const bool use_double_precision, const bool inplace, const bool normalize);
extern "C" void delete_config(VkFFTConfiguration* config);
extern "C" VkFFTApplication* new_app(const VkFFTConfiguration* const config, VkFFTResult* const res);
extern "C" void delete_app(VkFFTApplication* app);
extern "C" VkFFTResult fft(VkFFTApplication* app, void* input_buffer, void* output_buffer, int dir);
extern "C" unsigned long long max_fft_dimensions() { return VKFFT_MAX_FFT_DIMENSIONS; }