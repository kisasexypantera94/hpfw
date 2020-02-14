#pragma once

#include <complex>
#include <vector>

#include <fftw3.h>

namespace fft {

    template<size_t CHUNK_SIZE>
    class FFTHandler {
    public:
        FFTHandler() : plan(fftw_plan_dft_1d(static_cast<int>(CHUNK_SIZE),
                                             nullptr,
                                             nullptr,
                                             FFTW_FORWARD,
                                             FFTW_ESTIMATE)) {}

        ~FFTHandler() {
            fftw_destroy_plan(plan);
        }

        auto fft(std::vector<std::complex<double>> &input, std::vector<std::complex<double>> &output) const {
            fftw_execute_dft(plan,
                             (fftw_complex *) &input[0],
                             (fftw_complex *) &output[0]);
        }

    private:
        const fftw_plan plan;
    };

}
