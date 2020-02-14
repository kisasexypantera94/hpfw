#pragma once

#include <vector>

#include "../../helpers.h"
#include "fft.h"

using fft::FFTHandler;
using spectrogram = std::vector<std::vector<size_t>>;

template<size_t CHUNK_SIZE>
class Spectrogram {
public:
    auto spectrogram(std::vector<float> &samples) -> spectrogram {
        spectrogram s;

        chunks(samples.cbegin(),
               samples.cend(),
               CHUNK_SIZE,
               [this, &s](auto from, auto to) {
                   std::vector<std::complex<double>> complex_array(from, to);
                   fft_handler.fft(complex_array, complex_array);

                   for (const auto &c : complex_array) {

                   }
               });
    }

private:
    FFTHandler fft_handler;
};
