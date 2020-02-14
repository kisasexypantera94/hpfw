#pragma once

#include <complex>
#include <vector>

#include "../../helpers.h"
#include "fft.h"

namespace fp::algo {
    using fft::FFTHandler;

    template<typename T>
    class Shazam {
    public:
        Shazam() = default;

        ~Shazam() = default;

        auto calc(const std::vector<float> &samples) const -> fingerprint<T>;

    private:
        static constexpr size_t CHUNK_SIZE = 4096;
        const FFTHandler<CHUNK_SIZE> fft;
        const fingerprint <T> FREQ_RANGE = {40, 80, 120, 180, 300};
        static constexpr size_t FUZZ_FACTOR = 2;

        auto process_samples(const std::vector<float> &samples) const -> fingerprint<T>;

        auto get_max_freq(const std::vector<std::complex<double>> &complex_array, fingerprint <T> &hash_array) const;

        static auto hash(const fingerprint <T> &v) -> T;
    };

    template<typename T>
    auto Shazam<T>::hash(const fingerprint <T> &v) -> T {
        return (v[3] - (v[3] % FUZZ_FACTOR)) * 1e8 +
               (v[2] - (v[2] % FUZZ_FACTOR)) * 1e5 +
               (v[1] - (v[1] % FUZZ_FACTOR)) * 1e2 + (v[0] - (v[0] % FUZZ_FACTOR));
    }

    template<typename T>
    auto Shazam<T>::process_samples(const std::vector<float> &samples) const -> fingerprint <T> {
        fingerprint<T> hash_array;

        chunks(samples.cbegin(),
               samples.cend(),
               CHUNK_SIZE,
               [this, &hash_array](auto from, auto to) {
                   std::vector<std::complex<double>> complex_array(from, to);
                   fft.fft(complex_array, complex_array);
                   get_max_freq(complex_array, hash_array);
               });

        return hash_array;
    }

    template<typename T>
    auto Shazam<T>::get_max_freq(const std::vector<std::complex<double>> &complex_array,
                                 fingerprint <T> &hash_array) const {
        std::vector<double> high_scores(FREQ_RANGE.size(), 0);
        fingerprint <T> record_points(FREQ_RANGE.size(), 0);

        for (size_t freq = FREQ_RANGE[0]; freq < FREQ_RANGE.back(); ++freq) {
            double magnitude = abs(complex_array[freq]);

            size_t index =
                    std::upper_bound(FREQ_RANGE.begin(), FREQ_RANGE.end(), freq) -
                    FREQ_RANGE.begin();

            if (magnitude > high_scores[index]) {
                high_scores[index] = magnitude;
                record_points[index] = freq;
            }
        }

        hash_array.push_back(hash(record_points));
    }

    template<typename T>
    auto Shazam<T>::calc(const std::vector<float> &samples) const -> fingerprint <T> {
        return process_samples(samples);
    }

} // fp
