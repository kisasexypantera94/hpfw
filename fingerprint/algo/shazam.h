#pragma once

#include <array>
#include <complex>
#include <vector>

#include "../../helpers.h"

#include <Gist.h>

namespace fp::algo {
    template<typename N>
    class Shazam {
    public:
        Shazam() = default;

        ~Shazam() = default;

        auto calc(const std::vector<float> &samples) const -> fingerprint<N>;

    private:
        static constexpr size_t CHUNK_SIZE = 4096;
        static constexpr std::array<size_t, 5> FREQ_RANGE{40, 80, 120, 180, 300};
        static constexpr size_t FUZZ_FACTOR = 2;

        auto process_samples(const std::vector<float> &samples) const -> fingerprint<N>;

        auto get_max_freq(const std::vector<double> &magnitudes, fingerprint <N> &hash_array) const;

        static auto hash(const fingerprint <N> &v) -> N;
    };

    template<typename N>
    auto Shazam<N>::hash(const fingerprint <N> &v) -> N {
        return (v[3] - (v[3] % FUZZ_FACTOR)) * 1e8 +
               (v[2] - (v[2] % FUZZ_FACTOR)) * 1e5 +
               (v[1] - (v[1] % FUZZ_FACTOR)) * 1e2 + (v[0] - (v[0] % FUZZ_FACTOR));
    }

    template<typename N>
    auto Shazam<N>::process_samples(const std::vector<float> &samples) const -> fingerprint <N> {
        fingerprint<N> hash_array;
        Gist<double> gist{CHUNK_SIZE, 44100};

        chunks(samples.cbegin(),
               samples.cend(),
               CHUNK_SIZE,
               [this, &hash_array, &gist](auto from, auto to) {
                   gist.processAudioFrame({from, to});
                   get_max_freq(gist.getMagnitudeSpectrum(), hash_array);
               });

        return hash_array;
    }

    template<typename N>
    auto Shazam<N>::get_max_freq(const std::vector<double> &magnitudes, fingerprint <N> &hash_array) const {
        std::vector<double> high_scores(FREQ_RANGE.size(), 0);
        fingerprint <N> record_points(FREQ_RANGE.size(), 0);

        for (size_t freq = FREQ_RANGE[0]; freq < FREQ_RANGE.back(); ++freq) {
            size_t index =
                    std::upper_bound(FREQ_RANGE.begin(), FREQ_RANGE.end(), freq) -
                    FREQ_RANGE.begin();

            if (magnitudes[freq] > high_scores[index]) {
                high_scores[index] = magnitudes[freq];
                record_points[index] = freq;
            }
        }

        hash_array.emplace_back(hash(record_points));
    }

    template<typename N>
    auto Shazam<N>::calc(const std::vector<float> &samples) const -> fingerprint <N> {
        return process_samples(samples);
    }

} // fp
