#pragma once

#include <complex>
#include <vector>

#include <eigen3/unsupported/Eigen/FFT>

namespace fp {

    template<typename T>
    using fingerprint = std::vector<T>;

    namespace algo {

        template<typename T>
        struct Algo {
            virtual auto calc(const std::vector<float> &samples) -> fingerprint<T> = 0;
        };

        template<typename T>
        class Shazam : public Algo<T> {
        public:
            auto calc(const std::vector<float> &samples) -> fingerprint<T> override {
                return process_samples(samples);
            }

        private:
            static const size_t CHUNK_SIZE = 2048;
            const fingerprint<T> FREQ_RANGE = {40, 80, 120, 180, 300};
            static const size_t FUZZ_FACTOR = 2;


            auto process_samples(const std::vector<float> &samples) const -> fingerprint<T>;
            auto get_max_freq(const std::vector<std::complex<double>> &complex_array, fingerprint<T> &hash_array) const;
            auto hash(const fingerprint<T> &v) const -> T;
            auto fft(std::vector<std::complex<double>> &complex_array) const;
        };

        template<typename T>
        auto Shazam<T>::process_samples(const std::vector<float> &samples) const -> fingerprint<T> {
            size_t block_number = samples.size() / CHUNK_SIZE;
            fingerprint<T> hash_array;

            for (size_t i = 0; i < block_number; ++i) {
                std::vector<std::complex<double>> complex_array(CHUNK_SIZE);

                for (size_t j = 0; j < CHUNK_SIZE; ++j) {
                    complex_array[j] = std::complex<double>(samples[i * CHUNK_SIZE + j], 0);
                }

                fft(complex_array);
                get_max_freq(complex_array, hash_array);
            }

            return hash_array;
        }

        template<typename T>
        auto
        Shazam<T>::get_max_freq(const std::vector<std::complex<double>> &complex_array, fingerprint<T> &hash_array) const {
            std::vector<double> high_scores(FREQ_RANGE.size(), 0);
            fingerprint<T> record_points(FREQ_RANGE.size(), 0);

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
        auto Shazam<T>::hash(const fingerprint<T> &v) const -> T {
            return (v[3] - (v[3] % FUZZ_FACTOR)) * 1e8 +
                   (v[2] - (v[2] % FUZZ_FACTOR)) * 1e5 +
                   (v[1] - (v[1] % FUZZ_FACTOR)) * 1e2 + (v[0] - (v[0] % FUZZ_FACTOR));
        }

        template<typename T>
        auto Shazam<T>::fft(std::vector<std::complex<double>> &complex_array) const {
            size_t arr_size = complex_array.size();
            size_t k = arr_size;
            size_t n;
            double thetaT = 3.14159265358979323846264338328L / arr_size;
            std::complex<double> phiT = std::complex<double>(cos(thetaT), -sin(thetaT));
            std::complex<double> TT;

            while (k > 1) {
                n = k;
                k >>= 1;
                phiT = phiT * phiT;
                TT = 1.0L;

                for (size_t l = 0; l < k; l++) {
                    for (size_t a = l; a < arr_size; a += n) {
                        size_t b = a + k;
                        std::complex<double> t = complex_array[a] - complex_array[b];
                        complex_array[a] += complex_array[b];
                        complex_array[b] = t * TT;
                    }

                    TT *= phiT;
                }
            }
            // Decimate
            size_t m = static_cast<size_t>(log2(arr_size));

            for (size_t a = 0; a < arr_size; a++) {
                size_t b = a;
                // Reverse bits
                b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
                b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
                b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
                b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
                b = ((b >> 16) | (b << 16)) >> (32 - m);

                if (b > a) {
                    std::complex<double> t = complex_array[a];
                    complex_array[a] = complex_array[b];
                    complex_array[b] = t;
                }
            }
        }

    }

}
