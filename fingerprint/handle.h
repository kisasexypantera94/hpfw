#pragma once

#include <omp.h>

#include "algo.h"
#include "mpg123_wrapper.h"

namespace fp {

    template<typename T, typename D, template <typename> typename A>
    class FingerprintHandle {
    public:
        FingerprintHandle() = default;
        ~FingerprintHandle() = default;

        auto calc_fingerprints(const std::vector<std::string>& filenames) -> std::vector<fingerprint<T>> {
            std::vector<fingerprint<T>> res(filenames.size());

            #pragma omp parallel for
            for (auto i = 0; i < filenames.size(); ++i) {
                std::cout << "Processing " + filenames[i] << std::endl;

                try {
                    D decoder;
                    auto samples = decoder.decode(filenames[i]);
                    res[i] = algo.calc(samples);
                } catch (const std::exception& e) {
                    std::cerr << e.what() << std::endl;
                }
            }

            return res;
        }

    private:
        A<T> algo;
    };

}
