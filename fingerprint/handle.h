#pragma once

#include "../helpers.h"
#include "algo/mpg123_wrapper.h"

namespace fp {
    template<typename T>
    using fingerprint = std::vector<T>;


    template<typename A, typename D>
    class FingerprintHandle {
    public:
        using T = extract_value_type<A>::value_type;

        FingerprintHandle() = default;

        FingerprintHandle(const FingerprintHandle &) = default;

        ~FingerprintHandle() = default;

        auto calc_fingerprint(const std::string &filename) const -> fingerprint<T> {
            try {
                D decoder;

                auto samples = decoder.decode(filename);
                return algo.calc(samples);
            } catch (const std::exception &e) {
                std::cerr << e.what() << std::endl;
            }

            return {};
        }

    private:
        A algo;
    };

}
