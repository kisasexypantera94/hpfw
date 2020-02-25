//#pragma once
//
//#include "../helpers.h"
//#include "algo/mpg123_wrapper.h"
//
//namespace fp {
//    template<typename N>
//    using fingerprint = std::vector<N>;
//
//
//    template<typename A, typename D>
//    class FingerprintHandle {
//    public:
//        using N = typename extract_value_type<A>::value_type;
//
//        FingerprintHandle() = default;
//
//        FingerprintHandle(const FingerprintHandle &) = default;
//
//        ~FingerprintHandle() = default;
//
//        auto calc_fingerprint(const std::string &filename) const -> fingerprint<N> {
//            try {
//                D decoder;
//
//                auto samples = decoder.decode(filename);
//                return algo.calc(samples);
//            } catch (const std::exception &e) {
//                std::cerr << e.what() << std::endl;
//            }
//
//            return {};
//        }
//
//    private:
////        static constexpr A algo;
//    };
//
//}
