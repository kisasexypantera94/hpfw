#pragma once

#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <Eigen/Dense>

namespace hpfw::utils {

    template<typename Iterator, typename Func>
    auto chunks(Iterator begin, Iterator end, long k, long hop_length, Func f) {
        auto chunk_begin = begin;
        auto chunk_end = begin;
        std::advance(chunk_end, k);

        while (std::distance(chunk_begin, end) >= k) {
            f(chunk_begin, chunk_end);

            std::advance(chunk_begin, hop_length);
            std::advance(chunk_end, hop_length);
        }
    }

    template<typename T>
    struct extract_value_type {
        using value_type = T;
    };

    /// extract hashprint type from HashPrint
    template<
            template<typename, typename, size_t, size_t, typename> class X,
            typename N,
            typename SH,
            size_t FC,
            size_t T,
            typename R
    >
    struct extract_value_type<X<N, SH, FC, T, R>> {
        using value_type = N;
    };

} // utils

namespace cereal {

    template<class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    inline
    typename std::enable_if<traits::is_output_serializable < BinaryData < _Scalar>, Archive>::value, void>

    ::type
    save(Archive &ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> const &m) {
        int32_t rows = m.rows();
        int32_t cols = m.cols();
        ar(rows);
        ar(cols);
        ar(binary_data(m.data(), rows * cols * sizeof(_Scalar)));
    }

    template<class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    inline
    typename std::enable_if<traits::is_input_serializable < BinaryData < _Scalar>, Archive>::value, void>

    ::type
    load(Archive &ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> &m) {
        int32_t rows;
        int32_t cols;
        ar(rows);
        ar(cols);

        m.resize(rows, cols);

        ar(binary_data(m.data(), static_cast<std::size_t>(rows * cols * sizeof(_Scalar))));
    }

} // cereal

