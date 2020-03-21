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

    auto get_dir_files(const std::string &dir) {
        auto it = std::filesystem::directory_iterator(dir);
        std::vector<std::string> files;
        for (const auto &f : it) {
            const auto &filename = f.path();
            files.emplace_back(filename);
        }
        return files;
    }

    auto count_dir_files(const std::string &dir) -> uint64_t {
        namespace fs = std::filesystem;

        auto it = std::filesystem::directory_iterator(dir);
        return static_cast<uint64_t>(std::count_if(
                fs::begin(it),
                fs::end(it),
                [](const auto &entry) { return entry.is_regular_file(); }
        ));
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
            typename C
    >
    struct extract_value_type<X<N, SH, FC, T, C>> {
        using value_type = N;
    };

} // utils

namespace cereal {

    template<typename Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    inline
    typename std::enable_if<traits::is_output_serializable < BinaryData < _Scalar>, Archive>::value, void>::type
    save(Archive &ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> const &m) {
        int32_t rows = m.rows();
        int32_t cols = m.cols();
        ar(rows);
        ar(cols);
        ar(binary_data(m.data(), rows * cols * sizeof(_Scalar)));
    }

    template<typename Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    inline
    typename std::enable_if<traits::is_input_serializable < BinaryData < _Scalar>, Archive>::value, void>::type
    load(Archive &ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> &m) {
        int32_t rows;
        int32_t cols;
        ar(rows);
        ar(cols);

        m.resize(rows, cols);

        ar(binary_data(m.data(), static_cast<std::size_t>(rows * cols * sizeof(_Scalar))));
    }

} // cereal

