#pragma once

#include <functional>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <Gist.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include "../handle.h"
#include "../../helpers.h"
#include "tsai.h"
#include "mpg123_wrapper.h"

using namespace dec;

using std::vector;
using std::string;

using Eigen::Dynamic;
using Eigen::Map;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::Transpose;

namespace fp::algo {

    template<typename N = int32_t,
            size_t FramesContext = 32,
            size_t MelBins = 33,
            size_t ChunkSize = 512,
            typename Real = float>
    class HashPrint {
    public:
        HashPrint() = default;

        ~HashPrint() = default;

        auto calc(const vector<string> &filenames) const {
            using range_type = tbb::blocked_range<vector<string>::const_iterator>;
            std::mutex mtx;
            auto cov = tbb::parallel_reduce(
                    range_type(filenames.cbegin(), filenames.cend()),
                    CovarianceMatrix(FrameSize, FrameSize),
                    [this, &mtx](const range_type &r, CovarianceMatrix init) {
                        for (const std::string &filename : r) {
                            {
                                std::scoped_lock lock(mtx);
                                std::cout << filename << std::endl;
                            }

                            try {
                                Mpg123Wrapper decoder;

                                auto frames = calc_frames(decoder.decode(filename));
                                init += calc_cov(frames.transpose());
                            } catch (const std::exception &e) {
                                std::cerr << e.what() << std::endl;
                            }
                        }

                        return init;
                    },
                    [](CovarianceMatrix x, CovarianceMatrix y) {
                        return x + y;
                    }
            );

            return cov;
        }

    private:
        static constexpr size_t FrameSize = MelBins * FramesContext;
        using CovarianceMatrix = Matrix<Real, Dynamic, Dynamic>; // to pass static_assert
        using Spectrogram = Matrix<Real, MelBins, Dynamic>;
        using Frames = Matrix<Real, FrameSize, Dynamic>;

        static constexpr size_t W = FramesContext / 2;

        auto calc_frames(const vector<Real> &samples) const -> Frames {
            MFCC<Real> mfcc(static_cast<int>(ChunkSize), 44100);
            Gist<Real> gist(static_cast<int>(ChunkSize), 44100);
            mfcc.setNumCoefficients(static_cast<int>(MelBins));

            Spectrogram spectrogram(MelBins, samples.size() / ChunkSize);

            chunks(samples.cbegin(),
                   samples.cend(),
                   ChunkSize,
                   [this, &gist, &mfcc, &spectrogram](auto from, auto to) {
                       gist.processAudioFrame({from, to});
                       mfcc.calculateMelFrequencySpectrum(gist.getMagnitudeSpectrum());

                       auto cnt = std::distance(from, to);
                       spectrogram.col(cnt) = Matrix<Real, MelBins, 1>::Map(mfcc.melSpectrum.data());
                   });

            Frames frames(FrameSize, spectrogram.cols() - 2 * W + 1);
            for (size_t i = W, cnt = 0; i < spectrogram.cols() - W + 1; ++i, ++cnt) {
                frames.col(cnt) = Matrix<Real, FrameSize, 1>::Map(
                        spectrogram.block(0, i - W, MelBins, FramesContext).data()
                );
            }

            return frames;
        }

        auto calc_cov(const Matrix<Real, Dynamic, Dynamic> &mat) const -> CovarianceMatrix {
            auto centered = mat.rowwise() - mat.colwise().mean();
            return (centered.adjoint() * centered) / double(mat.rows() - 1);
        }
    };

}