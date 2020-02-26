#pragma once

#include <functional>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Gist.h>

#include "../handle.h"
#include "../../helpers.h"
#include "tsai.h"
#include "mpg123_wrapper.h"

#include <taskflow/taskflow.hpp>
#include <tbb/concurrent_vector.h>

using namespace dec;

using std::cout;
using std::endl;
using std::vector;
using std::string;

using Eigen::Dynamic;
using Eigen::Map;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::Transpose;
using Eigen::SelfAdjointEigenSolver;

using tbb::concurrent_vector;

namespace fp::algo {

    template<typename N = int32_t,
            size_t FramesContext = 32,
            size_t MelBins = 33,
            size_t ChunkSize = 512,
            typename Real = float>
    class HashPrint {
    public:
        HashPrint() {
            Eigen::initParallel();
        }

        ~HashPrint() = default;

        int calc(const vector<string> &filenames) {
            prepare(filenames);

            return 0;
        }

    private:
        static constexpr size_t FrameSize = MelBins * FramesContext;
        static constexpr size_t W = FramesContext / 2;

        using CovarianceMatrix = Matrix<Real, Dynamic, Dynamic>; // to pass static_assert
        using Spectrogram = Matrix<Real, MelBins, Dynamic>;
        using Frames = Matrix<Real, FrameSize, Dynamic>;
        using Filters = Matrix<Real, 16, Dynamic>;
        using Features = Matrix<Real, 16, Dynamic>;

        tf::Executor executor;
        Filters filters;

        concurrent_vector<Frames> prepare(const vector<string> &filenames) {
            concurrent_vector<Frames> frames = preprocess(filenames);
            concurrent_vector<Features> fingerprints = calc_features(frames);

            return frames;
        }


        concurrent_vector<Frames> preprocess(const vector<string> &filenames) {
            cout << "|----------------------------------------------|" << endl;
            cout << "|Calculating spectrograms and covariance matrix|" << endl;
            cout << "|----------------------------------------------|" << endl;
            CovarianceMatrix accum_cov(FrameSize, FrameSize);
            concurrent_vector<Frames> frames;
            std::mutex mtx;

            tf::Taskflow taskflow;
            taskflow.parallel_for(
                    filenames.cbegin(), filenames.cend(),
                    [&frames, &accum_cov, &mtx](const string &filename) {
                        try {
                            Mpg123Wrapper decoder;

                            auto samples = decoder.decode(filename);
                            Spectrogram s = calc_spectro(samples);
                            auto f = frames.push_back(calc_frames(s));

                            CovarianceMatrix cov = calc_cov(f->transpose());
                            {
                                std::scoped_lock lock(mtx);
                                std::cout << filename << std::endl;
                                accum_cov += cov;
                            }
                        } catch (const std::exception &e) {
                            std::cerr << e.what() << std::endl;
                        }
                    }
            );

            executor.run(taskflow).wait();

            filters = calc_filters(accum_cov);

            return frames;
        }

        Filters calc_filters(const CovarianceMatrix &cov) {
            cout << "|----------------------------------------------|" << endl;
            cout << "|             Calculating filters              |" << endl;
            cout << "|----------------------------------------------|" << endl;
            SelfAdjointEigenSolver<CovarianceMatrix> solver(cov);
            return solver.eigenvectors().block(0, 0, FrameSize, 16).transpose();
        }

        template<typename Iterable>
        concurrent_vector<Features> calc_features(const Iterable &frames) {
            cout << "|----------------------------------------------|" << endl;
            cout << "|            Calculating features              |" << endl;
            cout << "|----------------------------------------------|" << endl;
            tf::Taskflow taskflow;
            concurrent_vector<Features> features(frames.size());
            taskflow.parallel_for(
                    frames.cbegin(), frames.cend(),
                    [this, &features](const Frames &f) {
                        features.push_back(filters * f);
                    }
            );

            executor.run(taskflow).wait();

            return features;
        }

        static Frames calc_frames(const Spectrogram &spectrogram) {
            Frames frames(FrameSize, spectrogram.cols() - 2 * W + 1);
            for (size_t i = W, cnt = 0; i < spectrogram.cols() - W + 1; ++i, ++cnt) {
                frames.col(cnt) = Matrix<Real, FrameSize, 1>::Map(
                        spectrogram.block(0, i - W, MelBins, FramesContext).data()
                );
            }

            return frames;
        }

        static Spectrogram calc_spectro(const vector<Real> &samples) {
            MFCC<Real> mfcc(static_cast<int>(ChunkSize), 44100);
            Gist<Real> gist(static_cast<int>(ChunkSize), 44100);
            mfcc.setNumCoefficients(static_cast<int>(MelBins));

            Spectrogram spectrogram(MelBins, samples.size() / ChunkSize);

            size_t cnt = 0;
            chunks(samples.cbegin(),
                   samples.cend(),
                   ChunkSize,
                   [&gist, &mfcc, &spectrogram, &cnt](auto from, auto to) {
                       gist.processAudioFrame({from, to});
                       mfcc.calculateMelFrequencySpectrum(gist.getMagnitudeSpectrum());

                       spectrogram.col(cnt) = Matrix<Real, MelBins, 1>::Map(mfcc.melSpectrum.data());
                       ++cnt;
                   });

            return spectrogram;
        }

        static CovarianceMatrix calc_cov(const Matrix<Real, Dynamic, Dynamic> &mat) {
            const Matrix<Real, Dynamic, Dynamic> centered = mat.rowwise() - mat.colwise().mean();
            return (centered.adjoint() * centered) / double(mat.rows() - 1);
        }
    };

}