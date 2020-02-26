#pragma once

#include <functional>
#include <iostream>
#include <vector>
#include <filesystem>
#include <map>

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
using std::map;
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
            Fingerprints fingerprints = prepare(filenames);
            find(fingerprints);

            return 0;
        }


    private:
        static constexpr size_t FrameSize = MelBins * FramesContext;
        static constexpr size_t W = FramesContext / 2;


        using CovarianceMatrix = Matrix<Real, Dynamic, Dynamic>; // to pass static_assert
        using Spectrogram = Matrix<Real, MelBins, Dynamic>;
        using Frames = Matrix<Real, FrameSize, Dynamic>;

        using SongFramesPair = std::pair<string, Frames>;

        using Filters = Matrix<Real, 16, Dynamic>;

        using Features = Matrix<bool, 16, Dynamic>;
        using SongFeaturesPair = std::pair<string, Features>;
        using Fingerprints = concurrent_vector<SongFeaturesPair>;

        tf::Executor executor;
        Filters filters;


        void find(const Fingerprints &fingerprints) {
            auto it = std::filesystem::directory_iterator("/Users/chingachgook/dev/rust/khalzam/samples");
            for (const auto &f : it) {
                const auto &filename = f.path();
                if (filename.extension() != ".mp3") {
                    continue;
                }

                std::cout << "FINDING " << filename << std::endl;

                Mpg123Wrapper decoder;

                auto samples = decoder.decode(filename);
                Spectrogram s = calc_spectro(samples);
                Frames frames = calc_frames(s);
                Features fp = calc_delta(filters * frames);

                long long bestDistance = 10000000000, bestIndex = -1;
                string bestSong = "";
                for (size_t i = 0; i < fingerprints.size(); ++i) {
                    auto n = fingerprints[i].second.cols();
                    auto k = fp.cols();

                    for (long long l = 0; l < n - k; ++l) {
                        auto distance = (fp.array() != fingerprints[i]
                                .second.block(0, l, 16, k)
                                .array()).count();
                        if (distance < bestDistance) {
                            bestDistance = distance;
                            bestIndex = l;
                            bestSong = fingerprints[i].first;
                        }
                    }
                }

                cout << bestSong << " " << bestDistance << " " << bestIndex <<
                     endl;
            }
        }

        Fingerprints

        prepare(const vector<string> &filenames) {
            concurrent_vector<SongFramesPair> frames = preprocess(filenames);
            Fingerprints fingerprints = calc_features(frames);
            cout << frames[0].second.rows() << " " << frames[0].second.cols() << endl;
            cout << fingerprints[0].second.rows() << " " << fingerprints[0].second.cols() << endl;

            return fingerprints;
        }

        concurrent_vector<SongFramesPair> preprocess(const vector<string> &filenames) {
            cout << "|----------------------------------------------|" << endl;
            cout << "|Calculating spectrograms and covariance matrix|" << endl;
            cout << "|----------------------------------------------|" << endl;
            CovarianceMatrix accum_cov(FrameSize, FrameSize);
            concurrent_vector<SongFramesPair> frames;
            std::mutex mtx;

            tf::Taskflow taskflow;
            taskflow.parallel_for(
                    filenames.cbegin(), filenames.cend(),
                    [&frames, &accum_cov, &mtx](const string &filename) {
                        try {
                            Mpg123Wrapper decoder;

                            auto samples = decoder.decode(filename);
                            Spectrogram s = calc_spectro(samples);
                            auto f = frames.push_back({filename, calc_frames(s)});

                            CovarianceMatrix cov = calc_cov(f->second.transpose());
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
        concurrent_vector<SongFeaturesPair> calc_features(const Iterable &frames) {
            cout << "|----------------------------------------------|" << endl;
            cout << "|            Calculating features              |" << endl;
            cout << "|----------------------------------------------|" << endl;
            tf::Taskflow taskflow;
            concurrent_vector<SongFeaturesPair> features;
            taskflow.parallel_for(
                    frames.cbegin(), frames.cend(),
                    [this, &features](const SongFramesPair &f) {
                        features.push_back({f.first, calc_delta(filters * f.second)});
                    }
            );

            executor.run(taskflow).wait();

            return features;
        }

        static Matrix<bool, 16, Dynamic> calc_delta(const Matrix<Real, 16, Dynamic> &f, size_t t = 50) {
            Matrix<bool, 16, Dynamic> delta(16, f.cols() - t);
            for (size_t i = t, cnt = 0; i < f.cols() - t; ++i, ++cnt) {
                Matrix<Real, 16, 1> diff = f.col(i) - f.col(i + t);
                delta.col(cnt) = (diff.array() > 0.0);
            }

            return delta;
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