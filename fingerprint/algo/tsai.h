#pragma once

#include <functional>
#include <iostream>
#include <vector>
#include <filesystem>
#include <map>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Gist.h>

#include "../handle.h"
#include "../../helpers.h"
#include "tsai.h"
#include "mpg123_wrapper.h"

#include <taskflow/taskflow.hpp>
#include <tbb/concurrent_vector.h>
#include <fstream>

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
            size_t ChunkSize = 44100 / 10,
            typename Real = float>
    class HashPrint {
    public:
        HashPrint() {
            Eigen::initParallel();
        }

        ~HashPrint() = default;

        int calc(const vector<string> &filenames) {
            Fingerprints fingerprints = prepare(filenames);

            return 0;
        }


    private:
        static constexpr size_t FrameSize = MelBins * FramesContext;
        static constexpr size_t W = FramesContext / 2;


        using CovarianceMatrix = Matrix<Real, Dynamic, Dynamic>; // to pass static_assert
        using Spectrogram = Matrix<Real, MelBins, Dynamic>;
        using Frames = Matrix<Real, FrameSize, Dynamic, Eigen::RowMajor>;

        using SongFramesPair = std::pair<string, Frames>;

        using Filters = Matrix<Real, Dynamic, Dynamic>;

        using Features = Matrix<bool, Dynamic, 16>;
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

                std::pair<long long, string> mx;
                map<string, map<long long, long long>> cnt;
                for (size_t i = 0; i < fingerprints.size(); ++i) {
                    auto n = fingerprints[i].second.rows();
                    auto k = fp.rows();

                    for (long long kek = 0; kek < k; ++kek) {
                        for (long long l = 0; l < n; ++l) {
                            if ((fingerprints[i].second.row(l).array() != fp.row(kek).array()).count() == 0) {
                                auto offset = kek - l;
                                ++cnt[fingerprints[i].first][offset];
                                if (cnt[fingerprints[i].first][offset] > mx.first) {
                                    mx = {cnt[fingerprints[i].first][offset], fingerprints[i].first};
                                    bestIndex = offset;
                                }
                            }
                        }
                    }
                }

                cout << mx.second << " " << mx.first << " " << bestIndex << endl;
            }
        }

        Fingerprints

        prepare(const vector<string> &filenames) {
            concurrent_vector<SongFramesPair> frames = preprocess(filenames);
            Fingerprints fingerprints = calc_features(frames);
            find(fingerprints);

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
                            std::ofstream file("test.txt");
                            if (file.is_open()) {
                                file << s << '\n';
                            }

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

            accum_cov /= frames.size();


            filters = calc_filters(accum_cov);

            return frames;
        }

        Filters calc_filters(const CovarianceMatrix &cov) {
            cout << "|----------------------------------------------|" << endl;
            cout << "|             Calculating filters              |" << endl;
            cout << "|----------------------------------------------|" << endl;
            SelfAdjointEigenSolver<CovarianceMatrix> solver(cov);
            return solver.eigenvectors().rowwise().reverse().transpose().block(0, 0, 16, FrameSize);
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

        static Matrix<bool, Dynamic, 16> calc_delta(const Matrix<Real, 16, Dynamic> &f, size_t t = 50) {
            Matrix<bool, Dynamic, 16> delta(f.cols() - t, 16);
            for (size_t i = 0; i < f.cols() - t; ++i) {
                Matrix<Real, 16, 1> diff = f.col(i) - f.col(i + t);
                delta.row(i) = (diff.array() >= 0);
            }

            return delta;
        }

        static Frames calc_frames(const Spectrogram &spectrogram) {
            Frames frames(FrameSize, spectrogram.cols() - 2 * W + 1);
            for (size_t i = W, cnt = 0; i < spectrogram.cols() - W + 1; ++i, ++cnt) {
                Matrix<Real, Dynamic, Dynamic, Eigen::RowMajor> x = spectrogram.block(0, i - W, MelBins, FramesContext);
                x.resize(FrameSize, 1);

                frames.col(cnt) = x;
            }

            return frames;
        }

        static Spectrogram calc_spectro(const vector<Real> &samples) {
            MFCC<Real> mfcc(static_cast<int>(ChunkSize), 44100);
            Gist<Real> gist(static_cast<int>(ChunkSize), 44100);
            mfcc.setNumCoefficients(static_cast<int>(MelBins));

            std::cout << samples.size() << std::endl;
            Spectrogram spectrogram(MelBins, samples.size() / (44100 / 100));

            size_t cnt = 0;
            chunks(samples.cbegin(),
                   samples.cend(),
                   ChunkSize,
                   [&gist, &mfcc, &spectrogram, &cnt](auto from, auto to) {
                       gist.processAudioFrame({from, to});
                       mfcc.calculateMelFrequencySpectrum(gist.getMagnitudeSpectrum());

                       spectrogram.col(cnt) = Matrix<Real, MelBins, 1>(mfcc.melSpectrum.data());
                       ++cnt;
                   },
                   44100 / 100);

            double maxVal = std::max(1e-10, double(spectrogram.maxCoeff()));
            spectrogram = 10.0 * ((spectrogram).array() < 1e-10).select(1e-10, spectrogram).array().log10() -
                          10.0 * log10(maxVal);
            maxVal = std::max(1e-10, double(spectrogram.maxCoeff()));
            spectrogram = (spectrogram.array() < double(maxVal - 80.0)).select(maxVal - 80.0, spectrogram);

            return spectrogram;
        }

        static CovarianceMatrix calc_cov(const Matrix<Real, Dynamic, Dynamic> &mat) {
            const Matrix<Real, Dynamic, Dynamic> centered = mat.rowwise() - mat.colwise().mean();
            return (centered.adjoint() * centered) / double(mat.rows() - 1);
        }
    };

}