#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Gist.h>
#include <taskflow/taskflow.hpp>
#include <tbb/concurrent_vector.h>

#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>

#include "../../helpers.h"
#include "mpg123_wrapper.h"
#include "tsai.h"

namespace fp::algo {

    using std::cout;
    using std::endl;
    using std::vector;
    using std::map;
    using std::unordered_map;
    using std::string;

    using Eigen::Dynamic;
    using Eigen::Map;
    using Eigen::Matrix;
    using Eigen::SelfAdjointEigenSolver;
    using tbb::concurrent_vector;

    using namespace dec;

    template<typename N = uint16_t,
            size_t FramesContext = 32,
            size_t MelBins = 33,
            size_t NFFT = 44100 / 10, // TODO: 44100 -> file sample rate
            size_t HopLength = 44100 / 100,
            size_t T = 1,
            typename Real = double>
    class HashPrint {
    public:
        HashPrint() {
            Eigen::initParallel();
        }

        HashPrint(HashPrint &&hp) : filters(std::move(hp.filters)),
                                    db(std::move(hp.db)) {}

        ~HashPrint() = default;

        static auto from_dump(const std::string &dump_name) -> HashPrint {
            HashPrint hp;

            {
                std::ifstream is(dump_name, std::ios::binary);
                cereal::BinaryInputArchive archive(is);
                archive(hp);
            }

            return hp;
        }

        void prepare(const vector<string> &filenames) {
            if (db.size() > 0) {
                return;
            }

            build_db(calc_features(preprocess(filenames)));
        }

        auto dump(const std::optional<string> &filename) -> string {
            const auto dump_name = filename.value_or("db/dump.cereal");
            {
                std::ofstream os(dump_name, std::ios::binary);
                cereal::BinaryOutputArchive archive(os);
                archive(this);
            }

            return dump_name;
        }

        struct SearchResult {
            string filename;
            size_t cnt;
            long long offset;
        };

        auto find(const std::string &filename) -> SearchResult {
            std::cout << "FINDING " << filename << std::endl;

            Mpg123Wrapper decoder;

            auto samples = decoder.decode(filename);
            auto s = calc_spectro(samples);
            auto frames = calc_frames(s);
            auto fp = calc_delta(filters * frames);

            SearchResult res = {"", 0, 0};
            map<string, map<long long, size_t>> cnt;

            for (long long r = 0, num_rows = fp.rows(); r < num_rows; ++r) {
                const N n = bool_row_to_num(fp, r);

                for (const auto &[filename, offset] : db[n]) {
                    long long diff = r - offset;
                    auto c = ++cnt[filename][diff];
                    if (c > res.cnt) {
                        res = {filename, c, diff};
                    }
                }
            }

            return res;
        }

        template<class Archive>
        void save(Archive &ar) const {
            ar(db);
            ar(filters);
        }

        template<class Archive>
        void load(Archive &ar) {
            ar(db);
            ar(filters);
        }

    private:
        static constexpr size_t FrameSize = MelBins * FramesContext;
        static constexpr size_t W = FramesContext / 2;
        static constexpr size_t NumOfFilters = sizeof(N) * 8;

        using Spectrogram = Matrix<Real, MelBins, Dynamic>;
        using Frames = Matrix<Real, FrameSize, Dynamic, Eigen::RowMajor>;
        using CovarianceMatrix = Matrix<Real, Dynamic, Dynamic>; // to pass static_assert
        using Filters = Matrix<Real, NumOfFilters, Dynamic>;
        using Features = Matrix<bool, Dynamic, NumOfFilters>;

        struct FilenameFramesPair {
            string filename;
            Frames frames;
        };

        struct FilenameFeaturesPair {
            string filename;
            Features features;
        };

        struct FilenameOffsetPair {
            string filename;
            size_t offset;

            template<class Archive>
            void save(Archive &ar) const {
                ar(filename);
                ar(offset);
            }

            template<class Archive>
            void load(Archive &ar) {
                ar(filename);
                ar(offset);
            }
        };

        using DB = unordered_map<N, vector<FilenameOffsetPair>>;

        tf::Executor executor;
        Filters filters;
        DB db;

        auto preprocess(const vector<string> &filenames) -> concurrent_vector<FilenameFramesPair> {
            cout << "|----------------------------------------------|" << endl;
            cout << "|Calculating spectrograms and covariance matrix|" << endl;
            cout << "|----------------------------------------------|" << endl;
            CovarianceMatrix accum_cov(FrameSize, FrameSize);
            concurrent_vector<FilenameFramesPair> frames;
            std::mutex mtx;

            tf::Taskflow taskflow;
            taskflow.parallel_for(filenames.cbegin(), filenames.cend(),
                                  [&frames, &accum_cov, &mtx](const string &filename) {
                                      try {
                                          const auto samples = Mpg123Wrapper().decode(filename);
                                          const auto spectrogram = calc_spectro(samples);
                                          const auto f = frames.push_back({filename, calc_frames(spectrogram)});
                                          const auto cov = calc_cov(f->frames.transpose());
                                          {
                                              std::scoped_lock lock(mtx);
                                              std::cout << filename << std::endl;
                                              accum_cov += cov;
                                          }
                                      } catch (const std::exception &e) {
                                          std::cerr << e.what() << std::endl;
                                      }
                                  });

            executor.run(taskflow).wait();

            filters = calc_filters(accum_cov / frames.size());

            return frames;
        }

        template<typename Iterable>
        auto calc_features(const Iterable &frames) -> concurrent_vector<FilenameFeaturesPair> {
            cout << "|----------------------------------------------|" << endl;
            cout << "|            Calculating features              |" << endl;
            cout << "|----------------------------------------------|" << endl;
            concurrent_vector<FilenameFeaturesPair> features;
            tf::Taskflow taskflow;
            taskflow.parallel_for(
                    frames.cbegin(), frames.cend(),
                    [this, &features](const FilenameFramesPair &f) {
                        features.push_back({f.filename, calc_delta(filters * f.frames)});
                    }
            );

            executor.run(taskflow).wait();

            return features;
        }

        void build_db(const concurrent_vector<FilenameFeaturesPair> &features_vec) {
            for (const auto &[filename, features]: features_vec) {
                for (size_t i = 0, num_rows = features.rows(); i < num_rows; ++i) {
                    const N n = bool_row_to_num(features, i);
                    db[n].push_back({filename, i});
                }
            }
        }

        static auto calc_spectro(const vector<float> &samples) -> Spectrogram {
            MFCC<Real> mfcc(static_cast<int>(NFFT), 44100);
            Gist<Real> gist(static_cast<int>(NFFT), 44100);
            mfcc.setNumCoefficients(static_cast<int>(MelBins));

            Spectrogram spectrogram(MelBins, samples.size() / HopLength);

            chunks(samples.cbegin(),
                   samples.cend(),
                   NFFT,
                   HopLength,
                   [&gist, &mfcc, &spectrogram, cnt = 0](auto from, auto to) mutable {
                       gist.processAudioFrame({from, to});
                       mfcc.calculateMelFrequencySpectrum(gist.getMagnitudeSpectrum());

                       spectrogram.col(cnt) = Matrix<Real, MelBins, 1>::Map(mfcc.melSpectrum.data());
                       ++cnt;
                   }
            );

            double maxVal = std::max(1e-10, double(spectrogram.maxCoeff()));
            spectrogram = 10.0 * (spectrogram.array() < 1e-10).select(1e-10, spectrogram).array().log10()
                          - 10.0 * log10(maxVal);

            maxVal = spectrogram.maxCoeff();
            spectrogram = (spectrogram.array() < (maxVal - 80.0)).select(maxVal - 80.0, spectrogram);

            return spectrogram;
        }

        static auto calc_frames(const Spectrogram &spectrogram) -> Frames {
            Frames frames(FrameSize, spectrogram.cols() - 2 * W + 1);
            for (size_t i = W, cnt = 0; i < spectrogram.cols() - W + 1; ++i, ++cnt) {
                Matrix<Real, Dynamic, Dynamic, Eigen::RowMajor> x = spectrogram.block(0, i - W, MelBins, FramesContext);
                x.resize(FrameSize, 1);

                frames.col(cnt) = std::move(x);
            }

            return frames;
        }

        static auto calc_cov(const Eigen::Ref<Matrix<Real, Dynamic, Dynamic>> &mat) -> CovarianceMatrix {
            const Matrix<Real, Dynamic, Dynamic> centered = mat.rowwise() - mat.colwise().mean();
            return (centered.adjoint() * centered) / double(mat.rows() - 1);
        }

        static auto calc_filters(const CovarianceMatrix &cov) -> Filters {
            cout << "|----------------------------------------------|" << endl;
            cout << "|             Calculating filters              |" << endl;
            cout << "|----------------------------------------------|" << endl;
            SelfAdjointEigenSolver<CovarianceMatrix> solver(cov);
            return solver.eigenvectors().rowwise().reverse().transpose().block(0, 0, NumOfFilters, FrameSize);
        }

        static auto calc_delta(const Matrix<Real, NumOfFilters, Dynamic> &f) -> Features {
            Matrix<bool, Dynamic, NumOfFilters> delta(f.cols() - T, NumOfFilters);
            for (size_t i = 0; i < f.cols() - T; ++i) {
                delta.row(i) = ((f.col(i) - f.col(i + T)).array() >= 0);
            }

            return delta;
        }

        static auto bool_row_to_num(const Features &f, size_t r) -> N {
            size_t p = 0;
            return f.row(r).reverse().unaryExpr([&p](bool x) {
                return x * pow(2, p++);
            }).sum();
        }

    };

}