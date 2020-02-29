#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>

#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Gist.h>
#include <taskflow/taskflow.hpp>
#include <tbb/concurrent_vector.h>

#include "../helpers.h"
#include "../io/mpg123_wrapper.h"
#include "core.h"

namespace hpfw {

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

    using namespace hpfw::io;
    using namespace helpers;


    /// Algorithm:
    ///
    /// 1. Compute spectrogram. The first step is to compute a time-frequency representation of audio.
    /// This time-frequency representation can be selected to suit the characteristics of the problem at hand.
    /// At the end of this first step, the audio is represented at each frame by a vector of dimension B,
    /// where B is the number of frequency subbands.
    ///
    /// 2. Collect context frames. In addition to looking at the audio frame of interest, we also look at the
    /// neighboring frames to its left and right. When computing the fingerprint at a particular frame, we consider
    /// w frames of context. At the end of this second step, we represent each frame with a vector of dimension Bw.
    ///
    /// 3. Apply spectro-temporal filters. At each frame we apply N different spectro-temporal filters in order
    /// to compute N different spectro-temporal features. Each spectro-temporal feature is a linear combination of
    /// the spectrogram energy values for the current frame and surrounding context frames. The weights of this
    /// linear combination are specified by the coefficients in the spectro-temporal filters. These filters are
    /// learned in an unsupervised manner by solving a sequence of optimization problems.
    /// At the end of this third step, we have N spectro-temporal features per frame.
    ///
    /// 4. Compute deltas. For each of our N features, we compute the change in the feature value over a time lag T.
    /// If the feature value at frame n is given by xn, then the corresponding delta feature will be ∆n = xn − xn+T .
    /// At the end of this fourth step, we have N spectro-temporal delta features per frame.
    ///
    /// 5. Apply threshold. Each of the N delta features is compared to a threshold value of 0, which results in a
    /// binary value. Each of these binary values thus represents whether the delta feature is increasing or
    /// decreasing over time (across the time lag T). At the end of the fifth step, we have N binary values per frame.
    ///
    /// 6. Bit packing. The N binary values are packed into a single 32-bit or 64-bit integer which represents the
    /// hashprint value for a single frame. This compact binary representation will allow us to store fingerprints
    /// in memory efficiently, do reverse indexing, or compute Hamming distance between hashprints very quickly.
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

        /// Process audiofiles, calculate filters and build database.
        void prepare(const vector<string> &filenames) {
            if (db.size() > 0) {
                return;
            }

            build_db(calc_features(preprocess(filenames)));
        }

        struct SearchResult {
            string filename;
            size_t cnt;
            long long offset;
        };

        /// Process audiofile and find best match in database.
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

        /// Dump database and filters.
        auto dump(const std::optional<string> &filename) const -> string {
            const auto dump_name = filename.value_or("db/dump.cereal");
            {
                std::ofstream os(dump_name, std::ios::binary);
                cereal::BinaryOutputArchive archive(os);
                archive(db);
                archive(filters);
            }
            return dump_name;
        }

        /// Create new HashPrint instance from dump.
        static auto from_dump(const std::string &dump_name) -> HashPrint {
            HashPrint hp;
            {
                std::ifstream is(dump_name, std::ios::binary);
                cereal::BinaryInputArchive archive(is);
                archive(hp.db);
                archive(hp.filters);
            }
            return hp;
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

        /// Calculate frames and filters. Steps 1-3.
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

        /// Calculate features. Steps 3-5.
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

        /// Build database. Step 6.
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

        /// Find top N eigen vectors - these are the filters.
        static auto calc_filters(const CovarianceMatrix &cov) -> Filters {
            cout << "|----------------------------------------------|" << endl;
            cout << "|             Calculating filters              |" << endl;
            cout << "|----------------------------------------------|" << endl;
            SelfAdjointEigenSolver<CovarianceMatrix> solver(cov);
            return solver.eigenvectors().rowwise().reverse().transpose().block(0, 0, NumOfFilters, FrameSize);
        }

        /// Calculate deltas and apply threshold.
        static auto calc_delta(const Matrix<Real, NumOfFilters, Dynamic> &f) -> Features {
            Matrix<bool, Dynamic, NumOfFilters> delta(f.cols() - T, NumOfFilters);
            for (size_t i = 0; i < f.cols() - T; ++i) {
                delta.row(i) = (f.col(i) - f.col(i + T)).array() >= 0;
            }

            return delta;
        }

        /// Pack bool row into integer.
        static auto bool_row_to_num(const Features &f, size_t r) -> N {
            size_t p = 0;
            return f.row(r).reverse().unaryExpr([&p](bool x) {
                return x * pow(2, p++);
            }).sum();
        }

    }; // HashPrint

} // hpfw