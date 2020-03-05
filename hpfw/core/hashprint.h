#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>
#include <taskflow/taskflow.hpp>
#include <tbb/concurrent_vector.h>

#include "../utils.h"

namespace hpfw {

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
            size_t NFFT = 44100 * 100 / 1000, // TODO: 44100 -> file sample rate
            size_t HopLength = 44100 * 10 / 1000,
            size_t T = 1,
            typename Real = float>
    class HashPrint {
    public:
        HashPrint() {
            Eigen::initParallel();
            essentia::init();
        }

        HashPrint(HashPrint &&hp) : filters(std::move(hp.filters)) {}

        ~HashPrint() {
            essentia::shutdown();
        }

        using Fingerprint = std::vector<N>;
        struct FilenameFingerprintPair {
            std::string filename;
            Fingerprint fingerprint;
        };

        /// Process audiofiles, calculate filters and return fingerprints.
        auto prepare(const std::vector<std::string> &filenames) -> tbb::concurrent_vector<FilenameFingerprintPair> {
            return collect_fingerprints(preprocess(filenames));
        }

        auto calc_fingerprint(const std::string &filename) const -> Fingerprint {
            auto spectro = calc_spectro(filename);
            auto frames = calc_frames(spectro);
            return calc_fingerprint(filters * frames);
        }

        template<class Archive>
        void save(Archive &ar) const {
            ar(filters);
        }

        template<class Archive>
        void load(Archive &ar) {
            ar(filters);
        }

    private:
        static constexpr size_t FrameSize = MelBins * FramesContext;
        static constexpr size_t W = FramesContext / 2;
        static constexpr size_t NumOfFilters = sizeof(N) * 8;

        // TODO: make generic so CQT or MFCC can be easily used
        using Spectrogram = Eigen::Matrix<Real, MelBins, Eigen::Dynamic>;
        using Frames = Eigen::Matrix<Real, FrameSize, Eigen::Dynamic, Eigen::RowMajor>;
        using CovarianceMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>; // to pass static_assert
        using Filters = Eigen::Matrix<Real, NumOfFilters, Eigen::Dynamic>;

        struct FilenameFramesPair {
            std::string filename;
            Frames frames;
        };

        Filters filters;

        /// Calculate frames and filters. Steps 1-3.
        auto preprocess(const std::vector<std::string> &filenames) -> tbb::concurrent_vector<FilenameFramesPair> {
            using std::cout;
            using std::endl;

            cout << "|----------------------------------------------|" << endl;
            cout << "|Calculating spectrograms and covariance matrix|" << endl;
            cout << "|----------------------------------------------|" << endl;
            CovarianceMatrix accum_cov(FrameSize, FrameSize);
            tbb::concurrent_vector<FilenameFramesPair> frames;
            std::mutex mtx;

            tf::Taskflow taskflow;
            taskflow.parallel_for(
                    filenames.cbegin(), filenames.cend(),
                    [&frames, &accum_cov, &mtx](const std::string &filename) {
                        try {
                            const auto spectro = calc_spectro(filename);
                            const auto f = frames.push_back({filename, calc_frames(spectro)});
                            const auto cov = calc_cov(f->frames.transpose());
                            {
                                std::scoped_lock lock(mtx);
                                std::cout << filename << std::endl;
                                accum_cov.noalias() += cov;
                            }
                        } catch (const std::exception &e) {
                            std::cerr << e.what() << std::endl;
                        }
                    }
            );

            tf::Executor executor;
            executor.run(taskflow).wait();

            filters = calc_filters(accum_cov / frames.size());

            return frames;
        }

        /// Collect fingerprints. Steps 3-6.
        template<typename Iterable>
        auto collect_fingerprints(const Iterable &frames) const -> tbb::concurrent_vector<FilenameFingerprintPair> {
            using std::cout;
            using std::endl;

            cout << "|----------------------------------------------|" << endl;
            cout << "|            Calculating features              |" << endl;
            cout << "|----------------------------------------------|" << endl;
            tbb::concurrent_vector<FilenameFingerprintPair> fingerprints;
            tf::Taskflow taskflow;
            taskflow.parallel_for(
                    frames.cbegin(), frames.cend(),
                    [this, &fingerprints](const FilenameFramesPair &f) {
                        fingerprints.push_back({f.filename, calc_fingerprint(filters * f.frames)});
                    }
            );

            tf::Executor executor;
            executor.run(taskflow).wait();

            return fingerprints;
        }


        static auto calc_spectro(const std::string &filename) -> Spectrogram {
            using namespace essentia;
            using namespace essentia::standard;
            using std::vector;
            using uptr = std::unique_ptr<Algorithm>;

            AlgorithmFactory &factory = standard::AlgorithmFactory::instance();

            auto audio = uptr(factory.create("MonoLoader",
                                             "filename", filename,
                                             "sampleRate", 44100));

            auto fc = uptr(factory.create("FrameCutter",
                                          "frameSize", int(NFFT),
                                          "hopSize", int(HopLength)));

            auto w = uptr(factory.create("Windowing",
                                         "type", "hann"));

            auto spec = uptr(factory.create("Spectrum"));

            auto melbands = uptr(factory.create("MelBands",
                                                "inputSize", 2206,
                                                "numberBands", int(MelBins)));

            // MonoLoader -> FrameCutter -> Windowing -> Spectrum -> MelBands
            vector<float> audioBuffer;
            audio->output("audio").set(audioBuffer);

            vector<float> frame, windowedFrame;
            fc->input("signal").set(audioBuffer);
            fc->output("frame").set(frame);

            w->input("frame").set(frame);
            w->output("frame").set(windowedFrame);

            vector<float> spectrum, bands;
            spec->input("frame").set(windowedFrame);
            spec->output("spectrum").set(spectrum);

            melbands->input("spectrum").set(spectrum);
            melbands->output("bands").set(bands);


            audio->compute();

            Spectrogram spectrogram(MelBins, 0);
            while (true) {

                // compute a frame
                fc->compute();

                // if it was the last one (ie: it was empty), then we're done.
                if (frame.empty()) {
                    break;
                }

                // if the frame is silent, just drop it and go on processing
                if (isSilent(frame)) {
                    continue;
                }

                w->compute();
                spec->compute();
                melbands->compute();

                spectrogram.conservativeResize(spectrogram.rows(), spectrogram.cols() + 1);
                spectrogram.col(spectrogram.cols() - 1) = Eigen::Matrix<Real, MelBins, 1>::Map(bands.data());
            }

            double maxVal = std::max(1e-10, double(spectrogram.maxCoeff()));
            spectrogram = 10.0 * (spectrogram.array() < 1e-10).select(1e-10, spectrogram).array().log10()
                          - 10.0 * log10(maxVal);

            maxVal = spectrogram.maxCoeff();
            spectrogram = (spectrogram.array() < (maxVal - 80.0)).select(maxVal - 80.0, spectrogram);

            return spectrogram;
        }

        static auto calc_frames(const Spectrogram &spectrogram) -> Frames {
            using Eigen::Matrix;
            using Eigen::Dynamic;
            using Eigen::RowMajor;

            Frames frames(FrameSize, spectrogram.cols() - 2 * W + 1);
            for (size_t i = W, cnt = 0; i < spectrogram.cols() - W + 1; ++i, ++cnt) {
                Matrix<Real, Dynamic, Dynamic, RowMajor> x = spectrogram.block(0, i - W, MelBins, FramesContext);
                x.resize(FrameSize, 1);

                frames.col(cnt) = std::move(x);
            }

            return frames;
        }

        static auto calc_cov(const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &mat) -> CovarianceMatrix {
            using Eigen::Matrix;
            using Eigen::Dynamic;

            Matrix<Real, Dynamic, Dynamic> centered = mat.rowwise() - mat.colwise().mean();
            return (centered.adjoint() * centered) / double(mat.rows() - 1);
        }

        /// Find top N eigen vectors - these are the filters.
        static auto calc_filters(const CovarianceMatrix &cov) -> Filters {
            using std::cout;
            using std::endl;

            cout << "|----------------------------------------------|" << endl;
            cout << "|             Calculating filters              |" << endl;
            cout << "|----------------------------------------------|" << endl;
            Eigen::SelfAdjointEigenSolver<CovarianceMatrix> solver(cov);
            return solver.eigenvectors().rowwise().reverse().transpose().block(0, 0, NumOfFilters, FrameSize);
        }

        /// Calculate deltas and apply threshold.
        static auto calc_fingerprint(const Eigen::Matrix<Real, NumOfFilters, Eigen::Dynamic> &f) -> Fingerprint {
            using Eigen::Matrix;
            using Eigen::Dynamic;

            Fingerprint fp(f.cols() - T);
            for (size_t i = 0; i < f.cols() - T; ++i) {
                fp[i] = bool_col_to_num((f.col(i) - f.col(i + T)).array() >= 0);
            }

            return fp;
        }

        /// Pack bool column into integer.
        static auto bool_col_to_num(const Eigen::Matrix<bool, NumOfFilters, 1> &c) -> N {
            size_t p = 0;
            return c.reverse().unaryExpr([&p](bool x) -> N {
                return x * pow(2, p++);
            }).sum();
        }

    }; // HashPrint

} // hpfw