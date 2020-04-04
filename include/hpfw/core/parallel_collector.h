#pragma once

#include <spdlog/spdlog.h>
#include <taskflow/taskflow.hpp>
#include <tbb/concurrent_vector.h>

namespace hpfw {

    /// ParallelCollector is a parallel aggregation class which uses `HashPrint` tools to calculate fingerprints.
    ///
    /// \tparam Algo - specific `HashPrint` implementation
    /// \tparam Cache - storage for frames, filters and accumulated covariance matrix
    template<typename Algo, template<typename> typename Cache>
    class ParallelCollector {
    public:
        using Fingerprint = typename Algo::Fingerprint;
        using CovarianceMatrix = typename Algo::CovarianceMatrix;
        using Filters = typename Algo::Filters;

        struct FilenameFingerprintPair {
            std::string filename;
            Fingerprint fingerprint;

            template<class Archive>
            void serialize(Archive &ar) {
                ar(filename);
                ar(fingerprint);
            }
        };

        // TODO: cache config
        ParallelCollector() : algo(), cache("cache/") {
            accum_cov.resize(Algo::FrameSize, Algo::FrameSize);
            filters.resize(Algo::NumOfFilters, Algo::FrameSize);
        }

        ~ParallelCollector() = default;

        /// Process audiofiles, calculate filters.
        auto prepare(const std::vector<std::string> &filenames) -> tbb::concurrent_vector<FilenameFingerprintPair> {
            preprocess(filenames);
            return collect_fingerprints();
        }

        auto calc_fingerprint(const std::string &filename) const -> Fingerprint {
            const auto spectro = algo.sh.spectrogram(filename);
            const auto frames = algo.calc_frames(spectro);
            return algo.calc_fingerprint(filters * frames);
        }

        void save() const {
            spdlog::info("Saving cache");

            cache.set_cov(accum_cov);
            cache.set_filters(filters);
        }

        void load() {
            spdlog::info("Loading cache");

            cache.get_cov(accum_cov);
            cache.get_filters(filters);
        }

    private:
        const Algo algo;
        CovarianceMatrix accum_cov;
        Filters filters;
        Cache<Algo> cache;

        /// Calculate frames and filters. Steps 1-3.
        void preprocess(const std::vector<std::string> &filenames) {
            std::mutex mtx;
            tf::Taskflow taskflow;
            taskflow.parallel_for(
                    filenames.cbegin(), filenames.cend(),
                    [this, &mtx](const std::string &filename) {
                        try {
                            spdlog::info("Preprocessing '{}'", filename);

                            const auto spectro = algo.sh.spectrogram(filename);
                            const auto frames = algo.calc_frames(spectro);
                            const auto cov = algo.calc_cov(frames.transpose());
                            {
                                std::scoped_lock lock(mtx);
                                accum_cov += cov;
                            }
                            // Cache spectrogram to calculate features later.
                            // It will also be needed when adding new tracks.
                            cache.set_spectro(filename, spectro);
                        } catch (const std::exception &e) {
                            spdlog::error("Error preprocessing '{}': {}", filename, e.what());
                        }
                    }
            );

            tf::Executor executor;
            executor.run(taskflow).wait();

            spdlog::info("Calculating filters");
            filters = algo.calc_filters(accum_cov / cache.size());
        }

        /// Collect fingerprints. Steps 3-6.
        auto collect_fingerprints() const -> tbb::concurrent_vector<FilenameFingerprintPair> {
            auto spectros = cache.get_spectros();
            tbb::concurrent_vector<FilenameFingerprintPair> fingerprints;
            tf::Taskflow taskflow;
            taskflow.parallel_for(
                    spectros.begin(), spectros.end(),
                    [this, &fingerprints](const std::pair<std::string, typename Algo::Spectrogram> &p) {
                        const auto &[filename, spectro] = p;
                        const auto stem = std::string(std::filesystem::path(filename).stem());
                        spdlog::info("Calculating fingerprints for {}", stem);

                        const auto frames = algo.calc_frames(spectro);
                        fingerprints.push_back({stem, algo.calc_fingerprint(filters * frames)});
                    }
            );

            tf::Executor executor;
            executor.run(taskflow).wait();

            return fingerprints;
        }

    }; // ParallelCollector

} // hpfw