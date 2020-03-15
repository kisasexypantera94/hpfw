#pragma once

#include <limits>
#include <vector>

#include <cereal/types/vector.hpp>

#include "hpfw/core/hashprint.h"
#include "hpfw/spectrum/cqt.h"

namespace hpfw {

    using DefaultLiveIdAlgoConfig = HashPrint<uint64_t, spectrum::CQT<>, 20, 80>;

    template<typename Algo = DefaultLiveIdAlgoConfig>
    class LiveSongIdentification {
    public:
        LiveSongIdentification() = default;

        LiveSongIdentification(Algo &&algo) : algo(std::move(algo)) {}

        ~LiveSongIdentification() = default;

        auto index(const std::vector<std::string> &filenames) {
            build_db(algo.prepare(filenames));
        }

        auto search(const std::vector<std::string> &filenames) {
            for (const auto &f: filenames) {
                std::cout << "Finding " << f << std::endl;
                auto res = find(f);
                std::cout << res.filename << " " << res.cnt << " " << res.offset << std::endl
                          << std::endl;
            }
        }

        /// Save database and algorithm handler.
        auto save(const std::optional<std::string> &filename) const -> std::string {
            const auto dump_name = filename.value_or("db/dump.cereal");
            {
                std::ofstream os(dump_name, std::ios::binary);
                cereal::BinaryOutputArchive archive(os);
                archive(db);
                archive(algo);
            }
            return dump_name;
        }

        /// Load database and algorithm handler.
        auto load(const std::string &dump_name) -> LiveSongIdentification & {
            {
                std::ifstream is(dump_name, std::ios::binary);
                cereal::BinaryInputArchive archive(is);
                archive(db);
                archive(algo);
            }

            return *this;
        }

    private:
        struct SearchResult {
            std::string filename;
            size_t cnt;
            int64_t offset; // TODO: refactor types
        };

        using N = typename utils::extract_value_type<Algo>::value_type;
        // TODO: think about metric trees, so that hashes with hamming distance less than K could be easily found
        using DB = std::vector<typename Algo::FilenameFingerprintPair>;

        // TODO: store frames and accumulated covariance matrix for easy db rebuilding
        DB db;
        Algo algo;

        void build_db(tbb::concurrent_vector<typename Algo::FilenameFingerprintPair> &&fingerprints) {
            // TODO: implement move constructor if needed at all
            db = DB(std::make_move_iterator(fingerprints.begin()),
                    std::make_move_iterator(fingerprints.end()));
        }

        auto find(const std::string &filename) const -> SearchResult {
            auto fp = algo.calc_fingerprint(filename);

            SearchResult res = {"", std::numeric_limits<size_t>::max(), 0};
            // TODO: maybe parallelize search
            for (const auto &[ref_filename, ref_fp] : db) {
                uint64_t best_distance = std::numeric_limits<size_t>::max();
                int64_t best_offset = 0;

                size_t n = ref_fp.size();
                size_t k = fp.size();
                if (n < k) {
                    k = n;
                }

                for (size_t i = 0; i < n - k + 1; ++i) {
                    size_t start_col = i;
                    size_t end_col = start_col + k - 1;

                    size_t cnt = 0;
                    for (long long j = 0; j < k; ++j) {
                        cnt += __builtin_popcountll(
                                fp[j] ^ ref_fp[start_col + j]); // TODO: implement templated popcount
                    }

                    if (cnt < best_distance) {
                        best_distance = cnt;
                        best_offset = int64_t(i);
                    }
                }

                if (best_distance < res.cnt) {
                    res.cnt = best_distance;
                    res.offset = best_offset;
                    res.filename = ref_filename;
                }
            }

            return res;
        }
    };

}

