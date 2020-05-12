#pragma once

#include <cereal/types/vector.hpp>
#include <tbb/concurrent_vector.h>

namespace hpfw::db {

    template<typename Collector>
    class MemoryStorage {
    public:
        struct SearchResult {
            std::string filename;
            size_t cnt;
            int64_t offset; // TODO: refactor types
        };

        MemoryStorage() = default;

        ~MemoryStorage() = default;

        void build(tbb::concurrent_vector<typename Collector::FilenameFingerprintPair> &&hashprints) {
            // TODO: implement move constructor if needed at all
            db = DB(std::make_move_iterator(hashprints.begin()),
                    std::make_move_iterator(hashprints.end()));
        }

        auto find(const typename Collector::Hashprint &hp) const -> SearchResult {
            SearchResult res = {"", std::numeric_limits<size_t>::max(), 0};
            // TODO: maybe parallelize search
            for (const auto &[ref_filename, ref_hp] : db) {
                uint64_t best_distance = std::numeric_limits<size_t>::max();
                int64_t best_offset = 0;

                size_t n = ref_hp.size();
                size_t k = hp.size();
                if (n < k) {
                    k = n;
                }

                for (size_t i = 0; i < n - k + 1; ++i) {
                    size_t start_col = i;
                    size_t end_col = start_col + k - 1;

                    size_t cnt = 0;
                    for (size_t j = 0; j < k; ++j) {
                        cnt += __builtin_popcountll(
                                hp[j] ^ ref_hp[start_col + j]); // TODO: implement templated popcount
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

        /// Save database and algorithm handler.
        auto save(const std::optional<std::string> &filename) const -> std::string {
            const auto dump_name = filename.value_or("db/dump.cereal");
            {
                std::ofstream os(dump_name, std::ios::binary);
                cereal::BinaryOutputArchive archive(os);
                archive(db);
            }
            return dump_name;
        }

        /// Load database and algorithm handler.
        auto load(const std::string &dump_name) -> MemoryStorage & {
            {
                std::ifstream is(dump_name, std::ios::binary);
                cereal::BinaryInputArchive archive(is);
                archive(db);
            }

            return *this;
        }

    private:
        // TODO: think about metric trees, so that hashes with hamming distance less than K could be easily found
        using DB = std::vector<typename Collector::FilenameFingerprintPair>;

        DB db;
    };

} // hpfw::db