#pragma once

#include <map>
#include <unordered_map>

#include <cereal/types/vector.hpp>
#include <tbb/concurrent_vector.h>

#include <annoylib.h>
#include <kissrandom.h>

namespace hpfw::db {

    template<typename Collector>
    class AnnStorage {
    public:
        struct SearchResult {
            std::string filename;
            float cnt;
            int64_t offset; // TODO: refactor types
        };

        AnnStorage() : index(64) {}

        ~AnnStorage() = default;

        void build(const tbb::concurrent_vector<typename Collector::FilenameFingerprintPair> &hashprints) {
            spdlog::info("Building index");
            int32_t n = index.get_n_items() + 1;
            for (const auto &[filename, hashprint] : hashprints) {
                for (int64_t i = 0; i < hashprint.size(); ++i) {
                    song_dict[n] = {filename, i};
                    index.add_item(n, &hashprint[i]);
                    ++n;
                }
            }
            index.build(3000);
            spdlog::info("Size of index: {}", index.get_n_items());
        }

        auto find(const typename Collector::Hashprint &hashprint) -> SearchResult {
            std::unordered_map<std::string, std::unordered_map<int64_t, float>> cnt;
            SearchResult best_match = {"", 0, 0};

            for (int64_t i = 0; i < hashprint.size(); ++i) {
                std::vector<int32_t> results;
                std::vector<uint64> distances;
                index.get_nns_by_vector(&hashprint[i], 5, 5000, &results, &distances);

                for (int64_t j = 0; j < results.size(); ++j) {
                    FilenameOffsetPair p = song_dict[results[j]];
                    int64_t offset = i - p.offset;
                    cnt[p.filename][offset] += 1.0 / (float(distances[j] + 1));
                    float cur = cnt[p.filename][offset];
                    if (cur > best_match.cnt) {
                        best_match.filename = p.filename;
                        best_match.offset = offset;
                        best_match.cnt = cur;
                    }
                }
            }
            return best_match;
        }

        /// Save database and algorithm handler.
        auto save(const std::optional<std::string> &filename) -> std::string {
            const auto dump_name = filename.value_or("db/dump.cereal");
            index.save(dump_name.c_str());
            return dump_name;
        }

        /// Load database and algorithm handler.
        auto load(const std::string &dump_name) -> AnnStorage & {
            index.load(dump_name.c_str());
            return *this;
        }

    private:
        struct FilenameOffsetPair {
            std::string filename;
            int64_t offset;
        };

        AnnoyIndex<int32_t, uint64_t, Hamming, Kiss64Random> index;
        std::unordered_map<int32_t, FilenameOffsetPair> song_dict;
    };

} // hpfw::db