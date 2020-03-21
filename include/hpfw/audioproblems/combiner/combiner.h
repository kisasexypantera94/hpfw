#pragma once

#include <vector>
#include <filesystem>

#include "hpfw/core/hashprint.h"
#include "hpfw/spectrum/mel.h"
#include "hpfw/utils.h"

namespace hpfw {

    using DefaultCombinerAlgoConfig = HashPrint<uint16_t, spectrum::MelSpectrogram<>, 32, 50>;

    template<typename Algo = DefaultCombinerAlgoConfig>
    class AudioCombiner {
    public:
        AudioCombiner() = default;

        AudioCombiner(Algo &&algo) : algo(std::move(algo)) {}

        ~AudioCombiner() = default;

        void combine(const std::vector<std::string> &filenames) {
            if (db.size() == 0) {
                build_db(algo.prepare(filenames));
            }

            for (const auto &f : filenames) {
                auto res = find(f);
                std::cout << res.filename << " " << res.cnt << " " << res.confidence << " " << res.offset << std::endl
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
        auto load(const std::string &dump_name) -> AudioCombiner & {
            {
                std::ifstream is(dump_name, std::ios::binary);
                cereal::BinaryInputArchive archive(is);
                archive(db);
                archive(algo);
            }

            return *this;
        }

    private:
        struct FilenameOffsetPair {
            std::string filename;
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

        struct SearchResult {
            std::string filename;
            size_t cnt;
            size_t confidence;
            long long offset;
        };

        using N = typename utils::extract_value_type<Algo>::value_type;
        using DB = std::unordered_map<N, std::vector<FilenameOffsetPair>>;

        DB db;
        Algo algo;

        void build_db(const tbb::concurrent_vector<typename Algo::FilenameFingerprintPair> &fingerprints) {
            for (const auto &[filename, fp]: fingerprints) {
                for (size_t i = 0, num_cols = fp.size(); i < num_cols; ++i) {
                    const N n = fp[i];
                    db[n].push_back({filename, i});
                }
            }
        }

        /// Process audiofile and find best match in database.
        auto find(const std::string &filename) -> SearchResult {
            using std::map;
            using std::string;

            // TODO: use logging framework
            std::cout << "FINDING " << filename << std::endl;

            auto fingerprint = algo.calc_fingerprint(filename);

            SearchResult res = {"", 0, 0, 0};
            map<string, map<long long, size_t>> cnt;

            for (long long c = 0, num_cols = fingerprint.size(); c < num_cols; ++c) {
                const N n = fingerprint[c];

                for (const auto &[fname, offset] : db[n]) {
                    if (fname == filename) {
                        continue;
                    }
                    long long diff = c - offset;
                    auto count = ++cnt[fname][diff];
                    if (count > res.confidence) {
                        if (fname != res.filename) {
                            res = {fname, count, 1, diff};
                        } else {
                            res = {fname, count, res.confidence + 1, diff};
                        }
                    }
                }
            }

            return res;
        }

    }; // AudioCombiner

} // hpfw

