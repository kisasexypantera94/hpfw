#pragma once

#include <boost/iterator/transform_iterator.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>

#include "hpfw/utils.h"

namespace hpfw::cache {

    template<typename Frames, typename CovarianceMatrix, typename Filters>
    class DriveCache {
    public:
        explicit DriveCache(const std::string &cache) : cache_dir(cache) {
            if (!std::filesystem::exists(cache_dir)) {
                std::filesystem::create_directory(cache_dir);
                std::filesystem::create_directory(cache_dir + "frames");
            }
        }

        ~DriveCache() = default;

        void set_frames(const std::string &filename, const Frames &f) const {
            const auto stem = std::string(std::filesystem::path(filename).stem());
            save(cache_dir + "frames/" + stem, f);
        }

        void set_cov(const CovarianceMatrix &accum_cov) const {
            save(cache_dir + "accum_cov.cereal", accum_cov);
        }

        void set_filters(const Filters &f) const {
            save(cache_dir + "filters.cereal", f);
        }

        auto get_frames() const {
            namespace fs = std::filesystem;

            filenames = utils::get_dir_files(cache_dir + "frames/");

            return boost::make_iterator_range(filenames.cbegin(), filenames.cend()) |
                   boost::adaptors::transformed(load_frame);
        }

        void get_cov(CovarianceMatrix &accum_cov) const {
            load(cache_dir + "accum_cov.cereal", accum_cov);
        }

        void get_filters(Filters &filters) const {
            load(cache_dir + "filters.cereal", filters);
        }

    private:
        const std::string cache_dir;
        mutable std::vector<std::string> filenames;

        template<typename T>
        static void save(const std::string &filename, const T &obj) {
            {
                std::ofstream os(filename, std::ios::binary);
                cereal::BinaryOutputArchive archive(os);
                archive(obj);
            }
        }

        template<typename T>
        static void load(const std::string &filename, T &obj) {
            {
                std::ifstream is(filename, std::ios::binary);
                cereal::BinaryInputArchive archive(is);
                archive(obj);
            }
        }

        static auto load_frame(const std::string &f) -> std::pair<std::string, Frames> {
            Frames frames;
            load(f, frames);
            return std::make_pair(f, std::move(frames));
        }

    };

} // hpfw::cache
