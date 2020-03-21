#pragma once

#include <limits>
#include <vector>

#include "hpfw/core/hashprint.h"
#include "hpfw/spectrum/cqt.h"
#include "hpfw/core/cache.h"
#include "storage.h"
#include "hpfw/core/collector.h"

namespace hpfw {

    using DefaultLiveIdAlgoConfig = HashPrint<uint64_t, spectrum::CQT<>, 20, 80>;
    using DefaultLiveIdCollector = ParallelCollector<DefaultLiveIdAlgoConfig, cache::DriveCache<DefaultLiveIdAlgoConfig>>;

    template<typename Collector = DefaultLiveIdCollector,
            typename Storage = db::MemoryStorage<DefaultLiveIdCollector>>
    class LiveSongIdentification {
    public:
        LiveSongIdentification() {
            collector.load();
        }

        ~LiveSongIdentification() {
            collector.save();
        }

        void index(const std::vector<std::string> &filenames) {
            storage.build(collector.prepare(filenames));
        }

        auto search(const std::vector<std::string> &filenames) {
            uint16_t cnt_wrong = 0;
            for (const auto &f: filenames) {
                std::cout << "Finding " << f << std::endl;
                auto res = storage.find(collector.calc_fingerprint(f));
                auto res_name = std::string(std::filesystem::path(res.filename).stem());
                if (f.find(res_name) == std::string::npos) {
                    ++cnt_wrong;
                }
                std::cout << res.filename << " " << res.cnt << " " << res.offset << std::endl
                          << std::endl;
            }
            std::cout << cnt_wrong << " " << 1 - cnt_wrong / float(filenames.size()) << std::endl;
        }


    private:
        Collector collector;
        Storage storage;
    };

}

