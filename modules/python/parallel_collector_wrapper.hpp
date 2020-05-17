#include <hpfw/core/hashprint_handle.h>
#include <hpfw/core/parallel_collector.h>
#include <hpfw/core/cache.h>
#include <hpfw/spectrum/cqt.h>

namespace hpfw {

#ifdef __cplusplus
    extern "C" {
#endif

    struct FilenameHashprintPair {
        char *filename;
        uint64 *hashprint;
        int hp_size;
    };

    using DefaultLiveIdAlgoConfig = HashprintHandle<uint64_t, hpfw::spectrum::CQT<>, 20, 80>;
    using LiveIdCollector = ParallelCollector<DefaultLiveIdAlgoConfig, cache::DriveCache>;

    auto par_collector_new() -> LiveIdCollector *;
    void par_collector_del(LiveIdCollector *collector);
    auto par_collector_prepare(LiveIdCollector *collector, const char **filenames, int n) -> FilenameHashprintPair *;

#ifdef __cplusplus
    }
#endif

}
