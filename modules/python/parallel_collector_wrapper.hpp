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

    auto par_collector_prepare(LiveIdCollector *collector,
                               const char **filenames,
                               int n,
                               int *got) -> FilenameHashprintPair *;

    auto par_collector_calc_hashprint(LiveIdCollector *collector, const char *filename, int *size) -> uint64 *;

    void prepare_result_free(FilenameHashprintPair *res, int got);

    void calc_hashprint_result_free(uint64 *hp);


#ifdef __cplusplus
    }
#endif

}
