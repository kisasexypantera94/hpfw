#include "parallel_collector_wrapper.hpp"

namespace hpfw {

    auto par_collector_new() -> LiveIdCollector * {
        return new LiveIdCollector();
    }

    void par_collector_del(LiveIdCollector *collector) {
        delete collector;
    }

    auto par_collector_prepare(LiveIdCollector *collector,
                               const char **filenames,
                               int n,
                               int *got) -> FilenameHashprintPair * {
        auto hashprints = collector->prepare({filenames, filenames + n});
        *got = static_cast<int>(hashprints.size());

        auto hp = new FilenameHashprintPair[hashprints.size()];
        for (size_t i = 0; i < hashprints.size(); ++i) {
            auto str = hashprints[i].filename;
            hp[i].hp_size = hashprints[i].fingerprint.size();

            hp[i].filename = new char[str.size() + 1];
            hp[i].hashprint = new uint64[hp[i].hp_size];

            std::copy(hashprints[i].filename.begin(), hashprints[i].filename.end(), hp[i].filename);
            std::copy(hashprints[i].fingerprint.begin(), hashprints[i].fingerprint.end(), hp[i].hashprint);
            hp[i].filename[str.size()] = '\0';
        }

        return hp;
    }

    void prepare_result_free(FilenameHashprintPair *res, int got) {
        for (size_t i = 0; i < got; ++i) {
            delete[] res[i].filename;
            delete[] res[i].hashprint;
        }
        delete[] res;
    }

    auto par_collector_calc_hashprint(LiveIdCollector *collector, const char *filename, int *size) -> uint64 * {
        auto hashprint = collector->calc_hashprint(filename);
        *size = static_cast<int>(hashprint.size());
        auto hp = new uint64[hashprint.size()];
        std::copy(hashprint.begin(), hashprint.end(), hp);
        return hp;
    }

    void calc_hashprint_result_free(uint64 *hp) {
        delete[] hp;
    }

    void par_collector_save(LiveIdCollector *collector, const char *cache) {
        collector->save();
    }

    void par_collector_load(LiveIdCollector *collector, const char *cache) {
        collector->load();
    }

}
