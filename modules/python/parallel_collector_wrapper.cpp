#include "parallel_collector_wrapper.hpp"

namespace hpfw {

    auto par_collector_new() -> LiveIdCollector * {
        return new LiveIdCollector();
    }

    void par_collector_del(LiveIdCollector *collector) {
        delete collector;
    }

    auto par_collector_prepare(LiveIdCollector *collector, const char **filenames, int n) -> FilenameHashprintPair * {
        const std::vector<std::string> f(filenames, filenames + n);
        auto hashprints = collector->prepare(f);

        auto hp = new FilenameHashprintPair[hashprints.size()];
        for (size_t i = 0; i < hashprints.size(); ++i) {
            auto str = hashprints[i].filename;
            std::vector<char> cstr(str.c_str(), str.c_str() + str.size() + 1);
            hp[i].filename = str.data();
            hp[i].hp_size = hashprints[i].fingerprint.size();
            hp[i].hashprint = new uint64[hp[i].hp_size];
            std::copy(hashprints[i].fingerprint.begin(), hashprints[i].fingerprint.end(), hp[i].hashprint);
        }

        return hp;
    }
}
