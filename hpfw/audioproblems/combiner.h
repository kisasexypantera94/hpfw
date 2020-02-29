#pragma once

#include <vector>
#include <filesystem>

#include "../helpers.h"
#include "../core/core.h"

namespace hpfw {

    namespace fs = std::filesystem;

    using std::vector;
    using std::string;

    template<typename Algo = HashPrint<>>
    class AudioCombiner {
    public:
        AudioCombiner() = default;

        AudioCombiner(Algo &&algo) : algo(std::move(algo)) {}

        ~AudioCombiner() = default;

        void combine(const vector<string> &filenames) {
            algo.prepare(filenames);

            auto it = fs::directory_iterator("/Users/chingachgook/dev/rust/khalzam/samples");
            for (const auto &f : it) {
                const auto &filename = f.path();
                if (filename.extension() != ".mp3") {
                    continue;
                }

                auto res = algo.find(filename);
                cout << res.filename << " " << res.cnt << " " << res.offset << endl << endl;
            }

            algo.dump(std::nullopt);
        }

    private:
        Algo algo;
    };

} // hpfw

