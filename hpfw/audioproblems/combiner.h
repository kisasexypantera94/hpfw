#pragma once

#include <vector>
#include <filesystem>

#include "../utils.h"
#include "../core/hashprint.h"

namespace hpfw {

    template<typename Algo = HashPrint<>>
    class AudioCombiner {
    public:
        AudioCombiner() = default;

        AudioCombiner(Algo &&algo) : algo(std::move(algo)) {}

        ~AudioCombiner() = default;

        void combine(const std::vector<std::string> &filenames) {
            algo.prepare(filenames);

            auto it = std::filesystem::directory_iterator("/Users/chingachgook/dev/rust/khalzam/samples");
            for (const auto &f : it) {
                const auto &filename = f.path();
                if (filename.extension() != ".mp3") {
                    continue;
                }

                auto res = algo.find(filename);
                std::cout << res.filename << " " << res.cnt << " " << res.confidence << " " << res.offset << std::endl << std::endl;
            }

            algo.dump(std::nullopt);
        }

    private:
        Algo algo;
    };

} // hpfw

