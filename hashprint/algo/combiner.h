#pragma once

#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

#include "../../helpers.h"
#include "core.h"

using std::vector;
using std::string;
using namespace fp::algo;
using namespace dec;

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