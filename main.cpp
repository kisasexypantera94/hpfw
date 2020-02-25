#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>

#include "fingerprint/algo/combiner.h"

namespace fs = std::filesystem;

constexpr auto dir = "/Users/chingachgook/dev/rust/khalzam/resources";

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    auto it = fs::directory_iterator(dir);
    std::vector<std::string> files;
    for (const auto &f : it) {
        const auto &filename = f.path();
        if (filename.extension() != ".mp3") {
            continue;
        }

        files.emplace_back(filename);
    }

    AudioCombiner combiner;
    combiner.combine(files);

    return 0;
}
