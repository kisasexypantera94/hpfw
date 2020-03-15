#define NDEBUG

#include <filesystem>
#include <iostream>

#include <hpfw/audioproblems/combiner.h>

namespace fs = std::filesystem;
using namespace std;

constexpr auto combine_dir = "/Users/chingachgook/dev/rust/khalzam/resources";

auto get_filenames(const std::string &dir) {
    auto it = fs::directory_iterator(dir);
    std::vector<std::string> files;
    for (const auto &f : it) {
        const auto &filename = f.path();
        if (filename.extension() != ".wav") {
            continue;
        }

        files.emplace_back(filename);
    }

    return files;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    auto files = get_filenames(combine_dir);

    hpfw::AudioCombiner combiner;
    combiner.combine(files);

    return 0;
}
