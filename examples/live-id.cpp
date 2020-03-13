#define NDEBUG

#include <filesystem>
#include <iostream>

#include "hpfw/audioproblems/live_song_id.h"

namespace fs = std::filesystem;
using namespace std;

constexpr auto index_dir = "/Users/chingachgook/dev/rust/khalzam/resources";
constexpr auto search_dir = "/Users/chingachgook/dev/rust/khalzam/samples";

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

    auto index_files = get_filenames(index_dir);
    auto search_files = get_filenames(search_dir);

    hpfw::LiveSongIdentification liveid;
    liveid.index(index_files); // or liveid.load("dbname");
    liveid.search(search_files);

    return 0;
}
