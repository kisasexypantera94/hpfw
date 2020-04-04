#define NDEBUG

#include <filesystem>
#include <iostream>

#include <hpfw/audioproblems/live-song-id/live_song_id.h>

namespace fs = std::filesystem;
using namespace std;

constexpr auto index_dir = "/Users/chingachgook/dev/QtProjects/hpfw/original";
constexpr auto search_dir = "/Users/chingachgook/dev/QtProjects/hpfw/slices";

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    auto index_files = hpfw::utils::get_dir_files(index_dir);
    auto search_files = hpfw::utils::get_dir_files(search_dir);
    sort(search_files.begin(), search_files.end());

    hpfw::LiveSongIdentification liveid;
    liveid.index(index_files);
    liveid.search(search_files);

    return 0;
}
