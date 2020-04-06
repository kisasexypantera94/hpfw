# hpfw

## Requirements
* Eigen3
* FFTW
* TBB
* essentia
* cereal
* MKL

## Usage
```c++
#include <iostream>

#include <hpfw/audioproblems/live-song-id/live_song_id.h>

using namespace std;

constexpr auto index_dir = "original/";
constexpr auto search_dir = "slices/";

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
```

## Running examples
To configure:

```bash
cmake -S . -B build
```

To run examples
```bash
cmake --build build --target live-id 
./build/examples/live-id
```

## References
* Tsai, T. (2016). Audio Hashprints: Theory & Application. (Doctoral dissertation, EECS Department, University of California, Berkeley).


