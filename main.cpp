#include <algorithm>
#include <execution>
#include <filesystem>
#include <iostream>
#include <map>

#include "fingerprint/handle.h"
#include "fingerprint/algo/shazam.h"

namespace fs = std::filesystem;

using fp::FingerprintHandle;
using fp::fingerprint;
using fp::algo::Shazam;
using dec::Mpg123Wrapper;

constexpr auto dir = "/Users/chingachgook/dev/rust/khalzam/resources";

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    const FingerprintHandle<Shazam<size_t>, Mpg123Wrapper> h;

    auto it = fs::directory_iterator(dir);
    std::vector<std::string> files;
    for (const auto &f : it) {
        files.push_back(f.path());
    }

    std::vector<fingerprint<size_t>> res(files.size());

    std::mutex mtx;
    std::transform(
            std::execution::par,
            files.cbegin(),
            files.cend(),
            res.begin(),
            [&h, &mtx](const std::string &file) {
                {
                    std::scoped_lock lock(mtx);
                    std::cout << file << std::endl;
                }

                return h.calc_fingerprint(file);
            }
    );

    return 0;
}
