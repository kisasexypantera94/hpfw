#include <filesystem>
#include <iostream>

#include "fingerprint/handle.h"

using namespace std;
using namespace fp;
namespace fs = std::filesystem;

int main() {
    FingerprintHandle<size_t, dec::Mpg123Wrapper, fp::algo::Shazam> h;

    auto dir = "/Users/chingachgook/dev/rust/khalzam/resources";
    auto it = fs::directory_iterator(dir);
    vector<string> files;
    for (const auto& f : it) {
        files.push_back(f.path());
    }

    auto res = h.calc_fingerprints(files);
    for (const auto& x : res) {
       for (const auto& y : x) {
           cout << y << " ";
       }
       cout << endl;
    }

    return 0;
}
