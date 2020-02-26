#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>

#include "fingerprint/algo/combiner.h"

namespace fs = std::filesystem;
using namespace std;
using Eigen::MatrixXi;

constexpr auto dir = "/Users/chingachgook/dev/rust/khalzam/resources";

MatrixXi foo() {
    MatrixXi mat = MatrixXi::Random(4, 4);
    cout << &mat << endl;

    cout << mat << endl << endl;

    MatrixXi frames(12, mat.cols() - 2 * 1 + 1);
    for (size_t i = 1, cnt = 0; i < mat.cols() - 1 + 1; ++i, ++cnt) {
        frames.col(cnt) = Matrix<int, 12, 1>(
                mat.block(0, i - 1, 4, 2).data()
        );
    }

    cout << &frames << endl;

    return frames;
}

int main1() {
    MatrixXi kek = foo();
    cout << kek << endl;
}

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
