#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>

#include "fingerprint/algo/combiner.h"

namespace fs = std::filesystem;
using namespace std;
using Eigen::MatrixXi;
using Eigen::Matrix;

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

    return mat;
}

int main1() {
//    MatrixXi kek = foo();
//    MatrixXi kek2 = foo();
//    cout << kek << endl;
//
//    cout << (kek2.array() != kek.array()).count() << endl;

    Matrix<bool, 4, 4> mat = Matrix<bool, 4, 4>::Random(4, 4);
    cout << mat << endl;
    size_t p = 0;
    cout << mat.row(1) << endl;
    int n = mat.row(1).reverse().unaryExpr([&p](bool x) {
        return x * pow(2, p++);
    }).sum();
    cout << n << endl;
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
