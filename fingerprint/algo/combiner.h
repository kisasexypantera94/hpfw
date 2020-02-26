#pragma once

#include <vector>

#include "../../helpers.h"
#include "tsai.h"
#include "mpg123_wrapper.h"

using std::vector;
using std::string;
using namespace fp::algo;
using namespace dec;

template<typename Algo = HashPrint<>>
class AudioCombiner {
public:
    using fingerprint = std::vector<typename extract_value_type<Algo>::value_type>;

    AudioCombiner() = default;

    ~AudioCombiner() = default;

    void combine(const vector<string> &filenames) {
        algo.calc(filenames);
    }

private:
    Algo algo;
};