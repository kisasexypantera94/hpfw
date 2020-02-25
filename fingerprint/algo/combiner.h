#pragma once

#include <vector>

#include <tbb/parallel_for_each.h>
#include <tbb/concurrent_vector.h>

#include "../../helpers.h"
#include "tsai.h"
#include "mpg123_wrapper.h"

using std::vector;
using std::string;
using namespace fp::algo;
using namespace dec;

template<typename Algo = HashPrint<>, typename Decoder = Mpg123Wrapper>
class AudioCombiner {
public:
    using fingerprint = std::vector<typename extract_value_type<Algo>::value_type>;

    AudioCombiner() = default;

    ~AudioCombiner() = default;

    auto combine(const vector<string> &filenames) const {
        algo.calc(filenames);
    }

private:
    Algo algo;
};