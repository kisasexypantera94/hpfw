#pragma once

template<typename Iterator, typename Func>
auto chunks(Iterator begin, Iterator end, long k, Func f, long hop_length) {
    auto chunk_begin = begin;
    auto chunk_end = begin;
    std::advance(chunk_end, k);

    while (std::distance(chunk_begin, end) >= k) {
        f(chunk_begin, chunk_end);

        std::advance(chunk_begin, hop_length);
        std::advance(chunk_end, hop_length);
    }
}

template<typename T>
struct extract_value_type {
    using value_type = T;
};

template<template<typename> typename X, typename N>
struct extract_value_type<X<N>> {
    using value_type = N;
};

template<
        template<typename, size_t, size_t, size_t, typename> class X,
        typename N,
        size_t F,
        size_t M,
        size_t C,
        typename R
>
struct extract_value_type<X<N, F, M, C, R>> {
    using value_type = N;
};

