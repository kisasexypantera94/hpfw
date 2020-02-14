#pragma once

template<typename Iterator, typename Func, typename Distance, bool exact = true>
auto chunks(Iterator begin, Iterator end, Distance k, Func f) {
    Iterator chunk_begin;
    Iterator chunk_end;
    chunk_end = chunk_begin = begin;

    do {
        if (std::distance(chunk_end, end) < k) {
            chunk_end = end;
        } else {
            std::advance(chunk_end, k);
        }

        f(chunk_begin, chunk_end);

        chunk_begin = chunk_end;
    } while (std::distance(chunk_begin, end) >= (exact ? k : 1));
}

template<typename T>
struct extract_value_type {
    using value_type = T;
};

template<template<typename> class X, typename T>
struct extract_value_type<X<T>> {
    using value_type = T;
};

