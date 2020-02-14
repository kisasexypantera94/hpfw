#pragma once

#include <exception>
#include <mutex>
#include <vector>

#include <mpg123.h>

namespace dec {

    std::once_flag flag;

    class Mpg123Wrapper {
    public:
        Mpg123Wrapper();

        ~Mpg123Wrapper();

        auto decode(const std::string &filename) -> std::vector<float>;

    private:
        static void init();

        static constexpr size_t part_size = 1024;
        mpg123_handle *mh;
    };

    void Mpg123Wrapper::init() {
        if (mpg123_init() != MPG123_OK) {
            std::throw_with_nested(std::runtime_error("error initializing mpg123_init"));
        }
    }

    Mpg123Wrapper::Mpg123Wrapper() {
        std::call_once(flag, init);

        mh = mpg123_new(nullptr, nullptr);
        if (mh == nullptr) {
            std::throw_with_nested(std::runtime_error("error creating mpg123 handler"));
        }

        auto flags = MPG123_MONO_MIX | MPG123_QUIET | MPG123_FORCE_FLOAT;
        if (mpg123_param(mh, MPG123_FLAGS, flags, 0.) != MPG123_OK) {
            std::throw_with_nested(std::runtime_error("error adding parameters"));
        }
    }

    Mpg123Wrapper::~Mpg123Wrapper() {
        if (mpg123_close(mh) != MPG123_OK) {
            std::throw_with_nested(std::runtime_error("error closing handler"));
        }

        mpg123_delete(mh);
    }

    auto Mpg123Wrapper::decode(const std::string &filename) -> std::vector<float> {
        if (mpg123_open(mh, filename.c_str()) != MPG123_OK) {
            std::throw_with_nested(std::runtime_error("error opening file " + filename));
        }

        long rate;
        int channels, encoding;
        if (mpg123_getformat(mh, &rate, &channels, &encoding) != MPG123_OK) {
            std::throw_with_nested(std::runtime_error("error getting format from " + filename));
        }

        unsigned char part[part_size];
        size_t bytes_read;

        std::vector<float> samples;
        size_t bytes_processed = 0;

        do {
            auto err = mpg123_read(mh, part, part_size, &bytes_read);
            samples.resize((bytes_processed + bytes_read) / 4 + 1);
            memcpy((unsigned char *) samples.data() + bytes_processed, part, bytes_read);
            bytes_processed += bytes_read;

            if (err == MPG123_DONE) {
                break;
            }

            if (err != MPG123_OK) {
                break;
            }
        } while (bytes_read > 0);

        samples.resize(bytes_processed / 4);

        if (mpg123_close(mh) != MPG123_OK) {
            std::throw_with_nested(std::runtime_error("error closing " + filename));
        }

        return samples;
    }

}
