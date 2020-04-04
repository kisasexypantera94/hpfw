#pragma once

#include <Eigen/Dense>
#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>

#include "convert.h"

namespace hpfw::spectrum {

    /// CQT is a wrapper class around essentia extractor.
    ///
    /// \tparam SampleRate - sample rate of audiofile.
    /// \tparam HopLength - number of samples between successive CQT columns.
    /// \tparam BinsPerOctave - number of bins per octave.
    /// \tparam NumberBins - number of frequency bins, starting at C3.
    /// \tparam DownsampleFactor - hop size between CQT columns.
    template<size_t SampleRate = 44100, // TODO: maybe pass parameter to constructor instead
            size_t HopLength = 96,
            size_t BinsPerOctave = 24,
            size_t NumberBins = 121,
            size_t DownsampleFactor = 3>
    class CQT {
    public:
        using Spectrogram = Eigen::Matrix<float, NumberBins, Eigen::Dynamic>;

        CQT() {
            essentia::init();
        }

        ~CQT() {
            // TODO: unsafe when there is more than one object, maybe use abstract class
            essentia::shutdown();
        }

        static auto spectrogram(const std::string &filename) -> Spectrogram {
            using namespace essentia;
            using namespace essentia::standard;
            using std::vector;
            using std::complex;
            using uptr = std::unique_ptr<Algorithm>;

            AlgorithmFactory &factory = standard::AlgorithmFactory::instance();

            auto audio = uptr(factory.create("MonoLoader",
                                             "filename", filename,
                                             "sampleRate", int(SampleRate)));

            vector<float> audioBuffer;
            audio->output("audio").set(audioBuffer);

            audio->compute();

            auto cqt = uptr(factory.create("NSGConstantQ",
                                           "inputSize", int(audioBuffer.size()),
                                           "gamma", 0,
                                           "binsPerOctave", int(BinsPerOctave),
                                           "minimumWindow", int(HopLength),
                                           "window", "hann",
                                           "minFrequency", 130.81, // C3
                                           "maxFrequency", 4186.01)); // C8

            vector<vector<complex<float>>> spectrum;
            vector<complex<float>> tmp1;
            vector<complex<float>> tmp2;
            cqt->input("frame").set(audioBuffer);
            cqt->output("constantq").set(spectrum);
            cqt->output("constantqdc").set(tmp1);
            cqt->output("constantqnf").set(tmp2);

            cqt->compute();

            Spectrogram spectrogram(NumberBins, spectrum[0].size() / DownsampleFactor + 1);
            for (size_t i = 0, cnt = 0; i < spectrum[0].size(); i += DownsampleFactor, ++cnt) {
                vector<float> v(NumberBins);
                for (size_t j = 0; j < NumberBins; ++j) {
                    v[j] = std::abs(spectrum[j][i]);
                }

                spectrogram.col(cnt) = Eigen::Matrix<float, NumberBins, 1>(v.data());
            }

            return amplitude_to_db(spectrogram);
        }
    };

} // hpfw::spectrum