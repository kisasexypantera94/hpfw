#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>

#include "convert.h"

namespace hpfw::spectrum {

    /// CQT is a wrapper class around essentia extractor.
    ///
    /// \tparam SampleRate
    /// \tparam NFFT
    /// \tparam HopLength
    /// \tparam BinsPerOctave
    /// \tparam NumberBins
    /// \tparam MinFrequency
    template<size_t SampleRate = 44100,
            size_t HopLength = 96,
            size_t BinsPerOctave = 24,
            size_t NumberBins = 121,
            size_t DownsampleFactor = 3>
    class CQT {
    public:
        using Spectrogram = Eigen::Matrix<double, NumberBins, Eigen::Dynamic>;

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
                                           "minFrequency", 130.81,
                                           "maxFrequency", 4186.01));

            vector<vector<complex<float>>> spectrum;
            vector<complex<float>> tmp1;
            vector<complex<float>> tmp2;
            cqt->input("frame").set(audioBuffer);
            cqt->output("constantq").set(spectrum);
            cqt->output("constantqdc").set(tmp1);
            cqt->output("constantqnf").set(tmp2);

            cqt->compute();

            Spectrogram spectrogram(NumberBins, 0);
            for (size_t i = 0; i < spectrum[0].size(); i += DownsampleFactor) {
                vector<double> v(NumberBins);
                for (size_t j = 0; j < NumberBins; ++j) {
                    v[j] = std::abs(spectrum[j][i]);
                }

                spectrogram.conservativeResize(spectrogram.rows(), spectrogram.cols() + 1);
                spectrogram.col(spectrogram.cols() - 1) = Eigen::Matrix<double, NumberBins, 1>(v.data());
            }

            return amplitude_to_db(spectrogram);
        }
    };

} // hpfw::spectrum