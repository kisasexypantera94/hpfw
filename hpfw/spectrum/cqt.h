#pragma once

#include <Eigen/Dense>
#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>

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
            size_t NFFT = 16384, // TODO: play around with parameters
            size_t HopLength = 2880,
            size_t BinsPerOctave = 24,
            size_t NumberBins = 121>
    class CQT {
    public:
        using Spectrogram = Eigen::Matrix<float, 121, Eigen::Dynamic>;

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
            using uptr = std::unique_ptr<Algorithm>;

            AlgorithmFactory &factory = standard::AlgorithmFactory::instance();

            auto audio = uptr(factory.create("MonoLoader",
                                             "filename", filename,
                                             "sampleRate", int(SampleRate)));

            auto fc = uptr(factory.create("FrameCutter",
                                          "frameSize", int(NFFT),
                                          "hopSize", int(HopLength)));

            auto w = uptr(factory.create("Windowing",
                                         "size", int(NFFT),
                                         "type", "hann"));

            auto cqt = uptr(factory.create("SpectrumCQ",
                                           "binsPerOctave", int(BinsPerOctave),
                                           "minFrequency", 130.81,
                                           "numberBins", int(NumberBins)));

            // MonoLoader -> FrameCutter -> Windowing -> SpectrumCQ
            vector<float> audioBuffer;
            audio->output("audio").set(audioBuffer);

            vector<float> frame, windowedFrame;
            fc->input("signal").set(audioBuffer);
            fc->output("frame").set(frame);

            w->input("frame").set(frame);
            w->output("frame").set(windowedFrame);

            vector<float> spectrum;
            cqt->input("frame").set(windowedFrame);
            cqt->output("spectrumCQ").set(spectrum);

            audio->compute();

            Spectrogram spectrogram(121, 0);
            while (true) {

                // compute a frame
                fc->compute();

                // if it was the last one (ie: it was empty), then we're done.
                if (frame.empty()) {
                    break;
                }

                // if the frame is silent, just drop it and go on processing
                if (isSilent(frame)) {
                    continue;
                }

                w->compute();
                cqt->compute();

                spectrogram.conservativeResize(spectrogram.rows(), spectrogram.cols() + 1);
                spectrogram.col(spectrogram.cols() - 1) = Eigen::Matrix<float, 121, 1>::Map(spectrum.data());
            }

            return spectrogram;
        }
    };

} // hpfw::spectrum