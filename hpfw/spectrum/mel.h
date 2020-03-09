#pragma once

#include <Eigen/Dense>
#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>

namespace hpfw::spectrum {

    /// MelSpectrogram is a wrapper class around essentia extractor.
    ///
    /// \tparam SampleRate
    /// \tparam MelBins
    /// \tparam NFFT
    /// \tparam HopLength
    template<size_t SampleRate = 44100,
            size_t MelBins = 33,
            size_t NFFT = SampleRate * 100 / 1000,
            size_t HopLength = SampleRate * 10 / 1000>
    class MelSpectrogram {
    public:
        using Spectrogram = Eigen::Matrix<float, MelBins, Eigen::Dynamic>;

        MelSpectrogram() {
            essentia::init();
        }

        ~MelSpectrogram() {
            // TODO: unsafe when there is more than one object
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

            auto spec = uptr(factory.create("Spectrum",
                                            "size", int(NFFT)));

            auto melbands = uptr(factory.create("MelBands",
                                                "inputSize", int(NFFT) / 2 + 1,
                                                "numberBands", int(MelBins)));

            // MonoLoader -> FrameCutter -> Windowing -> Spectrum -> MelBands
            vector<float> audioBuffer;
            audio->output("audio").set(audioBuffer);

            vector<float> frame, windowedFrame;
            fc->input("signal").set(audioBuffer);
            fc->output("frame").set(frame);

            w->input("frame").set(frame);
            w->output("frame").set(windowedFrame);

            vector<float> spectrum, bands;
            spec->input("frame").set(windowedFrame);
            spec->output("spectrum").set(spectrum);

            melbands->input("spectrum").set(spectrum);
            melbands->output("bands").set(bands);


            audio->compute();

            Spectrogram spectrogram(MelBins, 0);
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
                spec->compute();
                melbands->compute();

                spectrogram.conservativeResize(spectrogram.rows(), spectrogram.cols() + 1);
                spectrogram.col(spectrogram.cols() - 1) = Eigen::Matrix<Real, MelBins, 1>::Map(bands.data());
            }

            double maxVal = std::max(1e-10, double(spectrogram.maxCoeff()));
            spectrogram = 10.0 * (spectrogram.array() < 1e-10).select(1e-10, spectrogram).array().log10()
                          - 10.0 * log10(maxVal);

            maxVal = spectrogram.maxCoeff();
            spectrogram = (spectrogram.array() < (maxVal - 80.0)).select(maxVal - 80.0, spectrogram);

            return spectrogram;
        }
    };

} // hpfw::spectrum