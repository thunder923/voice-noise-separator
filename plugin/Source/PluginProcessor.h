#pragma once
#include <JuceHeader.h>

// Real-time, low-latency cousin of core/separator.py:
// FFT 2048 / hop 512 overlap-add spectral masking with a block-wise
// (approx. 0.7 s) minimum-statistics noise profile — same principle as the
// Python core, simplified for real-time use.
// Output mode: Voice only / Noise only / Blend (voice+noise gains), so the
// "two tools" (voice-only and noise-only) live in this one plug-in.
class VoiceNoiseSeparatorProcessor : public juce::AudioProcessor
{
public:
    VoiceNoiseSeparatorProcessor();
    ~VoiceNoiseSeparatorProcessor() override = default;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "VoiceNoiseSeparator"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}
    void getStateInformation (juce::MemoryBlock&) override;
    void setStateInformation (const void*, int) override;

    juce::AudioProcessorValueTreeState parameters;

private:
    static constexpr int fftOrder = 11;                 // 2048
    static constexpr int fftSize  = 1 << fftOrder;
    static constexpr int hop      = fftSize / 4;        // 512, ~10.7 ms @48k

    juce::dsp::FFT fft { fftOrder };
    std::array<float, fftSize> window {};
    // per-channel streaming state
    struct ChannelState {
        std::vector<float> inFifo, outFifo, fftData, ola;
        std::vector<float> noiseMag;        // running noise profile per bin
        std::vector<float> blockMin;        // current block minima
        int inPos = 0, outPos = 0, blockFrames = 0, outReady = 0;
    };
    std::vector<ChannelState> channels;
    void processChannel (ChannelState& st, float* data, int n);
    void processFrame (ChannelState& st);

    std::atomic<float>* modeParam     = nullptr;  // 0 Voice, 1 Noise, 2 Blend
    std::atomic<float>* strengthParam = nullptr;  // 0..1
    std::atomic<float>* voiceGainParam  = nullptr;
    std::atomic<float>* noiseGainParam  = nullptr;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VoiceNoiseSeparatorProcessor)
};
