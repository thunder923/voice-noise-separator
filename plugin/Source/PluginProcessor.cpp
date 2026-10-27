#include "PluginProcessor.h"
#include "PluginEditor.h"

static juce::AudioProcessorValueTreeState::ParameterLayout makeLayout()
{
    using P = juce::AudioParameterFloat;
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> p;
    p.push_back (std::make_unique<juce::AudioParameterChoice> (
        "mode", "Mode", juce::StringArray { "Voice Only", "Noise Only", "Blend" }, 0));
    p.push_back (std::make_unique<P> ("strength", "Strength",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.01f), 0.8f));
    p.push_back (std::make_unique<P> ("voiceGain", "Voice Gain",
        juce::NormalisableRange<float> (0.0f, 2.0f, 0.01f), 1.0f));
    p.push_back (std::make_unique<P> ("noiseGain", "Noise Gain",
        juce::NormalisableRange<float> (0.0f, 2.0f, 0.01f), 1.0f));
    return { p.begin(), p.end() };
}

VoiceNoiseSeparatorProcessor::VoiceNoiseSeparatorProcessor()
    : AudioProcessor (BusesProperties()
        .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
        .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      parameters (*this, nullptr, "PARAMS", makeLayout())
{
    modeParam      = parameters.getRawParameterValue ("mode");
    strengthParam  = parameters.getRawParameterValue ("strength");
    voiceGainParam = parameters.getRawParameterValue ("voiceGain");
    noiseGainParam = parameters.getRawParameterValue ("noiseGain");
    for (int i = 0; i < fftSize; ++i)  // Hann
        window[(size_t) i] = 0.5f * (1.0f - std::cos (2.0f * juce::MathConstants<float>::pi
                                                     * (float) i / (float) (fftSize - 1)));
}

void VoiceNoiseSeparatorProcessor::prepareToPlay (double, int)
{
    channels.assign (2, ChannelState{});
    for (auto& st : channels)
    {
        st.inFifo.assign (fftSize, 0.0f);
        st.outFifo.assign (fftSize, 0.0f);
        st.fftData.assign (fftSize * 2, 0.0f);
        st.ola.assign (fftSize, 0.0f);
        st.noiseMag.assign (fftSize / 2 + 1, 1.0e-6f);
        st.blockMin.assign (fftSize / 2 + 1, 1.0e9f);
        st.inPos = st.outPos = st.blockFrames = st.outReady = 0;
    }
}

bool VoiceNoiseSeparatorProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    return layouts.getMainOutputChannelSet() == layouts.getMainInputChannelSet()
        && ! layouts.getMainOutputChannelSet().isDisabled();
}

// One FFT frame: estimate mask, split into voice / noise spectra, IFFT both is
// avoided — noise spectrum = input * (1-gain), so a second IFFT is only needed
// in Noise/Blend modes. Voice mode uses one IFFT (gain), Noise mode one
// IFFT (1-gain), Blend sums two IFFTs scaled by the gains.
void VoiceNoiseSeparatorProcessor::processFrame (ChannelState& st)
{
    std::copy (st.inFifo.begin(), st.inFifo.end(), st.fftData.begin());
    for (int i = 0; i < fftSize; ++i) st.fftData[(size_t) i] *= window[(size_t) i];
    fft.performRealOnlyForwardTransform (st.fftData.data());

    const float strength = strengthParam->load();
    const int mode = (int) modeParam->load();
    const int bins = fftSize / 2 + 1;

    // block minimum-statistics update (block ~= 64 frames)
    for (int b = 0; b < bins; ++b)
    {
        float re = st.fftData[(size_t) b * 2], im = st.fftData[(size_t) b * 2 + 1];
        float mag = std::sqrt (re * re + im * im);
        st.blockMin[(size_t) b] = std::min (st.blockMin[(size_t) b], mag);
    }
    if (++st.blockFrames >= 64)
    {
        st.blockFrames = 0;
        for (int b = 0; b < bins; ++b)
        {
            // Rayleigh-corrected block percentile approximation via minima x1.9,
            // smoothed into the running profile
            float est = st.blockMin[(size_t) b] * 1.9f;
            st.noiseMag[(size_t) b] = 0.5f * st.noiseMag[(size_t) b] + 0.5f * est;
            st.blockMin[(size_t) b] = 1.0e9f;
        }
    }

    // build gain curve
    std::vector<float> gain ((size_t) bins);
    const float floor_ = 1.0f - 0.97f * strength;
    for (int b = 0; b < bins; ++b)
    {
        float re = st.fftData[(size_t) b * 2], im = st.fftData[(size_t) b * 2 + 1];
        float mag = std::sqrt (re * re + im * im);
        float snr = std::max (mag / std::max (st.noiseMag[(size_t) b], 1.0e-10f) - 1.0f, 0.0f);
        float mask = snr / (snr + 1.0f);
        gain[(size_t) b] = floor_ + (1.0f - floor_) * std::pow (mask, 0.5f + 1.5f * strength);
    }
    auto applyGain = [&] (float voiceW, float noiseW, std::vector<float>& outSpec)
    {
        outSpec = st.fftData; // placeholder, replaced below
        juce::ignoreUnused (voiceW, noiseW);
    };
    juce::ignoreUnused (applyGain);

    // Compose output spectrum according to mode, in-place on a copy
    std::vector<float> spec = st.fftData;
    const float vg = voiceGainParam->load(), ng = noiseGainParam->load();
    for (int b = 0; b < bins; ++b)
    {
        float g = gain[(size_t) b];
        float w = (mode == 0) ? g
                : (mode == 1) ? (1.0f - g)
                              : vg * g + ng * (1.0f - g);
        spec[(size_t) b * 2]     *= w;
        spec[(size_t) b * 2 + 1] *= w;
        if (b > 0 && b < fftSize / 2)  // mirror for negative frequencies
        {
            int mb = fftSize - b;
            spec[(size_t) mb * 2]     = spec[(size_t) b * 2];
            spec[(size_t) mb * 2 + 1] = -spec[(size_t) b * 2 + 1];
        }
    }
    fft.performRealOnlyInverseTransform (spec.data());
    for (int i = 0; i < fftSize; ++i)
        st.ola[(size_t) i] += spec[(size_t) i] * window[(size_t) i] / (float) (fftSize / 2);
    std::copy_n (st.ola.begin(), hop, st.outFifo.begin() + st.outReady);
    st.outReady += hop;
    std::move (st.ola.begin() + hop, st.ola.end(), st.ola.begin());
    std::fill (st.ola.end() - hop, st.ola.end(), 0.0f);
}

void VoiceNoiseSeparatorProcessor::processChannel (ChannelState& st, float* data, int n)
{
    for (int i = 0; i < n; ++i)
    {
        st.inFifo[(size_t) st.inPos] = data[i];
        data[i] = st.outReady > 0 ? st.outFifo[(size_t) st.outPos] : 0.0f;
        if (st.outReady > 0) { st.outPos = (st.outPos + 1) % fftSize; --st.outReady; }
        if (++st.inPos == fftSize)
        {
            st.inPos = 0;
            processFrame (st);
            // shift input fifo by hop for next frame (overlap)
            std::move (st.inFifo.begin() + hop, st.inFifo.end(), st.inFifo.begin());
            st.inPos = fftSize - hop;
        }
    }
}

void VoiceNoiseSeparatorProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;
    int numCh = std::min (buffer.getNumChannels(), 2);
    for (int ch = 0; ch < numCh; ++ch)
        processChannel (channels[(size_t) ch], buffer.getWritePointer (ch), buffer.getNumSamples());
}

juce::AudioProcessorEditor* VoiceNoiseSeparatorProcessor::createEditor()
{
    return new VoiceNoiseSeparatorEditor (*this);
}

void VoiceNoiseSeparatorProcessor::getStateInformation (juce::MemoryBlock& dest)
{
    if (auto xml = parameters.copyState().createXml()) copyXmlToBinary (*xml, dest);
}

void VoiceNoiseSeparatorProcessor::setStateInformation (const void* data, int size)
{
    if (auto xml = getXmlFromBinary (data, size))
        if (xml->hasTagName (parameters.state.getType()))
            parameters.replaceState (juce::ValueTree::fromXml (*xml));
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new VoiceNoiseSeparatorProcessor();
}
