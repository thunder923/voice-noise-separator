#pragma once
#include <JuceHeader.h>
#include "PluginProcessor.h"

class VoiceNoiseSeparatorEditor : public juce::AudioProcessorEditor
{
public:
    explicit VoiceNoiseSeparatorEditor (VoiceNoiseSeparatorProcessor& p)
        : AudioProcessorEditor (&p), proc (p)
    {
        modeBox.addItemList ({ "Voice Only（声だけ）", "Noise Only（ノイズだけ）", "Blend（両方・バランス調整）" }, 1);
        addAndMakeVisible (modeBox);
        addAndMakeVisible (strengthSlider);
        addAndMakeVisible (voiceGainSlider);
        addAndMakeVisible (noiseGainSlider);
        strengthSlider.setSliderStyle (juce::Slider::LinearHorizontal);
        voiceGainSlider.setSliderStyle (juce::Slider::LinearHorizontal);
        noiseGainSlider.setSliderStyle (juce::Slider::LinearHorizontal);
        modeAtt = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment> (p.parameters, "mode", modeBox);
        strAtt  = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (p.parameters, "strength", strengthSlider);
        vgAtt   = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (p.parameters, "voiceGain", voiceGainSlider);
        ngAtt   = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (p.parameters, "noiseGain", noiseGainSlider);
        setSize (440, 260);
    }
    void resized() override
    {
        auto r = getLocalBounds().reduced (16);
        modeBox.setBounds (r.removeFromTop (28)); r.removeFromTop (10);
        strengthSlider.setBounds (r.removeFromTop (28)); r.removeFromTop (6);
        voiceGainSlider.setBounds (r.removeFromTop (28)); r.removeFromTop (6);
        noiseGainSlider.setBounds (r.removeFromTop (28));
    }
private:
    VoiceNoiseSeparatorProcessor& proc;
    juce::ComboBox modeBox;
    juce::Slider strengthSlider, voiceGainSlider, noiseGainSlider;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> modeAtt;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> strAtt, vgAtt, ngAtt;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VoiceNoiseSeparatorEditor)
};
