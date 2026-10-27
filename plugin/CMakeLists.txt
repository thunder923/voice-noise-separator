# VoiceNoiseSeparator — JUCE plugin (VST3 / AU / AAX)
# Build:
#   git clone https://github.com/juce-framework/JUCE (or add as submodule)
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --config Release
#
# Formats: VST3 and AU build freely.
# AAX: requires the AAX SDK from Avid (developer.avid.com/aax) placed where
# JUCE expects it, and — for loading in a *retail* Pro Tools — PACE Eden
# signing with a physical iLok dongle (Avid developer approval required).
# Unsigned AAX loads only in the free "Pro Tools Developer" build.
cmake_minimum_required(VERSION 3.22)
project(VoiceNoiseSeparator VERSION 0.1.0)

add_subdirectory(JUCE)  # path to JUCE

juce_add_plugin(VoiceNoiseSeparator
    COMPANY_NAME "Kanta"
    PLUGIN_MANUFACTURER_CODE Knta
    PLUGIN_CODE VnSp
    FORMATS VST3 AU AAX Standalone
    PRODUCT_NAME "VoiceNoiseSeparator"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD TRUE
)

target_sources(VoiceNoiseSeparator PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
)

target_compile_definitions(VoiceNoiseSeparator PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
)

target_link_libraries(VoiceNoiseSeparator PRIVATE
    juce::juce_audio_utils
    juce::juce_dsp
    juce::juce_recommended_config_flags
)
