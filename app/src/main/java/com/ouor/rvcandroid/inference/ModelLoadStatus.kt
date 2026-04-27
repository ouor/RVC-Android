package com.ouor.rvcandroid.inference

/**
 * Per-slot async load state for the three model files (synthesizer, HuBERT,
 * RMVPE). The synthesizer's [Loaded.summary] carries the parsed metadata so
 * the UI can both show the model's traits and decide whether RMVPE is
 * mandatory before the user even tries to convert.
 */
sealed class ModelLoadStatus {
    object Empty : ModelLoadStatus()
    object Loading : ModelLoadStatus()
    data class Loaded(val summary: ModelSummary? = null) : ModelLoadStatus()
    data class Failed(val error: String) : ModelLoadStatus()
}

data class ModelSummary(
    val sampleRate: Int,
    val f0: Boolean,
    val embedder: String,
    val embChannels: Int,
)
