package com.ouor.rvcandroid.audio

import android.content.Context
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log

private const val TAG = "Rvc.AudioMeta"

data class AudioMeta(
    val durationMs: Long,
    val sampleRate: Int,
    val channels: Int,
)

/**
 * Cheap pre-decode probe of an audio Uri. Used by the input picker to
 * decide whether the file fits the 60-second conversion window before we
 * commit to a full ffmpeg decode.
 *
 * Tries [MediaExtractor] first (gives us channel count from MediaFormat),
 * then falls back to [MediaMetadataRetriever] if the extractor doesn't
 * recognise the container — useful for FLAC/OGG which the platform
 * sometimes refuses through extractor but accepts through retriever.
 */
object AudioMetaProbe {
    fun probe(ctx: Context, uri: Uri): AudioMeta? =
        probeViaExtractor(ctx, uri) ?: probeViaRetriever(ctx, uri)

    private fun probeViaExtractor(ctx: Context, uri: Uri): AudioMeta? {
        val ex = MediaExtractor()
        return try {
            ex.setDataSource(ctx, uri, null)
            for (i in 0 until ex.trackCount) {
                val format = ex.getTrackFormat(i)
                val mime = format.getString(MediaFormat.KEY_MIME) ?: continue
                if (!mime.startsWith("audio/")) continue
                val sr = format.getIntOrNull(MediaFormat.KEY_SAMPLE_RATE) ?: continue
                val ch = format.getIntOrNull(MediaFormat.KEY_CHANNEL_COUNT) ?: 1
                val durUs = format.getLongOrNull(MediaFormat.KEY_DURATION) ?: continue
                return AudioMeta(durationMs = durUs / 1000, sampleRate = sr, channels = ch)
            }
            null
        } catch (t: Throwable) {
            Log.w(TAG, "extractor probe failed for $uri: ${t.message}")
            null
        } finally {
            runCatching { ex.release() }
        }
    }

    private fun probeViaRetriever(ctx: Context, uri: Uri): AudioMeta? {
        val mmr = MediaMetadataRetriever()
        return try {
            mmr.setDataSource(ctx, uri)
            val durationMs = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull() ?: return null
            val sampleRate = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_SAMPLERATE)
                ?.toIntOrNull() ?: return null
            // Retriever has no channel-count key, so default to mono — it's
            // only used for the metadata badge and we mix to mono anyway.
            AudioMeta(durationMs, sampleRate, channels = 1)
        } catch (t: Throwable) {
            Log.w(TAG, "retriever probe failed for $uri: ${t.message}")
            null
        } finally {
            runCatching { mmr.release() }
        }
    }

    private fun MediaFormat.getIntOrNull(key: String): Int? =
        if (containsKey(key)) getInteger(key) else null

    private fun MediaFormat.getLongOrNull(key: String): Long? =
        if (containsKey(key)) getLong(key) else null
}
