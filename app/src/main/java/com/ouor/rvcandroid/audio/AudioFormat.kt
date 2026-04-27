package com.ouor.rvcandroid.audio

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns

enum class AudioFormat(val mime: String, val ext: String, val displayName: String) {
    WAV("audio/wav", "wav", "WAV"),
    MP3("audio/mpeg", "mp3", "MP3"),
    AAC("audio/aac", "aac", "AAC"),
    M4A("audio/mp4", "m4a", "M4A"),
    FLAC("audio/flac", "flac", "FLAC"),
    OGG("audio/ogg", "ogg", "OGG");

    companion object {
        // ContentResolver MIME varies by source app: some emit audio/x-wav,
        // audio/mp3, audio/x-m4a, etc. Match the common alternates so we
        // don't miss-route a perfectly valid file.
        private val MIME_ALIASES = mapOf(
            "audio/x-wav" to WAV,
            "audio/wave" to WAV,
            "audio/mp3" to MP3,
            "audio/x-mp3" to MP3,
            "audio/mp4a-latm" to M4A,
            "audio/x-m4a" to M4A,
            "audio/x-flac" to FLAC,
            "audio/x-vorbis+ogg" to OGG,
            "audio/vorbis" to OGG,
        )

        fun fromMime(mime: String?): AudioFormat? {
            if (mime == null) return null
            entries.firstOrNull { it.mime.equals(mime, ignoreCase = true) }?.let { return it }
            return MIME_ALIASES[mime.lowercase()]
        }

        fun fromExtension(ext: String?): AudioFormat? {
            val e = ext?.lowercase() ?: return null
            return entries.firstOrNull { it.ext == e }
        }

        fun fromFilename(name: String?): AudioFormat? =
            fromExtension(name?.substringAfterLast('.', ""))

        fun detect(ctx: Context, uri: Uri): AudioFormat? {
            fromMime(ctx.contentResolver.getType(uri))?.let { return it }
            val name = displayNameOf(ctx, uri)
            return fromFilename(name)
        }

        private fun displayNameOf(ctx: Context, uri: Uri): String? {
            ctx.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
                ?.use { c -> if (c.moveToFirst()) return c.getString(0) }
            return uri.lastPathSegment
        }
    }
}
