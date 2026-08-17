//! Cutting UTF-8 text without splitting a character.
//!
//! Every excerpt, label and preview in zkb is bounded in **bytes** — a database
//! column, a terminal line, a report row. On an ASCII corpus that is the same as
//! bounding characters; on this one it is not, and a cut landing inside a
//! three-byte sequence produces bytes that are not text.
//!
//! This lives in one place because it was written four times and got it wrong
//! twice: `chunks.heading_path` was storing invalid UTF-8 for any record whose
//! label ran past 64 bytes (about 21 Chinese characters), and that column is
//! indexed by FTS. The failure is quiet — SQLite stores whatever bytes it is
//! given — so nothing reports it until something downstream tries to decode.

const std = @import("std");

/// The largest cut at or below `max_bytes` that lands on a character boundary.
///
/// Continuation bytes match `0b10xxxxxx`; walking back off them reaches the
/// start of the sequence. Returns `text.len` when no cut is needed, so callers
/// can compare against it to decide whether to add an ellipsis.
pub fn truncate(text: []const u8, max_bytes: usize) usize {
    if (text.len <= max_bytes) return text.len;
    var end = max_bytes;
    // `text[end]` is the first byte *dropped*: if it continues a sequence, the
    // sequence started earlier and the whole thing has to go.
    while (end > 0 and isContinuation(text[end])) end -= 1;
    return end;
}

/// `text[0..truncate(text, max_bytes)]`, for the common case.
pub fn cut(text: []const u8, max_bytes: usize) []const u8 {
    return text[0..truncate(text, max_bytes)];
}

pub fn isContinuation(byte: u8) bool {
    return (byte & 0xC0) == 0x80;
}

/// Whether `text` is well-formed UTF-8. Used by tests to assert that what zkb
/// stores can be read back as text.
pub fn isValid(text: []const u8) bool {
    return std.unicode.utf8ValidateSlice(text);
}
