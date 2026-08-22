//! Length + tag record framing for branch logs.
//!
//! Record = `u32 BE length (payload)` + `u8 tag` + payload. No CRC — an
//! internal format: a torn tail is anything that fails to decode as a full
//! record (short length prefix, short tag, short payload, or a length over
//! the cap), and recovery truncates back to the last good record.

use std::io;

use tokio::io::{AsyncRead, AsyncReadExt};

pub const TAG_HEADER: u8 = 0x00;
pub const TAG_SEGMENT_REF: u8 = 0x01;
pub const TAG_COMPACT_MAP: u8 = 0x02;
pub const TAG_SIDE_EFFECT: u8 = 0x03;

/// Payload length cap; beyond this a length prefix reads as torn rather than
/// provoking a giant allocation.
const MAX_PAYLOAD: usize = 1 << 26;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedRecord {
    pub tag: u8,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadOutcome {
    /// Clean end of stream exactly between records.
    Eof,
    /// Truncated or corrupt tail; the stream ends at the last good record.
    Torn,
    Record(DecodedRecord),
}

pub fn encode_record(tag: u8, payload: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(payload.len() + 5);
    buf.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    buf.push(tag);
    buf.extend_from_slice(payload);
    buf
}

pub async fn read_record<R: AsyncRead + Unpin>(src: &mut R) -> io::Result<ReadOutcome> {
    let mut len_buf = [0u8; 4];
    let mut filled = 0usize;
    while filled < 4 {
        match src.read(&mut len_buf[filled..]).await? {
            0 => break,
            n => filled += n,
        }
    }
    if filled == 0 {
        return Ok(ReadOutcome::Eof);
    }
    if filled < 4 {
        return Ok(ReadOutcome::Torn);
    }
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > MAX_PAYLOAD {
        return Ok(ReadOutcome::Torn);
    }
    let mut tag_buf = [0u8; 1];
    if src.read_exact(&mut tag_buf).await.is_err() {
        return Ok(ReadOutcome::Torn);
    }
    let mut payload = vec![0u8; len];
    if src.read_exact(&mut payload).await.is_err() {
        return Ok(ReadOutcome::Torn);
    }
    Ok(ReadOutcome::Record(DecodedRecord {
        tag: tag_buf[0],
        payload,
    }))
}
