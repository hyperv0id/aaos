use aaos_session_store::framing::{encode_record, read_record, ReadOutcome};

#[tokio::test]
async fn roundtrip_then_clean_eof() {
    let bytes = encode_record(0x01, br#"{"hash":"ab"}"#);
    let mut src: &[u8] = &bytes;
    match read_record(&mut src).await.unwrap() {
        ReadOutcome::Record(rec) => {
            assert_eq!(rec.tag, 0x01);
            assert_eq!(rec.payload, br#"{"hash":"ab"}"#.to_vec());
        }
        other => panic!("expected record, got {other:?}"),
    }
    assert!(matches!(
        read_record(&mut src).await.unwrap(),
        ReadOutcome::Eof
    ));
}

#[tokio::test]
async fn truncated_payload_is_torn() {
    let mut bytes = encode_record(0x01, b"0123456789");
    bytes.truncate(bytes.len() - 1);
    let mut src: &[u8] = &bytes;
    assert!(matches!(
        read_record(&mut src).await.unwrap(),
        ReadOutcome::Torn
    ));
}

#[tokio::test]
async fn stray_bytes_are_torn() {
    let mut src: &[u8] = &[0x00, 0x00];
    assert!(matches!(
        read_record(&mut src).await.unwrap(),
        ReadOutcome::Torn
    ));
}

#[tokio::test]
async fn missing_tag_is_torn() {
    let mut bytes = encode_record(0x00, b"payload");
    bytes.pop(); // drop last payload byte, then also lose the tag below
    let mut src: &[u8] = &bytes[..bytes.len() - 1];
    assert!(matches!(
        read_record(&mut src).await.unwrap(),
        ReadOutcome::Torn
    ));
}

#[tokio::test]
async fn over_cap_length_is_torn() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&((1u32 << 26) + 1).to_be_bytes());
    bytes.push(0x01);
    let mut src: &[u8] = &bytes;
    assert!(matches!(
        read_record(&mut src).await.unwrap(),
        ReadOutcome::Torn
    ));
}

#[tokio::test]
async fn empty_payload_roundtrips() {
    let bytes = encode_record(0x03, b"");
    let mut src: &[u8] = &bytes;
    match read_record(&mut src).await.unwrap() {
        ReadOutcome::Record(rec) => {
            assert_eq!(rec.tag, 0x03);
            assert!(rec.payload.is_empty());
        }
        other => panic!("expected record, got {other:?}"),
    }
}
