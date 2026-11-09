//! Wire-throat picture handling (批5 #14): `build_chat_request` must carry
//! every in-budget picture out, shrink the too-tall ones on the wire copy
//! only, and refuse the WHOLE send — with a size inventory — when a picture
//! crosses a budget line. The old count gate ("only the newest two travel")
//! is gone: nothing is silently dropped anywhere.

use super::*;
use crate::image_ops;

fn png_bytes(w: u32, h: u32) -> Vec<u8> {
    let img = image::RgbaImage::from_pixel(w, h, image::Rgba([200, 100, 50, 255]));
    let mut out = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .expect("encode test png");
    out
}

fn picture(w: u32, h: u32) -> llm_trait::ImageAttachment {
    llm_trait::ImageAttachment::Base64 {
        data: image_ops::to_base64(&png_bytes(w, h)),
        media_type: Some("image/png".to_string()),
        detail: None,
    }
}

/// protects: every in-budget picture reaches the request — three results
/// carrying three pictures plus one carrying a batch of two all travel; the
/// count gate that thinned older results is gone (AC4 "no silent drop" at
/// the wire). fails_when: any picture is trimmed, re-ordered or annotated
/// away to fit a count. no_claim: what the model does with the pictures.
#[test]
fn every_in_budget_picture_reaches_the_request() {
    let messages = vec![
        ChatMessage::user("look at the boards"),
        ChatMessage::tool_with_images("t1", "shot 1", vec![picture(200, 200)]),
        ChatMessage::tool_with_images("t2", "shot 2", vec![picture(200, 200)]),
        ChatMessage::tool_with_images(
            "t3",
            "a batch of two",
            vec![picture(200, 200), picture(200, 200)],
        ),
    ];
    let request = build_chat_request(&messages, &[], None, None, false, None)
        .expect("small pictures fit the budget");
    assert_eq!(request.messages.len(), 4, "no message is discarded");
    let carried: usize = request
        .messages
        .iter()
        .map(|m| match m {
            ChatMessage::Tool { images, .. } | ChatMessage::User { images, .. } => images.len(),
            _ => 0,
        })
        .sum();
    assert_eq!(carried, 4, "all four pictures travel");
}

/// protects: a picture over a budget line refuses the whole send before any
/// request exists, and the error carries the size inventory (AC4). The run
/// fails loudly instead of quietly sending less. fails_when: the refusal is
/// silent, partial, or lacks the sizes. no_claim: that the budget numbers
/// are the right ones (named constants in `image_ops`).
#[test]
fn an_over_budget_request_is_refused_with_an_inventory() {
    let fat = llm_trait::ImageAttachment::Base64 {
        data: "x".repeat(image_ops::MAX_IMAGE_BASE64_BYTES + 1),
        media_type: Some("image/png".to_string()),
        detail: None,
    };
    let messages = vec![ChatMessage::tool_with_images("t1", "a fat one", vec![fat])];
    let err = build_chat_request(&messages, &[], None, None, false, None)
        .expect_err("over the per-image line");
    let text = err.to_string();
    assert!(
        text.contains("request NOT sent"),
        "the refusal is loud: {text}"
    );
    assert!(
        text.contains("OVER per-image budget"),
        "the violator is named: {text}"
    );
}

/// protects: the shrink runs on the wire copy only — the request carries the
/// 1568px version while the caller's history slice keeps the original bytes
/// for audit and resume. fails_when: the pass rewrites the session's history
/// (the original screenshot is lost) or ships the full-size picture.
/// no_claim: the shrink's visual quality.
#[test]
fn the_throat_shrinks_the_wire_copy_and_keeps_the_history() {
    let original = png_bytes(3200, 800);
    let big = llm_trait::ImageAttachment::Base64 {
        data: image_ops::to_base64(&original),
        media_type: Some("image/png".to_string()),
        detail: None,
    };
    let messages = vec![ChatMessage::tool_with_images("t1", "shot", vec![big])];
    let request = build_chat_request(&messages, &[], None, None, false, None)
        .expect("a big picture shrinks to fit");
    let ChatMessage::Tool { images, .. } = &request.messages[0] else {
        panic!("expected a tool result");
    };
    let llm_trait::ImageAttachment::Base64 { data, .. } = &images[0] else {
        panic!("expected a base64 attachment");
    };
    let wire_bytes = image_ops::from_base64(data).expect("wire payload decodes");
    assert_eq!(
        image_ops::header_dimensions(&wire_bytes),
        Some((1568, 392)),
        "the wire copy is shrunk to the long-edge cap"
    );
    let ChatMessage::Tool {
        images: history, ..
    } = &messages[0]
    else {
        panic!("expected a tool result");
    };
    let llm_trait::ImageAttachment::Base64 {
        data: history_data, ..
    } = &history[0]
    else {
        panic!("expected a base64 attachment");
    };
    assert_eq!(
        *history_data,
        image_ops::to_base64(&original),
        "the history keeps the original bytes"
    );
}
