//! Image mechanics shared by tools and the wire layer: magic sniffing,
//! header-only dimension reads, the downscale-to-fit pass, and the
//! per-image / per-message byte budget that keeps request bodies under the
//! gateway's limit (the HTTP 413 cure — see [`apply_image_budget`]).
//!
//! One source of truth for "is this bytes stream a picture, how big is it,
//! and does it fit the wire": the `read_image` tool sniffs and labels with
//! it, and the engine budgets and shrinks with it. Numbers live here as
//! named constants so a product can argue with them in one place.

use base64::Engine as _;
use llm_trait::message::{ChatMessage, ImageAttachment};

/// Longest edge any picture may carry on the wire: large enough for a
/// desktop screenshot's text to stay legible to a vision model, small
/// enough that two of them never tripped the gateway's body limit (the
/// HTTP 413 of session 20261030_862ddf5c).
pub const MAX_WIRE_LONG_EDGE: u32 = 1568;

/// Per-image wire budget, in base64 characters (the on-the-wire size).
/// ~2.6 MB of raw pixels — after the downscale pass only the most
/// pathological screenshot can reach it, and such a one is refused with an
/// inventory instead of silently thinned.
pub const MAX_IMAGE_BASE64_BYTES: usize = 3_500_000;

/// Per-message wire budget for all of its pictures together, in base64
/// characters. A message keeps all of its pictures or none — this is the
/// line where "all" stops being affordable and the send is refused.
pub const MAX_MESSAGE_IMAGES_BASE64_BYTES: usize = 12_000_000;

/// MIME sniff from magic bytes — the extension and any declared MIME are
/// advisory at best (a `.jpg` that is really a PNG is a PNG; a `text/html`
/// wearing `image/png` is not a picture at all).
pub fn sniff_mime(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Some("image/png");
    }
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some("image/jpeg");
    }
    None
}

/// Decode base64 with the standard alphabet.
pub fn from_base64(data: &str) -> Result<Vec<u8>, String> {
    base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|e| format!("invalid base64 payload: {e}"))
}

/// Encode to base64 (standard alphabet, padded).
pub fn to_base64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// On-the-wire size of a raw payload once base64-encoded, without doing the
/// encode: padded standard base64 is `ceil(n / 3) * 4` characters.
pub fn base64_len(raw_len: usize) -> usize {
    raw_len.div_ceil(3) * 4
}

/// How much base64 head to decode for a dimension probe: enough for any
/// PNG and for all but the most EXIF-bloated JPEG (its SOF marker sits near
/// the head of the stream, well inside this window).
const DIMS_PROBE_B64_CHARS: usize = 64 * 1024;

/// Read a picture's pixel size from the HEAD of its base64 payload — a
/// bounded probe, never a full decode. `None` when the head carries no
/// decodable PNG/JPEG header: an exotic format, a corrupt payload, or not
/// an image at all. Non-ASCII characters are skipped (a payload may be
/// arbitrary text), so the probe never slices mid-character.
pub fn base64_prefix_dims(data: &str) -> Option<Dims> {
    let head: String = data
        .chars()
        .filter(char::is_ascii)
        .take(DIMS_PROBE_B64_CHARS)
        .collect();
    let usable = head.len() - (head.len() % 4);
    if usable == 0 {
        return None;
    }
    let raw = from_base64(&head[..usable]).ok()?;
    header_dimensions(&raw)
}

/// A picture's pixel size, (width, height).
pub type Dims = (u32, u32);

/// Header-only dimension read for PNG/JPEG — no pixel decode. `None` for
/// bytes that are not a decodable PNG/JPEG stream.
pub fn header_dimensions(bytes: &[u8]) -> Option<Dims> {
    let format = match sniff_mime(bytes)? {
        "image/png" => image::ImageFormat::Png,
        "image/jpeg" => image::ImageFormat::Jpeg,
        _ => return None,
    };
    let reader = image::ImageReader::with_format(std::io::Cursor::new(bytes), format);
    reader.into_dimensions().ok()
}

/// Shrink a picture so its longest edge is [`MAX_WIRE_LONG_EDGE`],
/// re-encoding in the SAME format (a PNG screenshot stays lossless, a JPEG
/// stays a JPEG). Returns the (possibly unchanged) bytes plus the original
/// and final pixel sizes.
///
/// A picture that already fits is returned byte-identical: no re-encode, no
/// gratuitous generation loss, and a stable payload across turns. Errors are
/// honest strings — the caller decides whether an un-shrinkable picture
/// still travels (it does, subject to the byte budget).
pub fn downscale_to_fit(bytes: &[u8]) -> Result<(Vec<u8>, Dims, Dims), String> {
    let format = match sniff_mime(bytes) {
        Some("image/png") => image::ImageFormat::Png,
        Some("image/jpeg") => image::ImageFormat::Jpeg,
        _ => return Err("not a PNG/JPEG image (magic bytes do not match)".to_string()),
    };
    let mut reader = image::ImageReader::with_format(std::io::Cursor::new(bytes), format);
    // Untrusted bytes get a hard allocation ceiling before any decode —
    // a "small" file can declare a monstrous canvas.
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(32_000);
    limits.max_image_height = Some(32_000);
    limits.max_alloc = Some(512 * 1024 * 1024);
    reader.limits(limits);
    let img = reader
        .decode()
        .map_err(|e| format!("image decode failed: {e}"))?;
    let original = (img.width(), img.height());
    let long_edge = original.0.max(original.1);
    if long_edge <= MAX_WIRE_LONG_EDGE {
        return Ok((bytes.to_vec(), original, original));
    }
    let scale = MAX_WIRE_LONG_EDGE as f32 / long_edge as f32;
    let new_w = ((original.0 as f32 * scale).round() as u32).max(1);
    let new_h = ((original.1 as f32 * scale).round() as u32).max(1);
    let resized = img.resize(new_w, new_h, image::imageops::FilterType::Lanczos3);
    let mut out = Vec::new();
    match format {
        image::ImageFormat::Png => resized
            .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
            .map_err(|e| format!("PNG re-encode failed: {e}"))?,
        _ => {
            let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 85);
            encoder
                .encode_image(&resized)
                .map_err(|e| format!("JPEG re-encode failed: {e}"))?;
        }
    }
    Ok((out, original, (new_w, new_h)))
}

/// Human byte count for inventories and guidance ("2.1 MB", "350 KB").
fn pretty_bytes(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1} MB", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.0} KB", n as f64 / 1_000.0)
    } else {
        format!("{n} B")
    }
}

/// Why the request never left the machine: the full size inventory of every
/// picture it carried, plus the lines that crossed the budget. Surfaced as
/// a run error — the request is refused WHOLE, never thinned.
#[derive(Debug, Clone)]
pub struct ImageBudgetError {
    pub inventory: Vec<String>,
}

impl std::fmt::Display for ImageBudgetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "request NOT sent - image budget exceeded (nothing is silently dropped):"
        )?;
        for line in &self.inventory {
            writeln!(f, "  {line}")?;
        }
        write!(
            f,
            "guidance: downscale the screenshot before pasting (long edge <= \
             {MAX_WIRE_LONG_EDGE}px) or split the pictures across fewer messages."
        )
    }
}

impl std::error::Error for ImageBudgetError {}

/// Prepare every picture for the wire and enforce the byte budget.
///
/// Per picture: shrink to [`MAX_WIRE_LONG_EDGE`] (same format), then measure
/// the base64 it now occupies. Per message: the sum of its pictures. Over
/// either line the WHOLE send is refused with [`ImageBudgetError`] — this is
/// the one place a picture may leave the wire, and it does so with an
/// inventory, never a silent drop. Word-only messages pass untouched.
pub fn apply_image_budget(messages: &mut [ChatMessage]) -> Result<(), ImageBudgetError> {
    let mut inventory: Vec<String> = Vec::new();
    let mut over_budget = false;

    for (mi, message) in messages.iter_mut().enumerate() {
        let images = match message {
            ChatMessage::Tool { images, .. } | ChatMessage::User { images, .. } => images,
            _ => continue,
        };
        if images.is_empty() {
            continue;
        }
        let mut message_bytes = 0usize;
        for (ii, image) in images.iter_mut().enumerate() {
            let ImageAttachment::Base64 {
                data, media_type, ..
            } = image
            else {
                // A URL reference carries no payload of ours to budget or
                // shrink — the provider fetches it. It still gets an
                // inventory line so the sizes listed are complete.
                inventory.push(format!(
                    "message {}, image {}: URL reference",
                    mi + 1,
                    ii + 1
                ));
                continue;
            };
            // Shrink first, judge second: a picture must not be refused for
            // size it would have shed two lines down. Unparseable bytes are
            // left as they are and judged by their measured size.
            if let Ok(raw) = from_base64(data)
                && sniff_mime(&raw).is_some()
                && let Ok((shrunk, original, final_dims)) = downscale_to_fit(&raw)
            {
                *data = to_base64(&shrunk);
                if original != final_dims {
                    inventory.push(format!(
                        "message {}, image {}: downscaled {}x{} -> {}x{}",
                        mi + 1,
                        ii + 1,
                        original.0,
                        original.1,
                        final_dims.0,
                        final_dims.1
                    ));
                }
            }
            let size = data.len();
            let flag = if size > MAX_IMAGE_BASE64_BYTES {
                over_budget = true;
                " [OVER per-image budget]"
            } else {
                ""
            };
            inventory.push(format!(
                "message {}, image {}: {} base64{}",
                mi + 1,
                ii + 1,
                pretty_bytes(size),
                flag
            ));
            if let Some(mime) = media_type {
                inventory.push(format!("    type: {mime}"));
            }
            message_bytes += size;
        }
        if message_bytes > MAX_MESSAGE_IMAGES_BASE64_BYTES {
            over_budget = true;
            inventory.push(format!(
                "message {}: {} of pictures total [OVER per-message budget of {}]",
                mi + 1,
                pretty_bytes(message_bytes),
                pretty_bytes(MAX_MESSAGE_IMAGES_BASE64_BYTES)
            ));
        }
    }

    if over_budget {
        Err(ImageBudgetError { inventory })
    } else {
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod test_fixtures {
    /// A screenshot-sized PNG of incompressible noise — the payload SCALE of
    /// a real desktop capture (megabytes of base64 at 1278x1152), which
    /// solid-color fixtures cannot reproduce (they compress to nothing).
    /// Grayscale keeps it inside the wire byte budget while still dwarfing
    /// any token limit. Deterministic LCG pixels: same bytes on every run.
    pub(crate) fn noisy_png(w: u32, h: u32) -> Vec<u8> {
        let state = std::cell::Cell::new(0x9E37_79B9_7F4A_7C15u64);
        let img = image::GrayImage::from_fn(w, h, |_, _| {
            let mut s = state
                .get()
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            s ^= s >> 29;
            state.set(s);
            image::Luma([(s >> 40) as u8])
        });
        let mut out = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
            .expect("encode noise png");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png_bytes(w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbaImage::from_pixel(w, h, image::Rgba([10, 20, 30, 255]));
        let mut out = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
            .expect("encode test png");
        out
    }

    fn base64_image(w: u32, h: u32) -> ImageAttachment {
        ImageAttachment::Base64 {
            data: to_base64(&png_bytes(w, h)),
            media_type: Some("image/png".to_string()),
            detail: None,
        }
    }

    /// protects: the magic bytes — not the file name, not a declared MIME —
    /// decide what a payload is (AC2's ".jpg named PNG is accepted, a
    /// text/html impostor is refused"). fails_when: an extension or a
    /// content-type header can smuggle non-image bytes past the tool.
    /// no_claim: what happens to bytes whose type is correctly read.
    #[test]
    fn sniff_reads_the_magic_not_the_name() {
        assert_eq!(sniff_mime(&png_bytes(1, 1)), Some("image/png"));
        let mut jpg = vec![0xFF, 0xD8, 0xFF, 0xE0];
        jpg.extend_from_slice(&[0; 8]);
        assert_eq!(sniff_mime(&jpg), Some("image/jpeg"));
        assert_eq!(sniff_mime(b"<html><body>not a picture</body></html>"), None);
        assert_eq!(sniff_mime(b""), None);
    }

    /// protects: dimensions come from the image header without a pixel
    /// decode (the metadata face of `read_image`). fails_when: a corrupt
    /// header yields a made-up size. no_claim: pixel fidelity.
    #[test]
    fn header_dimensions_read_png_and_jpeg_headers() {
        assert_eq!(header_dimensions(&png_bytes(37, 11)), Some((37, 11)));
        assert_eq!(header_dimensions(b"not an image"), None);
    }

    /// protects: the bounded base64-head probe reads a picture's size without
    /// decoding the whole payload (the token estimator's eyes), and survives
    /// junk payloads and non-ASCII text without panicking. fails_when: the
    /// probe needs the full decode, guesses a size for non-images, or slices
    /// mid-character. no_claim: what a full pixel decode would find past the
    /// probe window.
    #[test]
    fn base64_prefix_dims_probe_reads_the_head() {
        let png = png_bytes(1278, 1152);
        let b64 = to_base64(&png);
        assert_eq!(base64_prefix_dims(&b64), Some((1278, 1152)));
        // A junk tail behind a valid header does not fool the probe...
        let mut padded = png.clone();
        padded.extend_from_slice(&[0x55; 200_000]);
        assert_eq!(base64_prefix_dims(&to_base64(&padded)), Some((1278, 1152)));
        // ...and non-image or non-ASCII payloads answer None, never a panic.
        assert_eq!(base64_prefix_dims(&to_base64(b"not an image at all")), None);
        assert_eq!(base64_prefix_dims("你好世界"), None);
        assert_eq!(base64_prefix_dims(""), None);
        // A payload cut mid-quartet still probes fine: the head holds the
        // header and the probe trims to whole quartets instead of panicking.
        let cut = &b64[..b64.len() - 2];
        assert_eq!(base64_prefix_dims(cut), Some((1278, 1152)));
    }

    /// protects: the downscale pass keeps a fitting picture byte-identical
    /// (no gratuitous re-encode) and shrinks a too-large one to the
    /// 1568px long edge in the same format. fails_when: every send re-encodes
    /// and degrades, or a too-large picture keeps its size. no_claim: the
    /// exact resampling filter's visual quality.
    #[test]
    fn downscale_shrinks_only_what_does_not_fit() {
        let small = png_bytes(320, 200);
        let (kept, original, final_dims) = downscale_to_fit(&small).expect("small fits");
        assert_eq!(kept, small, "a fitting picture is byte-identical");
        assert_eq!(original, (320, 200));
        assert_eq!(final_dims, (320, 200));

        let big = png_bytes(3200, 800);
        let (shrunk, original, final_dims) = downscale_to_fit(&big).expect("big shrinks");
        assert_eq!(original, (3200, 800));
        assert_eq!(final_dims, (1568, 392), "long edge lands on the cap");
        assert!(shrunk.len() < big.len(), "the wire payload is smaller");
        assert_eq!(sniff_mime(&shrunk), Some("image/png"), "format preserved");
    }

    /// protects: the base64 size math the budget judges by (padded standard
    /// alphabet). fails_when: the measured size drifts from the true wire
    /// size and the budget lies. no_claim: which side of the line a given
    /// screenshot lands on.
    #[test]
    fn base64_size_math_matches_the_wire() {
        assert_eq!(base64_len(0), 0);
        assert_eq!(base64_len(1), 4);
        assert_eq!(base64_len(3), 4);
        assert_eq!(base64_len(4), 8);
        for n in [0usize, 1, 2, 3, 57, 58, 1000] {
            assert_eq!(base64_len(n), to_base64(&vec![0u8; n]).len());
        }
    }

    /// protects: every picture of an in-budget message travels — the count
    /// gate is gone and the budget is measured, not guessed (AC4's "no
    /// silent drop" path). fails_when: an older picture is quietly thinned
    /// to make room. no_claim: how a vision model uses the pictures.
    #[test]
    fn in_budget_messages_keep_every_picture() {
        let mut messages = vec![
            ChatMessage::user("look at the boards"),
            ChatMessage::tool_with_images(
                "t1",
                "shot 1",
                vec![base64_image(200, 200), base64_image(200, 200)],
            ),
            ChatMessage::tool_with_images("t2", "shot 2", vec![base64_image(200, 200)]),
        ];
        apply_image_budget(&mut messages).expect("small pictures fit");
        let count: usize = messages
            .iter()
            .map(|m| match m {
                ChatMessage::Tool { images, .. } | ChatMessage::User { images, .. } => images.len(),
                _ => 0,
            })
            .sum();
        assert_eq!(count, 3, "all three pictures are still attached");
    }

    /// protects: a picture over the per-image line refuses the whole send
    /// with a size inventory naming each item — the refusal is loud and
    /// complete (AC4). fails_when: the oversized picture is silently dropped
    /// or silently shrunk further, or the error lacks the sizes. no_claim:
    /// that 3.5 MB is the right line (a named constant, adjustable).
    #[test]
    fn an_oversized_image_refuses_the_send_with_an_inventory() {
        // Unparseable filler bytes: no downscale pass applies, so the
        // measured base64 size is what the budget judges.
        let fat = ImageAttachment::Base64 {
            data: "x".repeat(MAX_IMAGE_BASE64_BYTES + 1),
            media_type: Some("image/png".to_string()),
            detail: None,
        };
        let mut messages = vec![ChatMessage::tool_with_images("t1", "a fat one", vec![fat])];
        let err = apply_image_budget(&mut messages).expect_err("over the per-image line");
        let text = err.to_string();
        assert!(
            text.contains("request NOT sent"),
            "the refusal is explicit: {text}"
        );
        assert!(
            text.contains("OVER per-image budget"),
            "the violator is named: {text}"
        );
        assert!(
            text.contains("image 1:"),
            "the inventory lists the item: {text}"
        );
        let ChatMessage::Tool { images, .. } = &messages[0] else {
            panic!("expected a tool result");
        };
        assert_eq!(
            images.len(),
            1,
            "the picture is still attached to the message"
        );
    }

    /// protects: the per-message line judges the SUM of a message's
    /// pictures, each individually under the per-image line. fails_when:
    /// four mid-size pictures sneak past as a 12 MB+ message. no_claim: the
    /// per-message number's wisdom (named constant, adjustable).
    #[test]
    fn a_message_sum_over_the_cap_refuses_the_send() {
        let mid = ImageAttachment::Base64 {
            data: "y".repeat(MAX_IMAGE_BASE64_BYTES - 10),
            media_type: Some("image/png".to_string()),
            detail: None,
        };
        let mut messages = vec![ChatMessage::tool_with_images(
            "t1",
            "four mid-size pictures",
            vec![mid.clone(), mid.clone(), mid.clone(), mid],
        )];
        let err = apply_image_budget(&mut messages).expect_err("sum crosses the line");
        let text = err.to_string();
        assert!(
            text.contains("OVER per-message budget"),
            "the sum violator is named: {text}"
        );
    }

    /// protects: a fitting picture that CAN be parsed is downscaled before
    /// it is judged — the wire sees the post-shrink size, and the inventory
    /// says so. fails_when: the budget refuses a picture for size it would
    /// have shed. no_claim: the shrink's visual quality.
    #[test]
    fn the_budget_judges_post_shrink_sizes() {
        let big = base64_image(3200, 800);
        let raw_len = match &big {
            ImageAttachment::Base64 { data, .. } => data.len(),
            _ => unreachable!(),
        };
        assert!(
            raw_len < MAX_IMAGE_BASE64_BYTES,
            "fixture: under the line before"
        );
        let mut messages = vec![ChatMessage::tool_with_images("t1", "shot", vec![big])];
        apply_image_budget(&mut messages).expect("shrinks to fit");
        let ChatMessage::Tool { images, .. } = &messages[0] else {
            panic!("expected a tool result");
        };
        let ImageAttachment::Base64 { data, .. } = &images[0] else {
            panic!("expected a base64 attachment");
        };
        assert!(data.len() < raw_len, "the wire payload was shrunk");
        assert_eq!(sniff_mime(&from_base64(data).unwrap()), Some("image/png"));
    }
}
