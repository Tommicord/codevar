use codevar_fcware::{Codec, compress, decompress};
use std::time::Instant;

fn chain_delta_huffman(data: &[u8]) -> Result<Vec<u8>, codevar_fcware::CompressorError> {
    let d = compress(data, Codec::Delta)?;
    compress(&d, Codec::Huffman)
}
fn unchain(data: &[u8]) -> Result<Vec<u8>, codevar_fcware::CompressorError> {
    let d = decompress(data)?;
    decompress(&d)
}

fn bench(name: &str, data: &[u8]) {
    println!("== {name} ({} bytes)", data.len());
    let variants: Vec<(&str, Vec<u8>)> = vec![
        ("Huffman", compress(data, Codec::Huffman).unwrap()),
        ("Delta", compress(data, Codec::Delta).unwrap()),
        ("Delta->Huffman", chain_delta_huffman(data).unwrap()),
        ("Dysu4k", compress(data, Codec::dysu_default()).unwrap()),
    ];
    for (n, out) in variants {
        let ok = match n {
            "Delta->Huffman" => unchain(&out).as_deref() == Ok(data),
            _ => decompress(&out).as_deref() == Ok(data),
        };
        println!("  {n:16} {:>8} bytes ratio {:.4} rt {ok}", out.len(), out.len() as f64 / data.len() as f64);
    }
}

fn gradient_rgba(w: usize, h: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            v.push((x & 0xff) as u8);
            v.push((y & 0xff) as u8);
            v.push(((x + y) & 0xff) as u8);
            v.push(255);
        }
    }
    v
}
fn photo_like_rgba(w: usize, h: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / w as f32;
            let fy = y as f32 / h as f32;
            let r = (255.0 * (fx * 3.0).sin().abs()) as u8;
            let g = (255.0 * (fy * 5.0).cos().abs()) as u8;
            let b = (255.0 * ((fx + fy) * 7.0).sin().abs()) as u8;
            v.extend_from_slice(&[r.wrapping_add((y / 32) as u8), g, b.wrapping_add(((x / 16) % 3) as u8), 255]);
        }
    }
    v
}

fn main() {
    bench("gradient_rgba_512", &gradient_rgba(512, 512));
    bench("photo_like_rgba_512", &photo_like_rgba(512, 512));
    for f in [
        "/usr/share/backgrounds/warty-final-ubuntu.png",
        "/usr/share/backgrounds/jdituicha-raccoon1-dark.jpg",
        "/usr/share/pixmaps/debian-logo.png",
        "/usr/share/backgrounds/moskalenko-v-Low_poly_Raccoon_Light.webp",
    ] {
        if let Ok(d) = std::fs::read(f) { bench(f, &d); }
    }
}
