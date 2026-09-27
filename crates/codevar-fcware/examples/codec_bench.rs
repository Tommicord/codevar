use codevar_fcware::{Codec, compress, decompress};
use std::time::Instant;

fn codecs() -> Vec<(&'static str, Codec)> {
    vec![
        ("LzMatch", Codec::LzMatch),
        ("Huffman", Codec::Huffman),
        ("Substring", Codec::Substring),
        ("Dysu4k", Codec::dysu_default()),
        ("Dysu16k", Codec::Dysu { block_size: 16384 }),
        ("Dysu64k", Codec::Dysu { block_size: 65536 }),
        ("Delta", Codec::Delta),
    ]
}

fn bench(name: &str, data: &[u8]) {
    println!("== {name} ({} bytes)", data.len());
    for (cname, codec) in codecs() {
        let t = Instant::now();
        let out = compress(data, codec);
        let dt = t.elapsed();
        match out {
            Ok(c) => {
                let back = decompress(&c);
                let ok = back.as_deref() == Ok(data);
                println!(
                    "  {cname:10} {:>8} bytes  ratio {:.3}  enc {:?}  roundtrip {}",
                    c.len(),
                    c.len() as f64 / data.len() as f64,
                    dt,
                    ok
                );
            }
            Err(e) => println!("  {cname:10} error {e:?}"),
        }
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

fn noise_rgba(w: usize, h: usize) -> Vec<u8> {
    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut v = Vec::with_capacity(w * h * 4);
    for _ in 0..(w * h * 4) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        v.push((state >> 24) as u8);
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
            let b2 = b.wrapping_add(((x / 16) % 3) as u8);
            v.extend_from_slice(&[r.wrapping_add((y / 32) as u8), g, b2, 255]);
        }
    }
    v
}

fn main() {
    bench("gradient_rgba_512", &gradient_rgba(512, 512));
    bench("noise_rgba_512", &noise_rgba(512, 512));
    bench("photo_like_rgba_512", &photo_like_rgba(512, 512));

    let files = [
        "/usr/share/backgrounds/warty-final-ubuntu.png",
        "/usr/share/backgrounds/jdituicha-raccoon1-dark.jpg",
        "/usr/share/pixmaps/debian-logo.png",
        "/usr/share/icons/Yaru-dark/48x48/places/folder.png",
        "/usr/share/backgrounds/moskalenko-v-Low_poly_Raccoon_Light.webp",
    ];
    for f in files {
        match std::fs::read(f) {
            Ok(d) => bench(&format!("file {f}"), &d),
            Err(e) => println!("skip {f}: {e}"),
        }
    }
}
