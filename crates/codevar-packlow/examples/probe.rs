use codevar_packlow::gpu::{self, BlockSet};

fn main() {
    println!("gpu_available: {}", gpu::gpu_available());
    let blocks: Vec<Vec<u8>> = (0..8).map(|i| vec![i as u8; 4096]).collect();
    let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
    let set = BlockSet::new(&refs);
    println!("signatures: {:?}", gpu::try_block_signatures(&set));
}
