use memmap2::MmapOptions;
use std::io::Write;
use std::fs::File;

use clap::Parser;

#[derive(Debug,Parser)]
#[clap(name="philtre", version="0.0.0", author="Tom Womack")]
pub struct PhiltreCmdLine
{
	/// Input filename
	infn: String,
	/// Output filename
	outfn: String
}

struct SieveIndex
{
  x: i64,
  y: u32
}

struct Chunk
{
  start_ix: usize,
  end_ix: usize
}

fn find_fast_byte_after(start:&[u8], target:u8) -> usize
{
  for a in 0..start.len()
  {
    if start[a]==target { return a }
  }
  return start.len()
}

fn fast_read_unsigned(number: &[u8]) -> u64
{
  assert!(number.len() <= 16);

  let mut k: u64 = 0;
  let L = number.len()-1;
  let mut M = 1;
  for r in 0..=L
  {
    k = k + M * ((number[L-r]-48) as u64);
    M = M * 10
  }
  k
}

fn fast_read_signed(number: &[u8]) -> i64
{
  if number[0]==b'-'
  { -(fast_read_unsigned(&number[1..]) as i64) }
  else { fast_read_unsigned(number) as i64 }
}

fn fast_read_xy(xy: &[u8]) -> SieveIndex
{
  let colon = find_fast_byte_after(xy, b':');
  let comma = find_fast_byte_after(xy, b',');
  println!("{} {} {}  {} {} {}", xy[colon-1], xy[colon], xy[colon+1], xy[comma-1], xy[comma], xy[comma+1]);
  let xx = fast_read_signed(&xy[0..comma]);
  let yy = fast_read_unsigned(&xy[comma+1..colon]) as u32;
  return SieveIndex { x: xx,y: yy }
}

fn main() {
    let args = PhiltreCmdLine::parse();
    println!("Hello, world! {} {}", args.infn, args.outfn);

    let file = File::open(args.infn).unwrap();
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    let siz: usize = mmap.len();
    let sz: u64 = siz as u64;
    println!("File is {} bytes long", siz);

    // Chunks for handling the file in a multi-threaded way
    // We want each chunk to begin just after an 0x0a byte
    // and end at an 0x0a byte
    const n_chunks: usize = 280;
    let sharding_prime: u64 = 11;
    let mut v: [Chunk;n_chunks] = [0;n_chunks].map(|_| Chunk { start_ix:0, end_ix:0 });

    v[0].start_ix = 0;
    v[n_chunks-1].end_ix = siz-1;

    for a in 1..n_chunks
    {
      let st = ((sz*(a as u64))/(n_chunks as u64)) as usize;
      let wug = find_fast_byte_after(&mmap[st..], 0x0a);
      v[a-1].end_ix = st+wug;
      v[a].start_ix = st+wug+1;
    }

    for a in 0..n_chunks
    {
      let foo = fast_read_xy(&mmap[v[a].start_ix..]);
      println!("{} {}..={} {} {}",a, v[a].start_ix, v[a].end_ix, foo.x, foo.y)
    }
}
