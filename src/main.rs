use memmap::MmapOptions;
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

fn find_fast_byte_after(start:&[u8], target:u8)
{

}

fn fast_read_unsigned(number: &[u8]) -> u64
{
  number.len() as u64
}

fn fast_read_signed(number: &[u8]) -> i64
{
  let mut negative: bool = false;
  if number[0]==b'-'
  { -(fast_read_unsigned(&number[1..]) as i64) }
  else { fast_read_unsigned(number) as i64 }
}

fn main() {
    let args = PhiltreCmdLine::parse();
    println!("Hello, world! {} {}", args.infn, args.outfn);

    let file = File::open(args.infn).unwrap();
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    let siz: usize = mmap.len();
    println!("File is {} bytes long", siz);
    println!("{} {}", mmap[0] as u8, mmap[siz-1] as u8);
    println!("{}", mmap[siz] as u8);

    // Chunks for handling the file in a multi-threaded way
    // We want each chunk to begin just after an 0x0a byte
    // and end at an 0x0a byte
    let chunks: u32 = 280;
    
}
