use memmap2::MmapOptions;
use std::fs::File;
use std::io;

use std::collections::HashMap;

use std::sync::Mutex;

use clap::Parser;

use rayon::prelude::*;

use bitvec::prelude::*;

pub mod saturating_trits;
use crate::saturating_trits::STritArray;

// need an error type
#[derive(Debug, Clone)]
struct ParseError;

impl ToString for ParseError {
    fn to_string(&self) -> String {
        "Parsing error".to_string()
    }
}

#[derive(Debug, Clone)]
struct PrimeError { naughty_number: u64 }

impl ToString for PrimeError {
    fn to_string(&self) -> String {
    format!("Composite 'prime' {}", self.naughty_number)
    }
}

impl PrimeError {
    pub fn new(bad: u64) -> PrimeError { PrimeError { naughty_number: bad} }
}

// and an overarching error type

#[derive(Debug)]
enum PhiltreError { ParseError, PrimeError }

struct SingletonCounter
{
  rational_side: STritArray,
  algebraic_side: STritArray
}

#[derive(Debug, Parser)]
#[clap(name = "philtre", version = "0.0.0", author = "Tom Womack")]
pub struct PhiltreCmdLine {
    /// Input filename
    infn: String,
    /// Output filename
    outfn: String,
    /// Sharding prime
    #[arg(long, short, default_value_t=97)]
    shard: usize,
    /// Chunk size (megabytes)
    #[arg(long, short, default_value_t=1)]
    chunk: usize
}

#[derive(Clone, Hash)]
struct SieveIndex {
    x: i64,
    y: u32,
}

impl PartialEq for SieveIndex {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl Eq for SieveIndex {}

struct Chunk<'a> {
    start_ix: usize,
    end_ix: usize,
    start_line: usize,
    end_line: usize,
    line_valid: BitVec,
    line_starts: Vec<usize>,
    chunk: &'a [u8],
}

impl Chunk<'_> {
    pub fn new() -> Self {
        Chunk {
            start_ix: 0,
            end_ix: 0,
            start_line: 0,
            end_line: 0,
            line_valid: BitVec::new(),
            line_starts: Vec::new(),
            chunk: &[],
        }
    }
}

fn find_fast_byte_after(start: &[u8], target: u8) -> usize {
    for a in 0..start.len() {
        if start[a] == target {
            return a;
        }
    }
    return start.len();
}

fn compress_prime(number: u64) -> Result<u64, PrimeError>
{
  let compressor: [u8; 210] = [48, 0, 48, 48, 48, 48, 48, 48, 48, 48, 48, 1, 48, 2, 48, 48, 48, 3, 48, 4, 48,
                               48, 48, 5, 48, 48, 48, 48, 48, 6, 48, 7, 48, 48, 48, 48, 48, 8, 48, 48, 48, 9,
			       48, 10, 48, 48, 48, 11, 48, 48, 48, 48, 48, 12, 48, 48, 48, 48, 48, 13, 48, 14,
			       48, 48, 48, 48, 48, 15, 48, 48, 48, 16, 48, 17, 48, 48, 48, 48, 48, 18, 48, 48,
			       48, 19, 48, 48, 48, 48, 48, 20, 48, 48, 48, 48, 48, 48, 48, 21, 48, 48, 48, 22,
			       48, 23, 48, 48, 48, 24, 48, 25, 48, 48, 48, 26, 48, 48, 48, 48, 48, 48, 48, 27,
			       48, 48, 48, 48, 48, 28, 48, 48, 48, 29, 48, 48, 48, 48, 48, 30, 48, 31, 48, 48,
			       48, 32, 48, 48, 48, 48, 48, 33, 48, 34, 48, 48, 48, 48, 48, 35, 48, 48, 48, 48,
			       48, 36, 48, 48, 48, 37, 48, 38, 48, 48, 48, 39, 48, 48, 48, 48, 48, 40, 48, 41,
			       48, 48, 48, 48, 48, 42, 48, 48, 48, 43, 48, 44, 48, 48, 48, 45, 48, 46, 48, 48,
			       48, 48, 48, 48, 48, 48, 48, 47];

  if number<210 { return Ok(number) }
  let rem = number%210;
  let cc = compressor[rem as usize] as u64;
  if cc == 48 { return Err(PrimeError::new(number)) }
  let b = number/210;
  Ok(210+48*b+cc)
}

fn fast_read_hex(number: &[u8]) -> Result<u64, ParseError>
{
  let mut k: u64 = 0;
  let l = number.len()-1;
  let mut m = 1;
  for r in 0..=l
  {
    let ascii = number[l-r];
    let hex = match ascii
    {
      48..57 => ascii-48,
      97..102 => ascii+10-97,
      65..70 => ascii+10-65,
      _ => { return Err(ParseError) }
    };
    k = k + m * (hex as u64);
    m = m*16;
  }
  Ok(k)
}

fn fast_read_unsigned(number: &[u8]) -> Result<u64, ParseError> {
    if number.len() > 16 {
        println!("Unexpectedly long integer of {} characters", number.len());
        for i in 0..number.len() {
            println!("{} {}", number[i] as char, number[i]);
        }
        return Err(ParseError);
        //        assert!(number.len() <= 16);
    }

    let mut k: u64 = 0;
    let l = number.len() - 1;
    let mut m = 1;
    for r in 0..=l {
        k = k + m * ((number[l - r] - 48) as u64);
        m = m * 10
    }
    Ok(k)
}

fn fast_read_signed(number: &[u8]) -> Result<i64, ParseError> {
    if number[0] == b'-' {
        let n = fast_read_unsigned(&number[1..])?;
        Ok(-(n as i64))
    } else {
        let n = fast_read_unsigned(number)?;
        Ok(n as i64)
    }
}

fn fast_read_xy(xy: &[u8]) -> Result<SieveIndex, ParseError> {
    let colon = find_fast_byte_after(xy, b':');
    let comma = find_fast_byte_after(xy, b',');
    //  println!("{} {} {}  {} {} {}", xy[colon-1], xy[colon], xy[colon+1], xy[comma-1], xy[comma], xy[comma+1]);
    let xx = fast_read_signed(&xy[0..comma]);
    let yy = fast_read_unsigned(&xy[comma + 1..colon]);
    Ok(SieveIndex {
        x: xx?,
        y: (yy? as u32),
    })
}

fn handle_line() -> Result<(), ParseError>
{
	todo!();
}

// note that Chunk.line_starts has *one more entry than the number of lines*
// so as to encode the length of the final line without needing a whole new
// large array for lengths

// so line_starts.len() is one more than the number of lines and one more than
// line_valid.size()

impl Chunk<'_> {
    pub fn identify_lines(&mut self) -> usize {
        let mut nlines: usize = 0;
        let mut ptr: usize = 0;
        let chunk_len = self.chunk.len();
        while ptr < chunk_len {
            self.line_starts.push(ptr);
            let eol = find_fast_byte_after(&self.chunk[ptr..], b'\n');
            nlines = 1 + nlines;
            ptr = ptr + 1 + eol;
        }
        self.line_starts.push(ptr);
        // mark all the lines as valid
        self.line_valid.resize(nlines, true);
        nlines
    }

    pub fn invalidate_comments(&mut self) -> usize {
        let mut ncomments: usize = 0;
        for i in 0..self.line_valid.len() {
            if self.line_valid[i] && self.chunk[self.line_starts[i]] == 0x23 {
                self.line_valid.set(i, false);
                ncomments += 1;
            }
        }
        ncomments
    }

    pub fn valid_length(&self) -> usize {
        let mut vl: usize = 0;
        for i in 0..self.line_starts.len() - 1 {
            if self.line_valid[i] {
                vl += self.line_starts[i + 1] - self.line_starts[i];
            }
        }
        vl
    }

    pub fn write_out(&self, dest_m: &Mutex<&mut [u8]>) -> usize {
        let mut dest = (*dest_m).lock().unwrap();
        let mut ptr: usize = 0;
        for i in 0..=self.line_starts.len() - 2 {
            if self.line_valid[i] {
                let mut line_length = self.line_starts[i + 1] - self.line_starts[i];
                if i == self.line_starts.len() - 1 {
                    line_length -= 1;
                }
                (&mut dest[ptr..ptr + line_length]).copy_from_slice(
                    &self.chunk[self.line_starts[i]..self.line_starts[i] + line_length],
                );
                ptr += line_length;
            }
        }
        ptr
    }

    pub fn sharded_read(
        &self,
        shards: &Vec<Mutex<HashMap<SieveIndex, usize>>>,
        sharding_prime: usize,
    ) -> (usize, Vec<usize>) {
        let mut lines_read = 0;
        let mut bad_line_vector = Vec::new();
        for i in 0..=(self.end_line - self.start_line) {
            if self.line_valid[i] {
                lines_read += 1;
                let current_line = self.start_line + i;
                let xy_fallible = fast_read_xy(&self.chunk[self.line_starts[i]..]);
                // this is the first time we've read the line, it's totally possible there's an error
                // in which case we mark the line as invalid and carry on
                match xy_fallible {
                    Err(err) => {
                        println!(
                            "Problem parsing line {}: {}",
                            i + self.start_line,
                            err.to_string()
                        );
                        // because we're doing this without being able to modify self,
                        // we can't do "self.line_valid.set(i, false);"
                        // instead we record the bad line for a later serial marking pass
                        bad_line_vector.push(i);
                    }
                    Ok(xy) => {
                        let shard: usize = (xy.x.rem_euclid(sharding_prime as i64)) as usize
                            + sharding_prime * (xy.y.rem_euclid(sharding_prime as u32) as usize)
                            - 1;
                        let shard_mutex = shards.get(shard).unwrap();
                        // block where we actually need to do something locked
                        {
                            let mut data = shard_mutex.lock().unwrap();
                            if !data.contains_key(&xy) || data[&xy] > current_line {
                                data.insert(xy, current_line);
                            }
                        }
                    }
                }
            }
        }
        (lines_read, bad_line_vector)
    }

    pub fn mark_dupes(
        &mut self,
        shards: &Vec<Mutex<HashMap<SieveIndex, usize>>>,
        sharding_prime: usize,
    ) -> usize {
        let mut duplicates = 0;
        for i in 0..self.line_valid.len() {
            if self.line_valid[i] {
                // by the time we're calling this function, we have already done fast_read_xy on
                // every line in the file and marked unreadable ones as invalid,
                // so we can unwrap() and the panic if it's unreadable is a logic error
                let xy = fast_read_xy(&self.chunk[self.line_starts[i]..]).unwrap();
                let shard: usize = (xy.x.rem_euclid(sharding_prime as i64)) as usize
                    + sharding_prime * (xy.y.rem_euclid(sharding_prime as u32) as usize)
                    - 1;
                let shard_mutex = shards.get(shard).unwrap();
                {
                    let data = shard_mutex.lock().unwrap();
                    let current_line = self.start_line + i;
                    if data[&xy] != current_line {
                        self.line_valid.set(i, false);
                        duplicates += 1;
                    }
                }
            }
        }
        duplicates
    }

    pub fn count_singletons(
        &mut self,
	counter: &SingletonCounter) -> usize
    {
	let mut bad_lines = 0;
        for i in 0..self.line_valid.len() {
	    if self.line_valid[i] {
                let first_colon = find_fast_byte_after(&self.chunk[self.line_starts[i]..], b':');
                let second_colon = find_fast_byte_after(&self.chunk[self.line_starts[i]+first_colon..], b':');
		let newline = find_fast_byte_after(&self.chunk[self.line_starts[i]+first_colon+second_colon..], b'\n');
		let rat_slice = &self.chunk[self.line_starts[i]+first_colon .. self.line_starts[i]+first_colon+second_colon];
		let alg_slice = &self.chunk[self.line_starts[i]+first_colon+second_colon .. self.line_starts[i]+first_colon+second_colon+newline];
		let rat_ok = count_it(rat_slice, counter.rational_side);
		let alg_ok = count_it(alg_slice, counter.algebraic_side);
		if rat_ok.is_err() || alg_ok.is_err()
		{
		    self.line_valid.set(i, false);
		    bad_lines += 1;
		}
    	    }
	}

	bad_lines
    }
}

fn emit_uncancelled_lines(output_filename: String, v: &[Chunk]) -> io::Result<()> {
    // Get the sum of the lengths of the uncancelled parts of the chunks
    let n = v.len();
    let valid_lengths: Vec<usize> = v.par_iter().map(|c| c.valid_length()).collect();
    let mut start: Vec<usize> = Vec::new();

    let mut sp: usize = 0;
    for i in 0..n {
        start.push(sp);
        sp += valid_lengths[i];
    }
    let total_length: usize = sp;

    // And actually perform the writes
    {
        let ofile1 = File::create(output_filename.clone()).unwrap();
        ofile1.set_len(total_length as u64)?;
    }
    // unfortunately ofile1 is opened write-only by ::create and mmap requires RdWr
    let ofile = File::options()
        .read(true)
        .write(true)
        .open(output_filename)?;
    let mut mmap_w = unsafe { MmapOptions::new().map_mut(&ofile).unwrap() };

    // Use split_at_mut to let us pass the segments of the file to the iterator in parallel
    let mut segments: Vec<Mutex<&mut [u8]>> = Vec::new();
    let mut rest = &mut mmap_w[..];
    for i in 0..n {
        let (seg, rest2) = rest.split_at_mut(valid_lengths[i]);
        segments.push(Mutex::new(seg));
        rest = rest2;
    }
    println!("Writing out uncancelled lines");
    let out_bytes: usize = (0..n)
        .into_par_iter()
        .map(|i| v[i].write_out(&segments[i]))
        .sum();
    println!("Wrote out {} bytes", out_bytes);
    mmap_w.flush()?;
    println!("mmap flush complete");
    Ok(())
}

fn main() {
    let args = PhiltreCmdLine::parse();

    let file = File::open(args.infn).unwrap();
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    let siz: usize = mmap.len();
    let sz: u64 = siz as u64;

    // Chunks for handling the file in a multi-threaded way
    // We want each chunk to begin just after an 0x0a byte
    // and end at an 0x0a byte
    let chunk_size:usize = args.chunk * 1048576;
    let n_chunks: usize = siz/chunk_size;

    println!("File is {} bytes long; using {} chunks", siz, n_chunks);

    // We want really quite a lot of shards to avoid lock contention between the threads

    let sharding_prime: usize = args.shard;
    let mut v: Vec<Chunk> = (0..n_chunks).map(|_| Chunk::new()).collect();

    v[0].start_ix = 0;
    v[n_chunks - 1].end_ix = siz - 1;

    for a in 1..n_chunks {
        let st = ((sz * (a as u64)) / (n_chunks as u64)) as usize;
        let wug = find_fast_byte_after(&mmap[st..], 0x0a);
        v[a - 1].end_ix = st + wug;
        v[a].start_ix = st + wug + 1;
    }
    for a in &mut v {
        a.chunk = &mmap[a.start_ix..=a.end_ix];
    }

    let n_shards: usize = sharding_prime * sharding_prime - 1;
    let xys: Vec<Mutex<HashMap<SieveIndex, usize>>> = (0..n_shards).map(|_| Mutex::new(HashMap::new())).collect();

    // count the lines (needed so each chunk knows where it starts)
    let lines_per_chunk: Vec<usize> = v.par_iter_mut().map(|a| a.identify_lines()).collect();
    println!("{} lines in file", lines_per_chunk.iter().sum::<usize>());

    let mut nx: usize = 0;
    for a in 0..n_chunks {
        v[a].start_line = nx;
        nx = nx + lines_per_chunk[a];
        v[a].end_line = nx - 1;
    }

/*    for a in 0..n_chunks {
        let mut t = v[a].start_ix;
        let mut ll = 0;
        while mmap[t] == 0x23 {
            println!(
                "Found comment line at offset {} (chunk {} line {}={})",
                t,
                a,
                ll,
                v[a].start_line + ll
            );
            t += 1 + find_fast_byte_after(&mmap[t..], 0x0a);
            ll += 1;
        }
        let foo = fast_read_xy(&mmap[t..]).unwrap();
        println!(
            "{} {}..={} ({}..={}) {} {}",
            a, v[a].start_ix, v[a].end_ix, v[a].start_line, v[a].end_line, foo.x, foo.y
        )
    } */

    // mark all the comment lines as invalid

    let n_comments: usize = v.par_iter_mut().map(|a| a.invalidate_comments()).sum();
    println!("{} comment lines found\n", n_comments);

    let og_pair: Vec<(usize, Vec<usize>)> = v
        .par_iter()
        .map(|a| a.sharded_read(&xys, sharding_prime))
        .collect();
    let og: (usize, usize) = og_pair
        .iter()
        .fold((0, 0), |(sx, sy), (x, y)| (sx + x, sy + y.len()));
    println!(
        "{} non-comment lines read for sharding, {} bad\n",
        og.0, og.1
    );

    for a in 0..n_chunks {
        for b in &(og_pair[a].1) {
            v[a].line_valid.set(*b, false);
        }
    }

    println!(
        "Chunk 0 of sharded read has {} entries (expect about {})",
        xys .get(0)
            .unwrap()
            .lock()
            .unwrap()
            .len(),
        og.0 / n_shards
    );
    println!("Starting duplicate removal");
    let n_duplicates: usize = v
        .par_iter_mut()
        .map(|a| a.mark_dupes(&xys, sharding_prime))
        .sum();
    println!("{} duplicates found", n_duplicates);
    emit_uncancelled_lines(args.outfn, &v).unwrap();
}
