use memmap2::MmapOptions;
use std::fs::File;
use std::io;
use std::io::Write;

use std::collections::HashMap;

use std::sync::Mutex;
use std::sync::RwLock;

use clap::Parser;

use rayon::prelude::*;

use bitvec::prelude::*;

// need an error type
#[derive(Debug, Clone)]
struct ParseError;

impl ToString for ParseError {
    fn to_string(&self) -> String {
        "Parsing error".to_string()
    }
}

#[derive(Debug, Parser)]
#[clap(name = "philtre", version = "0.0.0", author = "Tom Womack")]
pub struct PhiltreCmdLine {
    /// Input filename
    infn: String,
    /// Output filename
    outfn: String,
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
    let L = number.len() - 1;
    let mut M = 1;
    for r in 0..=L {
        k = k + M * ((number[L - r] - 48) as u64);
        M = M * 10
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

// note that Chunk.line_starts has *one more entry than the number of lines*
// so as to encode the length of the final line without needing a whole new
// large array for lengths

// so line_starts.len() is one more than the number of lines and one more than
// line_valid.size()

impl Chunk<'_> {
    pub fn identify_lines(&mut self) -> usize {
        let mut nlines: usize = 0;
        let mut ptr: usize = 0;
        let L = self.chunk.len();
        while ptr < L {
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
        println!(
            "Writing out from chunk {}..={} to dest slice of size {}",
            self.start_ix,
            self.end_ix,
            dest.len()
        );
        println!(
            "Chunk size is {}, last two self.line_starts are {} {}",
            self.chunk.len(),
            self.line_starts[self.line_starts.len() - 2],
            self.line_starts[self.line_starts.len() - 1]
        );
        println!(
            "Last three bytes of my src are 0x{:02x} 0x{:02x} 0x{:02x}",
            self.chunk[self.chunk.len() - 3],
            self.chunk[self.chunk.len() - 2],
            self.chunk[self.chunk.len() - 1]
        );

        let mut ptr: usize = 0;
        for i in 0..=self.line_starts.len() - 2 {
            if self.line_valid[i] {
                let mut line_length = self.line_starts[i + 1] - self.line_starts[i];
                if (i == self.line_starts.len() - 1) {
                    line_length -= 1;
                }
                if ((ptr + line_length) as i64 - dest.len() as i64).abs() < 100 {
                    println!("Calling memcpy on my line {} (I have {} lines); src {}+{} dest {}+{} src_len {} dest_len {}", i, self.line_starts.len(), self.line_starts[i], self.line_starts[i]+line_length, ptr, ptr+line_length, self.chunk.len(), dest.len());
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
        shards: &RwLock<Vec<Mutex<HashMap<SieveIndex, usize>>>>,
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
                        let shards_reader = shards.read().unwrap();
                        let shard_mutex = shards_reader.get(shard).unwrap();
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
        shards: &RwLock<Vec<Mutex<HashMap<SieveIndex, usize>>>>,
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
                let shards_reader = shards.read().unwrap();
                let shard_mutex = shards_reader.get(shard).unwrap();
                // ideally I'd have a second version of the data without the mutex at this point
                // because it's read-only access
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
        println!("Segment {} created with size {}", i, valid_lengths[i]);
        rest = rest2;
    }

    let out_bytes: usize = (0..n)
        .into_par_iter()
        .map(|i| v[i].write_out(&segments[i]))
        .sum();
    println!("Wrote out {} bytes", out_bytes);
    mmap_w.flush()?;
    Ok(())
}

fn main() {
    let args = PhiltreCmdLine::parse();

    let file = File::open(args.infn).unwrap();
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    let siz: usize = mmap.len();
    let sz: u64 = siz as u64;
    println!("File is {} bytes long", siz);

    // Chunks for handling the file in a multi-threaded way
    // We want each chunk to begin just after an 0x0a byte
    // and end at an 0x0a byte
    const n_chunks: usize = 280;

    // We want really quite a lot of shards to avoid lock contention between the threads

    let sharding_prime: usize = 397;
    let mut v: [Chunk; n_chunks] = [0; n_chunks].map(|_| Chunk::new());

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
    let mut xys: Vec<Mutex<HashMap<SieveIndex, usize>>> = Vec::new();
    for a in 0..n_shards
    {
	xys.push(Mutex::new(HashMap::new()));
    }
    let xys_under_rwlock: RwLock<_> = RwLock::new(xys);

    // count the lines (needed so each chunk knows where it starts)
    let lines_per_chunk: Vec<usize> = v.par_iter_mut().map(|a| a.identify_lines()).collect();
    println!("{} lines in file", lines_per_chunk.iter().sum::<usize>());

    let mut nx: usize = 0;
    for a in 0..n_chunks {
        v[a].start_line = nx;
        nx = nx + lines_per_chunk[a];
        v[a].end_line = nx - 1;
    }

    for a in 0..n_chunks {
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
    }

    // mark all the comment lines as invalid

    let n_comments: usize = v.par_iter_mut().map(|a| a.invalidate_comments()).sum();
    println!("{} comment lines found\n", n_comments);

    let og_pair: Vec<(usize, Vec<usize>)> = v
        .par_iter()
        .map(|a| a.sharded_read(&xys_under_rwlock, sharding_prime))
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
        xys_under_rwlock
            .read()
            .unwrap()
            .get(0)
            .unwrap()
            .lock()
            .unwrap()
            .len(),
        og.0 / n_shards
    );

    // Now that we've done the part requiring aggressive locking,
    // can we make the thing read-only for the part where we do the labelling,
    // counting and output?

    let n_duplicates: usize = v
        .par_iter_mut()
        .map(|a| a.mark_dupes(&xys_under_rwlock, sharding_prime))
        .sum();
    println!("{} duplicates found", n_duplicates);
    emit_uncancelled_lines(args.outfn, &v).unwrap();
}
