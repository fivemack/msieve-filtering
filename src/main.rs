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
    chunk: &'a [u8]
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
        chunk: &[]
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

fn fast_read_unsigned(number: &[u8]) -> u64 {
    if number.len() > 16 {
        println!("{}", number.len());
        for i in 0..number.len() {
            println!("{}", number[i] as i8);
        }
        assert!(number.len() <= 16);
    }

    let mut k: u64 = 0;
    let L = number.len() - 1;
    let mut M = 1;
    for r in 0..=L {
        k = k + M * ((number[L - r] - 48) as u64);
        M = M * 10
    }
    k
}

fn fast_read_signed(number: &[u8]) -> i64 {
    if number[0] == b'-' {
        -(fast_read_unsigned(&number[1..]) as i64)
    } else {
        fast_read_unsigned(number) as i64
    }
}

fn fast_read_xy(xy: &[u8]) -> SieveIndex {
    let colon = find_fast_byte_after(xy, b':');
    let comma = find_fast_byte_after(xy, b',');
    //  println!("{} {} {}  {} {} {}", xy[colon-1], xy[colon], xy[colon+1], xy[comma-1], xy[comma], xy[comma+1]);
    let xx = fast_read_signed(&xy[0..comma]);
    let yy = fast_read_unsigned(&xy[comma + 1..colon]) as u32;
    return SieveIndex { x: xx, y: yy };
}

impl Chunk<'_> {
    pub fn identify_lines(&mut self, data: &[u8]) -> usize {
        let chunk = &data[self.start_ix..self.end_ix];
        let mut nlines: usize = 0;
        let mut ptr: usize = 0;
        let L = chunk.len();
        while ptr < L {
            self.line_starts.push(ptr);
            let eol = find_fast_byte_after(&chunk[ptr..], b'\n');
            nlines = 1 + nlines;
            ptr = ptr + 1 + eol;
        }
        self.line_starts.push(ptr);
        // mark all the lines as valid
        self.line_valid.resize(nlines, true);
        nlines
    }

    // there are 2015110 comment lines in finish-9282-1472/msieve.dat

    pub fn invalidate_comments(&mut self) -> usize {
        let mut ncomments: usize = 0;
        for i in 0..self.line_starts.len()
        {
            if self.line_valid[i] && self.chunk[self.line_starts[i]]==0x23
            {
                self.line_valid.set(i, false);
                ncomments+=1;
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

    pub fn write_out(&self, data: &[u8], dest_m: &Mutex<&mut [u8]>) {
        let mut dest = (*dest_m).lock().unwrap();
        let mut ptr: usize = 0;
        for i in 0..self.line_starts.len() - 1 {
            if self.line_valid[i] {
                let line_length = self.line_starts[i + 1] - self.line_starts[i];
                (&mut dest[ptr..ptr + line_length])
                    .copy_from_slice(&data[self.line_starts[i]..self.line_starts[i + 1]]);
                ptr += line_length;
            }
        }
    }
}

fn sharded_read(
    shards: &RwLock<[Mutex<HashMap<SieveIndex, usize>>]>,
    sharding_prime: usize,
    start_line: usize,
    chunk: &[u8],
) -> u64 {
    let mut current_line: usize = start_line;
    let mut ptr: usize = 0;
    let L = chunk.len();
    while ptr < L {
        let eol = find_fast_byte_after(&chunk[ptr..], b'\n');
        let xy = fast_read_xy(&chunk[ptr..]);
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

        current_line = 1 + current_line;
        ptr = ptr + 1 + eol;
    }
    current_line as u64
}

fn mark_dupes(
    shards: &RwLock<[Mutex<HashMap<SieveIndex, usize>>]>,
    sharding_prime: usize,
    start_line: usize,
    end_line: usize,
    chunk: &[u8],
) -> BitVec {
    let mut current_line: usize = start_line;
    let mut bv: BitVec = BitVec::new();
    bv.resize(end_line - start_line, false);
    let mut bvix: usize = 0;
    let mut ptr: usize = 0;
    let L = chunk.len();
    while ptr < L {
        let eol = find_fast_byte_after(&chunk[ptr..], b'\n');
        let xy = fast_read_xy(&chunk[ptr..]);
        let shard: usize = (xy.x.rem_euclid(sharding_prime as i64)) as usize
            + sharding_prime * (xy.y.rem_euclid(sharding_prime as u32) as usize)
            - 1;
        let shards_reader = shards.read().unwrap();
        let shard_mutex = shards_reader.get(shard).unwrap();
        // ideally I'd have a second version of the data without the mutex at this point
        // because it's read-only access
        {
            let data = shard_mutex.lock().unwrap();
            if data[&xy] == current_line {
                bv.set(bvix, true);
            }
        }

        current_line = 1 + current_line;
        bvix = 1 + bvix;
        ptr = ptr + 1 + eol;
    }
    bv
}

fn emit_uncancelled_lines(output_filename: String, v: &[Chunk], mmap_r: &[u8]) -> io::Result<()> {
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
    let ofile = File::create(output_filename).unwrap();
    ofile.set_len(total_length as u64)?;
    let mut mmap_w = unsafe { MmapOptions::new().map_mut(&ofile).unwrap() };

    // Use split_at_mut to let us pass the segments of the file to the iterator in parallel
    let mut segments: Vec<Mutex<&mut [u8]>> = Vec::new();
    let mut rest = &mut mmap_w[..];
    for i in 0..n {
        let (seg, rest2) = rest.split_at_mut(valid_lengths[i]);
        segments.push(Mutex::new(seg));
        rest = rest2;
    }

    let _ = (0..n)
        .into_par_iter()
        .map(|i| v[i].write_out(&mmap_r, &segments[i]));
    mmap_w.flush()?;
    Ok(())
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

    // We want really quite a lot of shards to avoid lock contention between the threads

    const sharding_prime: usize = 97;
    let mut v: [Chunk; n_chunks] = [0; n_chunks].map(|_| Chunk::new());

    v[0].start_ix = 0;
    v[n_chunks - 1].end_ix = siz - 1;

    for a in 1..n_chunks {
        let st = ((sz * (a as u64)) / (n_chunks as u64)) as usize;
        let wug = find_fast_byte_after(&mmap[st..], 0x0a);
        v[a - 1].end_ix = st + wug;
        v[a].start_ix = st + wug + 1;
    }
    for a in &mut v
    {
    a.chunk = &mmap[a.start_ix..a.end_ix];
    }

    const n_shards: usize = sharding_prime * sharding_prime - 1;
    let xys: [Mutex<HashMap<SieveIndex, usize>>; n_shards] =
        [0; n_shards].map(|_| Mutex::new(HashMap::new()));
    let xys_under_rwlock: RwLock<_> = RwLock::new(xys);

    // count the lines (needed so each chunk knows where it starts)
    let lines_per_chunk: Vec<usize> = v.par_iter_mut().map(|a| a.identify_lines(&mmap)).collect();
    println!("{} lines in file", lines_per_chunk.iter().sum::<usize>());

    let mut nx: usize = 0;
    for a in 1..n_chunks {
        nx = nx + lines_per_chunk[a - 1];
        v[a].start_line = nx;
        v[a - 1].end_line = nx - 1;
    }

    for a in 0..n_chunks {
        let mut t = v[a].start_ix; let mut ll = 0;
        while mmap[t] == 0x23 {
            println!("Found comment line at offset {} (chunk {} line {}={})", t, a, ll, v[a].start_line+ll);
            t += 1 + find_fast_byte_after(&mmap[t..], 0x0a);
        ll += 1;
        }
        let foo = fast_read_xy(&mmap[t..]);
        println!(
            "{} {}..={} ({}) {} {}",
            a, v[a].start_ix, v[a].end_ix, v[a].start_line, foo.x, foo.y
        )
    }

    // mark all the comment lines as invalid

    let n_comments: usize = v.par_iter_mut().map(|a| a.invalidate_comments()).sum();
    println!("{} comment lines found\n", n_comments);

    let dummy: Vec<u64> = v
        .par_iter()
        .map(|a| {
            sharded_read(
                &xys_under_rwlock,
                sharding_prime,
                a.start_line,
                &mmap[a.start_ix..a.end_ix],
            )
        })
        .collect();
    println!(
        "Chunk 0 of sharded read has {} entries",
        xys_under_rwlock
            .read()
            .unwrap()
            .get(0)
            .unwrap()
            .lock()
            .unwrap()
            .len()
    );

    // Now that we've done the part requiring aggressive locking,
    // can we make the thing read-only for the part where we do the labelling,
    // counting and output?

    // non-obvious question: do we compute the labels twice (once to count and once for output)
    // or do we store them

    // first version: store them
    let chunked_labels: Vec<BitVec> = v
        .par_iter()
        .map(|a| {
            mark_dupes(
                &xys_under_rwlock,
                sharding_prime,
                a.start_line,
                a.end_line,
                &mmap[a.start_ix..a.end_ix],
            )
        })
        .collect();
    emit_uncancelled_lines(args.outfn, &v, &mmap).unwrap();
}
