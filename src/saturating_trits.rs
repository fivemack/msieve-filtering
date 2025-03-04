// you can pack five trits into a byte

use std::sync::Mutex;

enum SaturatingTritValue { None, One, Lots }

struct STritArray
{
  data: Vec<Mutex<Vec<u8>>>,
  ms: usize,
  sz: usize
}

fn sati(old: u8, ix: usize) -> u8
{
  assert!(ix<5);
  match ix {
  0   => if old%3==2 { old } else { old+1 },
  1   => { let b = old/3;  if b%3==2 { old } else { old+3  } },
  2   => { let b = old/9;  if b%3==2 { old } else { old+9  } },
  3   => { let b = old/27; if b%3==2 { old } else { old+27 } },
  4   => if old>=162 { old } else { old+81 }
  5.. => 243
  }
}

fn extract_trit(dat: u8, ix: usize) -> u8
{
  assert!(ix<5);
  match ix {
  0 => dat%3,
  1 => (dat/3)%3,
  2 => (dat/9)%3,
  3 => (dat/27)%3,
  4.. => dat/81
  }
}

fn wrap(trit: u8) -> SaturatingTritValue
{
  assert!(trit<3);
  match trit
  {
    0 => SaturatingTritValue::None,
    1 => SaturatingTritValue::One,
    2.. => SaturatingTritValue::Lots	
  }
}

impl STritArray
{
  pub fn new() -> Self { STritArray { data: Vec::new(), ms: 7, sz: 0 } }
  pub fn init(sz:usize, mutex_shift:usize) -> Self {
  let blocksz = 5<<mutex_shift;
  let nblocks = (sz+blocksz-1)/blocksz;
  STritArray {
  ms: mutex_shift,
  sz: sz,
  data: (0..nblocks).map(|_| { let mut v=Vec::new(); v.resize(1<<mutex_shift,0); Mutex::new(v) }).collect() } }

  fn read(&self, ix:usize) -> SaturatingTritValue
  {
    let sub_byte = ix%5;
    let byte_ix = ix/5;
    let block_ix = byte_ix>>self.ms;
    let subblock_ix = byte_ix&((1<<self.ms)-1);
    wrap(extract_trit(*self.data.get(block_ix).unwrap().lock().unwrap().get(subblock_ix).unwrap(), sub_byte))
  }
    
  fn increment(&self, ix:usize) -> ()
  {
    let sub_byte = ix%5;
    let byte_ix = ix/5;
    let block_ix = byte_ix>>self.ms;
    let subblock_ix = byte_ix&((1<<self.ms)-1);
    let mut block = self.data.get(block_ix).unwrap().lock().unwrap();
    block[subblock_ix] = sati(block[subblock_ix], sub_byte);
  }
}