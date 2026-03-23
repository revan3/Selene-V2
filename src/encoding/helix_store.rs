// src/encoding/helix_store.rs
//
// Helix — mmap-based persistent spike-pattern vocabulary store.
//
// File layout:
//   [0..4]   magic  b"HLXS"
//   [4..8]   version u32 LE = 1
//   [8..12]  n_records u32 LE  (number of valid records)
//   [12..16] reserved u32 = 0
//   [16..]   records[]  each = 96 bytes:
//               [0..64]  SpikePattern ([u64;8] little-endian)
//               [64..96] label: UTF-8, null-padded to 32 bytes
//
// Total header = 16 bytes. Each record = 96 bytes.
// Max ~44 million records per GiB.
//
// Operations:
//   insert(word, pattern)  — O(1) append, dedup on lookup
//   get(word)              — O(n) scan (word lookup by label)
//   nearest(pattern, threshold) — O(n) Jaccard scan (SIMD-friendly popcount)
//   flush()                — msync to disk

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;
use memmap2::MmapMut;

use super::spike_codec::{SpikePattern, similarity};

// ── Constants ─────────────────────────────────────────────────────────────────

const MAGIC: &[u8; 4] = b"HLXS";
const VERSION: u32     = 1;
const HEADER_SIZE: u64 = 16;
const RECORD_SIZE: u64 = 96;
const LABEL_SIZE: usize = 32;
// Pre-allocate file space in chunks to avoid frequent remaps
const GROW_BY_RECORDS: u64 = 4096; // 4096 * 96 ≈ 384 KiB per growth

// ── Record I/O helpers ────────────────────────────────────────────────────────

fn write_record(buf: &mut [u8], pattern: &SpikePattern, label: &str) {
    debug_assert_eq!(buf.len(), 96);
    // Write 8 × u64 LE
    for (i, &word) in pattern.iter().enumerate() {
        buf[i*8..(i+1)*8].copy_from_slice(&word.to_le_bytes());
    }
    // Write label (null-padded to 32 bytes)
    let bytes = label.as_bytes();
    let len   = bytes.len().min(LABEL_SIZE - 1); // leave null terminator
    buf[64..64+len].copy_from_slice(&bytes[..len]);
    for b in &mut buf[64+len..96] { *b = 0; }
}

fn read_pattern(buf: &[u8]) -> SpikePattern {
    debug_assert!(buf.len() >= 64);
    let mut p = [0u64; 8];
    for i in 0..8 {
        p[i] = u64::from_le_bytes(buf[i*8..(i+1)*8].try_into().unwrap());
    }
    p
}

fn read_label(buf: &[u8]) -> &str {
    debug_assert!(buf.len() >= 96);
    let label_bytes = &buf[64..96];
    // Find null terminator
    let end = label_bytes.iter().position(|&b| b == 0).unwrap_or(LABEL_SIZE);
    std::str::from_utf8(&label_bytes[..end]).unwrap_or("")
}

// ── HelixStore ────────────────────────────────────────────────────────────────

pub struct HelixStore {
    mmap:      MmapMut,
    _file:     File,
    n_records: u32,       // cached from header
    capacity:  u32,       // how many record slots the current file has
}

impl HelixStore {
    /// Open or create a Helix store at `path`.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path.as_ref())?;

        let meta = file.metadata()?;
        let file_len = meta.len();

        // If new file, write header + reserve initial space
        let initial_capacity = GROW_BY_RECORDS;
        if file_len < HEADER_SIZE {
            let needed = HEADER_SIZE + initial_capacity * RECORD_SIZE;
            file.set_len(needed)?;
            let mut mmap = unsafe { MmapMut::map_mut(&file)? };
            // magic
            mmap[0..4].copy_from_slice(MAGIC);
            // version
            mmap[4..8].copy_from_slice(&VERSION.to_le_bytes());
            // n_records = 0
            mmap[8..12].copy_from_slice(&0u32.to_le_bytes());
            // reserved
            mmap[12..16].copy_from_slice(&0u32.to_le_bytes());
            mmap.flush()?;

            return Ok(Self {
                mmap,
                _file: file,
                n_records: 0,
                capacity: initial_capacity as u32,
            });
        }

        // Validate existing file
        let mmap = unsafe { MmapMut::map_mut(&file)? };

        if &mmap[0..4] != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                "HelixStore: invalid magic bytes (expected HLXS)"));
        }
        let _ver   = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        let n_rec  = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        let cap    = ((file_len - HEADER_SIZE) / RECORD_SIZE) as u32;

        Ok(Self {
            mmap,
            _file: file,
            n_records: n_rec,
            capacity: cap,
        })
    }

    /// Number of records stored.
    pub fn len(&self) -> u32 { self.n_records }
    pub fn is_empty(&self) -> bool { self.n_records == 0 }

    /// Insert (word, pattern). If word already exists, update its pattern.
    pub fn insert(&mut self, word: &str, pattern: &SpikePattern) -> io::Result<()> {
        // Check for existing label — update in place
        if let Some(idx) = self.find_label_index(word) {
            let offset = (HEADER_SIZE + idx as u64 * RECORD_SIZE) as usize;
            write_record(&mut self.mmap[offset..offset+96], pattern, word);
            return self.mmap.flush_range(offset, 96);
        }

        // Grow file if needed
        if self.n_records >= self.capacity {
            self.grow()?;
        }

        let offset = (HEADER_SIZE + self.n_records as u64 * RECORD_SIZE) as usize;
        write_record(&mut self.mmap[offset..offset+96], pattern, word);

        // Update n_records in header
        self.n_records += 1;
        self.mmap[8..12].copy_from_slice(&self.n_records.to_le_bytes());
        self.mmap.flush_range(8, 4)?;
        self.mmap.flush_range(offset, 96)?;

        Ok(())
    }

    /// Look up a word by exact label — returns its spike pattern.
    pub fn get(&self, word: &str) -> Option<SpikePattern> {
        let idx = self.find_label_index(word)?;
        let offset = (HEADER_SIZE + idx as u64 * RECORD_SIZE) as usize;
        Some(read_pattern(&self.mmap[offset..offset+64]))
    }

    /// Nearest-neighbour Jaccard search. Returns `(word, similarity)` pairs above threshold,
    /// sorted descending, up to `top_n`.
    pub fn nearest<'a>(
        &'a self,
        pattern:   &SpikePattern,
        threshold: f32,
        top_n:     usize,
    ) -> Vec<(String, f32)> {
        let mut hits: Vec<(String, f32)> = Vec::new();

        for i in 0..self.n_records as u64 {
            let offset = (HEADER_SIZE + i * RECORD_SIZE) as usize;
            let sp  = read_pattern(&self.mmap[offset..offset+64]);
            let sim = similarity(pattern, &sp);
            if sim >= threshold {
                let label = read_label(&self.mmap[offset..offset+96]).to_owned();
                hits.push((label, sim));
            }
        }

        hits.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        hits.truncate(top_n);
        hits
    }

    /// Iterate all (label, pattern) pairs — used to rebuild in-memory HashMap.
    pub fn iter_all(&self) -> impl Iterator<Item = (String, SpikePattern)> + '_ {
        (0..self.n_records as u64).map(move |i| {
            let offset = (HEADER_SIZE + i * RECORD_SIZE) as usize;
            let sp    = read_pattern(&self.mmap[offset..offset+64]);
            let label = read_label(&self.mmap[offset..offset+96]).to_owned();
            (label, sp)
        })
    }

    /// Flush all dirty pages to disk.
    pub fn flush(&self) -> io::Result<()> {
        self.mmap.flush()
    }

    // ── Private helpers ────────────────────────────────────────────────────

    fn find_label_index(&self, word: &str) -> Option<u32> {
        for i in 0..self.n_records as u64 {
            let offset = (HEADER_SIZE + i * RECORD_SIZE) as usize;
            if read_label(&self.mmap[offset..offset+96]) == word {
                return Some(i as u32);
            }
        }
        None
    }

    /// Grow the file by GROW_BY_RECORDS additional slots.
    /// This requires remapping, which invalidates the current mmap.
    fn grow(&mut self) -> io::Result<()> {
        // We need to extend the file and remap
        let new_capacity = self.capacity as u64 + GROW_BY_RECORDS;
        let new_len = HEADER_SIZE + new_capacity * RECORD_SIZE;

        // Flush before resize
        self.mmap.flush()?;

        // Re-open via _file reference — set_len through the stored File handle
        // Trick: we drop the mmap and recreate after set_len
        // Since we can't easily reborrow _file after MmapMut is live,
        // we flush and use ftruncate through File's raw fd.
        // On Windows/Linux, set_len on an open file works before remapping.
        self._file.set_len(new_len)?;

        // Remap
        self.mmap = unsafe { MmapMut::map_mut(&self._file)? };
        self.capacity = new_capacity as u32;

        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::spike_codec::encode;
    use std::io::Write as _;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(name)
    }

    #[test]
    fn create_and_insert() {
        let path = tmp_path("helix_test_create.hlx");
        let _ = std::fs::remove_file(&path);

        let mut store = HelixStore::open(&path).unwrap();
        assert_eq!(store.len(), 0);

        let p = encode("selene");
        store.insert("selene", &p).unwrap();
        assert_eq!(store.len(), 1);

        let retrieved = store.get("selene").unwrap();
        assert_eq!(retrieved, p);
    }

    #[test]
    fn update_existing_record() {
        let path = tmp_path("helix_test_update.hlx");
        let _ = std::fs::remove_file(&path);

        let mut store = HelixStore::open(&path).unwrap();
        let p1 = encode("amor");
        let p2 = encode("amor_v2"); // different word → different pattern
        store.insert("amor", &p1).unwrap();
        store.insert("amor", &p2).unwrap(); // should update, not add
        assert_eq!(store.len(), 1, "update must not increase record count");

        let retrieved = store.get("amor").unwrap();
        assert_eq!(retrieved, p2);
    }

    #[test]
    fn nearest_returns_best() {
        let path = tmp_path("helix_test_nearest.hlx");
        let _ = std::fs::remove_file(&path);

        let mut store = HelixStore::open(&path).unwrap();
        for word in &["gato", "cachorro", "peixe", "passaro"] {
            store.insert(word, &encode(word)).unwrap();
        }

        let query = encode("gato");
        let hits = store.nearest(&query, 0.5, 5);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].0, "gato");
        assert!((hits[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn persist_and_reload() {
        let path = tmp_path("helix_test_persist.hlx");
        let _ = std::fs::remove_file(&path);

        let words = ["rio", "mar", "lago"];
        {
            let mut store = HelixStore::open(&path).unwrap();
            for w in &words {
                store.insert(w, &encode(w)).unwrap();
            }
            store.flush().unwrap();
        }

        // Reopen
        let store = HelixStore::open(&path).unwrap();
        assert_eq!(store.len(), words.len() as u32);
        for w in &words {
            let p = store.get(w);
            assert!(p.is_some(), "word '{}' not found after reload", w);
            assert_eq!(p.unwrap(), encode(w));
        }
    }

    #[test]
    fn iter_all_covers_all_records() {
        let path = tmp_path("helix_test_iter.hlx");
        let _ = std::fs::remove_file(&path);

        let words = ["alpha", "beta", "gamma", "delta"];
        let mut store = HelixStore::open(&path).unwrap();
        for w in &words {
            store.insert(w, &encode(w)).unwrap();
        }

        let collected: Vec<_> = store.iter_all().collect();
        assert_eq!(collected.len(), words.len());
    }
}
