//! Platform-level data formats, shared by every Xbox Series title.
//!
//! Each module is pure (bytes in, bytes out, no I/O): [`texture`] (the
//! GPU's tiled texture layouts and copies to and from linear rows),
//! [`pagetable`] (the title-memory page table the GPU's shaders look
//! addresses up in), [`extents`] (where the copies a submission flushed
//! sit in memory, so a byte window reaches only the copies that can
//! hold it). Planned: BCPack textures, audio codecs (XMA2, Opus), and
//! the zlib streams used by hardware decompression.

pub mod bcpack;
pub mod extents;
pub mod pagetable;
pub mod texture;
pub mod xma;
