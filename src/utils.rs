#[cfg(not(target_arch = "wasm32"))]
pub use rayon::prelude::*;

#[cfg(target_arch = "wasm32")]
pub use wasm_compat::*;

#[cfg(target_arch = "wasm32")]
pub mod wasm_compat {
    pub trait IntoParallelIterator {
        type Item;
        type Iter: Iterator<Item = Self::Item>;
        fn into_par_iter(self) -> Self::Iter;
    }

    impl<I: IntoIterator> IntoParallelIterator for I {
        type Item = I::Item;
        type Iter = I::IntoIter;
        fn into_par_iter(self) -> Self::Iter {
            self.into_iter()
        }
    }

    pub trait IntoParallelRefIterator<'data> {
        type Item: 'data;
        type Iter: Iterator<Item = Self::Item>;
        fn par_iter(&'data self) -> Self::Iter;
    }

    impl<'data, I: 'data + ?Sized> IntoParallelRefIterator<'data> for I
    where
        &'data I: IntoIterator,
    {
        type Item = <&'data I as IntoIterator>::Item;
        type Iter = <&'data I as IntoIterator>::IntoIter;
        fn par_iter(&'data self) -> Self::Iter {
            self.into_iter()
        }
    }

    pub trait IntoParallelRefMutIterator<'data> {
        type Item: 'data;
        type Iter: Iterator<Item = Self::Item>;
        fn par_iter_mut(&'data mut self) -> Self::Iter;
    }

    impl<'data, I: 'data + ?Sized> IntoParallelRefMutIterator<'data> for I
    where
        &'data mut I: IntoIterator,
    {
        type Item = <&'data mut I as IntoIterator>::Item;
        type Iter = <&'data mut I as IntoIterator>::IntoIter;
        fn par_iter_mut(&'data mut self) -> Self::Iter {
            self.into_iter()
        }
    }

    // Abstract parallel slice traits (simulating rayon)
    pub trait ParallelSlice<T> {
        fn par_chunks<'a>(&'a self, chunk_size: usize) -> std::slice::Chunks<'a, T>;
    }

    impl<T> ParallelSlice<T> for [T] {
        fn par_chunks<'a>(&'a self, chunk_size: usize) -> std::slice::Chunks<'a, T> {
            self.chunks(chunk_size)
        }
    }

    pub trait ParallelSliceMut<T> {
        fn par_chunks_mut<'a>(&'a mut self, chunk_size: usize) -> std::slice::ChunksMut<'a, T>;
    }

    impl<T> ParallelSliceMut<T> for [T] {
        fn par_chunks_mut<'a>(&'a mut self, chunk_size: usize) -> std::slice::ChunksMut<'a, T> {
            self.chunks_mut(chunk_size)
        }
    }
}
