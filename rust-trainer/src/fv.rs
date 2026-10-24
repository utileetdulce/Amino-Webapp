// Fixed-capacity inline vector so the whole game state is `Copy` (cloning = memcpy).
use std::ops::{Index, IndexMut};

#[derive(Clone, Copy)]
pub struct FV<T: Copy + Default, const N: usize> {
    a: [T; N],
    n: u8,
}

impl<T: Copy + Default, const N: usize> Default for FV<T, N> {
    fn default() -> Self { Self { a: [T::default(); N], n: 0 } }
}

impl<T: Copy + Default, const N: usize> FV<T, N> {
    pub fn new() -> Self { Self::default() }
    #[inline] pub fn len(&self) -> usize { self.n as usize }
    #[inline] pub fn is_empty(&self) -> bool { self.n == 0 }
    #[inline] pub fn push(&mut self, v: T) { self.a[self.n as usize] = v; self.n += 1; }
    #[inline] pub fn pop(&mut self) -> Option<T> { if self.n == 0 { None } else { self.n -= 1; Some(self.a[self.n as usize]) } }
    pub fn remove(&mut self, i: usize) -> T {
        let v = self.a[i];
        let n = self.n as usize;
        for k in i..n - 1 { self.a[k] = self.a[k + 1]; }
        self.n -= 1;
        v
    }
    pub fn insert(&mut self, i: usize, v: T) {
        let n = self.n as usize;
        let mut k = n;
        while k > i { self.a[k] = self.a[k - 1]; k -= 1; }
        self.a[i] = v;
        self.n += 1;
    }
    #[inline] pub fn as_slice(&self) -> &[T] { &self.a[..self.n as usize] }
    #[inline] pub fn as_mut_slice(&mut self) -> &mut [T] { let n = self.n as usize; &mut self.a[..n] }
    #[inline] pub fn iter(&self) -> std::slice::Iter<'_, T> { self.as_slice().iter() }
    pub fn clear(&mut self) { self.n = 0; }
}

impl<T: Copy + Default + PartialEq, const N: usize> FV<T, N> {
    pub fn index_of(&self, v: T) -> Option<usize> { self.iter().position(|x| *x == v) }
    pub fn contains(&self, v: T) -> bool { self.index_of(v).is_some() }
}

impl<T: Copy + Default, const N: usize> Index<usize> for FV<T, N> {
    type Output = T;
    #[inline] fn index(&self, i: usize) -> &T { &self.a[i] }
}
impl<T: Copy + Default, const N: usize> IndexMut<usize> for FV<T, N> {
    #[inline] fn index_mut(&mut self, i: usize) -> &mut T { &mut self.a[i] }
}
