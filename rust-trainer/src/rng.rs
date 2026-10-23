// xoshiro256** for all non-state randomness (JS used Math.random)
#[derive(Clone)]
pub struct Rng { s: [u64; 4] }
impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut z = seed;
        let mut next = || { z = z.wrapping_add(0x9E3779B97F4A7C15); let mut x = z; x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB); x ^ (x >> 31) };
        Rng { s: [next(), next(), next(), next()] }
    }
    #[inline] pub fn u64(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0]; self.s[3] ^= self.s[1]; self.s[1] ^= self.s[2]; self.s[0] ^= self.s[3];
        self.s[2] ^= t; self.s[3] = self.s[3].rotate_left(45);
        r
    }
    #[inline] pub fn f64(&mut self) -> f64 { (self.u64() >> 11) as f64 / (1u64 << 53) as f64 }
    #[inline] pub fn below(&mut self, n: usize) -> usize { (self.f64() * n as f64) as usize }
    pub fn gauss(&mut self) -> f64 {
        let mut u = 0.0; let mut v = 0.0;
        while u == 0.0 { u = self.f64(); } while v == 0.0 { v = self.f64(); }
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
    pub fn shuffle<T>(&mut self, a: &mut [T]) {
        let mut i = a.len();
        while i > 1 { i -= 1; let j = self.below(i + 1); a.swap(i, j); }
    }
}
