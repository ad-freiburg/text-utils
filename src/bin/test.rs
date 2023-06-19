use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Geometric};

fn main() {
    let geo = Geometric::new(0.5).expect("failed to create geometric distribution");
    let mut rng = ChaCha8Rng::from_entropy();

    let sum: u64 = (0..10000).map(|_| geo.sample(&mut rng)).sum();
    println!("sum: {sum}, avg: {}", sum as f64 / 10000.0);
}
