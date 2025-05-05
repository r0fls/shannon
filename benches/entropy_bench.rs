#![feature(test)]
extern crate test;

use test::Bencher;
use shannon::entropy::*;

#[bench]
fn bench_entropy(b: &mut Bencher) {
    let data: Vec<u32> = (0..1000).map(|i| i % 10).collect();
    b.iter(|| entropy(&data));
}

#[bench]
fn bench_joint_entropy(b: &mut Bencher) {
    let x: Vec<u32> = (0..1000).map(|i| i % 10).collect();
    let y: Vec<u32> = (0..1000).map(|i| (i + 5) % 10).collect();
    b.iter(|| joint_entropy(&x, &y));
}

#[bench]
fn bench_mutual_information(b: &mut Bencher) {
    let x: Vec<u32> = (0..1000).map(|i| i % 10).collect();
    let y: Vec<u32> = (0..1000).map(|i| (i + 5) % 10).collect();
    b.iter(|| mutual_information(&x, &y));
}
