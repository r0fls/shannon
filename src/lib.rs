pub mod entropy;
pub mod divergence;
pub mod rmt;

use nalgebra::DMatrix;

/// Main public API for Shannon crate.
pub struct Shannon;

impl Shannon {
    /// Create a new Shannon instance.
    pub fn new() -> Self {
        Shannon
    }

    /// Compute Kullback-Leibler divergence.
    pub fn kl_divergence(&self, p: &[f64], q: &[f64]) -> f64 {
        divergence::kl_divergence(p, q)
    }

    /// Compute Jensen-Shannon divergence.
    pub fn js_divergence(&self, p: &[f64], q: &[f64]) -> f64 {
        divergence::js_divergence(p, q)
    }

    /// Compute Marchenko-Pastur density.
    pub fn marchenko_pastur(&self, x: f64, q: f64, sigma: f64) -> f64 {
        rmt::marchenko_pastur_density(x, q, sigma)
    }

    /// Compute Wigner semicircle density.
    pub fn wigner_semicircle(&self, x: f64, r: f64) -> f64 {
        rmt::wigner_semicircle_density(x, r)
    }

    /// Compute eigenvalue spacings.
    pub fn eigen_spacings(&self, matrix: &nalgebra::DMatrix<f64>) -> Vec<f64> {
        rmt::eigenvalue_spacings(matrix)
    }

    pub fn random_wigner(&self, n: usize) -> DMatrix<f64> {
        rmt::random_wigner_matrix(n)
    }

    pub fn random_wishart(&self, p: usize, n: usize) -> DMatrix<f64> {
        rmt::random_wishart_matrix(p, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::divergence::{kl_divergence, js_divergence};
    use nalgebra::DMatrix;

    #[test]
    fn test_kl_js_divergence() {
        let p = vec![0.1, 0.4, 0.5];
        let q = vec![0.2, 0.3, 0.5];

        let kl = kl_divergence(&p, &q);
        let js = js_divergence(&p, &q);

        assert!(kl >= 0.0);
        assert!(js >= 0.0);
    }

    #[test]
    fn test_shannon_api() {
        let s = Shannon::new();
        let p = vec![0.1, 0.4, 0.5];
        let q = vec![0.2, 0.3, 0.5];

        let kl = s.kl_divergence(&p, &q);
        let js = s.js_divergence(&p, &q);

        assert!(kl >= 0.0);
        assert!(js >= 0.0);
    }

    #[test]
    fn test_rmt_api() {
        let s = Shannon::new();
        let mp = s.marchenko_pastur(1.0, 0.5, 1.0);
        let ws = s.wigner_semicircle(0.0, 2.0);
        let matrix = DMatrix::<f64>::identity(3, 3);
        let spacings = s.eigen_spacings(&matrix);

        assert!(mp >= 0.0);
        assert!(ws >= 0.0);
        assert_eq!(spacings.len(), 2);
    }
    #[test]
    fn test_random_matrices() {
        let s = Shannon::new();
        let wigner = s.random_wigner(5);
        let wishart = s.random_wishart(3, 5);
        assert_eq!(wigner.shape(), (5, 5));
        assert_eq!(wishart.shape(), (3, 3));
    }
}
