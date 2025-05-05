pub mod entropy;
pub mod divergence;

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::divergence::{kl_divergence, js_divergence};

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
}
