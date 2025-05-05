/// Kullback-Leibler and Jensen-Shannon divergence functions.

/// Compute Kullback-Leibler divergence between two discrete probability distributions.
/// Both `p` and `q` must sum to 1 and have the same length.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q.iter())
        .filter(|(pi, qi)| *pi > &0.0 && *qi > &0.0)
        .map(|(pi, qi)| pi * (pi / qi).ln())
        .sum()
}

/// Compute Jensen-Shannon divergence between two discrete probability distributions.
/// Both `p` and `q` must sum to 1 and have the same length.
pub fn js_divergence(p: &[f64], q: &[f64]) -> f64 {
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(pi, qi)| 0.5 * (pi + qi)).collect();
    0.5 * kl_divergence(p, &m) + 0.5 * kl_divergence(q, &m)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kl_js_values() {
        let p = vec![0.1, 0.4, 0.5];
        let q = vec![0.2, 0.3, 0.5];
        let kl = kl_divergence(&p, &q);
        let js = js_divergence(&p, &q);
        assert!(kl >= 0.0);
        assert!(js >= 0.0);
    }
}
