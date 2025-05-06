use rand_distr::{StandardNormal, Distribution};
use nalgebra::DMatrix;

pub fn marchenko_pastur_density(x: f64, q: f64, sigma: f64) -> f64 {
    let lambda_minus = sigma.powi(2) * (1.0 - q.sqrt()).powi(2);
    let lambda_plus = sigma.powi(2) * (1.0 + q.sqrt()).powi(2);
    if x < lambda_minus || x > lambda_plus {
        0.0
    } else {
        (1.0 / (2.0 * std::f64::consts::PI * sigma.powi(2) * q * x))
            * ((lambda_plus - x) * (x - lambda_minus)).sqrt()
    }
}

pub fn wigner_semicircle_density(x: f64, r: f64) -> f64 {
    if x.abs() > r {
        0.0
    } else {
        (2.0 / (std::f64::consts::PI * r.powi(2))) * (r.powi(2) - x.powi(2)).sqrt()
    }
}

pub fn eigenvalue_spacings(matrix: &DMatrix<f64>) -> Vec<f64> {
    let mut eigvals: Vec<f64> = matrix.eigenvalues().unwrap().as_slice().to_vec();
    eigvals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eigvals.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Generate a random symmetric (Wigner) matrix of size n x n.
pub fn random_wigner_matrix(n: usize) -> DMatrix<f64> {
    let mut m = DMatrix::<f64>::zeros(n, n);
    let normal = StandardNormal;
    for i in 0..n {
        for j in i..n {
            let val: f64 = normal.sample(&mut rand::thread_rng());
            m[(i, j)] = val;
            m[(j, i)] = val;
        }
    }
    m
}

/// Generate a random Wishart (sample covariance) matrix of size p x p.
pub fn random_wishart_matrix(p: usize, n: usize) -> DMatrix<f64> {
    let normal = StandardNormal;
    let mut data = DMatrix::<f64>::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            data[(i, j)] = normal.sample(&mut rand::thread_rng());
        }
    }
    &data.transpose() * &data
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_marchenko_pastur_density() {
        let val = marchenko_pastur_density(1.0, 0.5, 1.0);
        assert!(val >= 0.0);
    }

    #[test]
    fn test_wigner_semicircle_density() {
        let val = wigner_semicircle_density(0.0, 2.0);
        assert!(val > 0.0);
    }

    #[test]
    fn test_eigenvalue_spacings() {
        let m = DMatrix::<f64>::identity(3, 3);
        let spacings = eigenvalue_spacings(&m);
        assert_eq!(spacings.len(), 2);
    }
}
