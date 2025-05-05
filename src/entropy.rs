use std::collections::HashMap;

/// Compute the Shannon entropy of a discrete dataset.
pub fn entropy<T: Eq + std::hash::Hash>(data: &[T]) -> f64 {
    let mut freq = HashMap::new();
    for item in data {
        *freq.entry(item).or_insert(0) += 1;
    }
    let len = data.len() as f64;
    freq.values()
        .map(|&count| {
            let p = count as f64 / len;
            -p * p.log2()
        })
        .sum()
}

/// Compute the joint entropy of two discrete datasets.
pub fn joint_entropy<T: Eq + std::hash::Hash, U: Eq + std::hash::Hash>(
    x: &[T],
    y: &[U],
) -> f64 {
    let mut freq = HashMap::new();
    for (xi, yi) in x.iter().zip(y.iter()) {
        *freq.entry((xi, yi)).or_insert(0) += 1;
    }
    let len = x.len() as f64;
    freq.values()
        .map(|&count| {
            let p = count as f64 / len;
            -p * p.log2()
        })
        .sum()
}

/// Compute the mutual information between two discrete datasets.
pub fn mutual_information<T: Eq + std::hash::Hash, U: Eq + std::hash::Hash>(
    x: &[T],
    y: &[U],
) -> f64 {
    entropy(x) + entropy(y) - joint_entropy(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_uniform() {
        let data = vec![1, 2, 3, 4];
        let h = entropy(&data);
        assert!(h > 0.0);
    }

    #[test]
    fn test_joint_entropy() {
        let x = vec![1, 2, 3, 4];
        let y = vec![4, 3, 2, 1];
        let h = joint_entropy(&x, &y);
        assert!(h > 0.0);
    }

    #[test]
    fn test_mutual_information_independent() {
        let x = vec![1, 2, 3, 4];
        let y = vec![4, 3, 2, 1];
        let mi = mutual_information(&x, &y);
        assert!(mi >= 0.0);
    }
}
