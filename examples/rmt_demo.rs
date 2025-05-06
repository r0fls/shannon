use shannon::Shannon;

fn main() {
    let s = Shannon::new();

    // Example: Marchenko-Pastur density
    let mp = s.marchenko_pastur(1.0, 0.5, 1.0);
    println!("Marchenko-Pastur density at x=1.0: {}", mp);

    // Example: Wigner semicircle density
    let ws = s.wigner_semicircle(0.0, 2.0);
    println!("Wigner semicircle density at x=0.0: {}", ws);

    // Example: Generate random Wigner matrix and compute eigenvalue spacings
    let wigner = s.random_wigner(5);
    println!("Random Wigner matrix:\n{}", wigner);
    let spacings = s.eigen_spacings(&wigner);
    println!("Eigenvalue spacings: {:?}", spacings);

    // Example: Generate random Wishart matrix
    let wishart = s.random_wishart(5, 10);
    println!("Random Wishart matrix:\n{}", wishart);
}
