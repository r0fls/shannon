# Shannon

A Rust library for information theory and random matrix theory, inspired by Claude Shannon.

---

## ✨ Features

✅ Entropy and mutual information
✅ Kullback-Leibler (KL) and Jensen-Shannon (JS) divergence
✅ Random matrix theory:
- Marchenko-Pastur law
- Wigner semicircle law
- Eigenvalue spacings
- Random Wigner and Wishart matrix generation

---

## 📦 Install

```bash
cargo add shannon
```

## 📖 Usage

To run the examples:

```bash
cargo run --example rmt_demo
```

### 📜 Example

```rust
use shannon::Shannon;

fn main() {
    let s = Shannon::new();

    let mp = s.marchenko_pastur(1.0, 0.5, 1.0);
    let ws = s.wigner_semicircle(0.0, 2.0);

    println!("Marchenko-Pastur density: {}", mp);
    println!("Wigner semicircle density: {}", ws);
}
```
