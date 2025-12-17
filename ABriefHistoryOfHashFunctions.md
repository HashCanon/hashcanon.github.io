# A Brief History of Hash Functions

> A hash function is a deterministic transformation of an arbitrary-length message into a fixed-length digest that is hard to invert and hard to collide.

## Why they matter

* Integrity checks and digital signatures
* Content addressing and versioning (e.g., Git)
* Password storage (with salts/derivation)
* Cryptography and blockchains (addresses, PoW/PoS, Merkle trees)

## Security properties (cryptographic hashes)

* **Preimage resistance:** given `h`, infeasible to find `m` with `H(m)=h`
* **Second-preimage resistance:** given `(m, H(m))`, infeasible to find `m'≠m` with the same hash
* **Collision resistance:** infeasible to find `(m₁, m₂)` with `m₁≠m₂` and `H(m₁)=H(m₂)` (≈ `2^{n/2}` for an `n`-bit digest)
* **Avalanche/pseudorandomness:** tiny input change ⇒ large unpredictable output change

## Constructions

* **Merkle–Damgård** (MD4/5, SHA-1/2, RIPEMD): compression function + padding; length-extension exists (use HMAC for MACs)
* **Sponge**: absorb/squeeze over a permutation (Keccak/SHA-3, BLAKE family)

## Timeline (condensed)

* **1990** — **MD4** (Rivest): fast, broken
* **1992** — **MD5**: widely deployed; practical collisions (2004) → **deprecated**
* **1995** — **SHA-1** (160-bit): practical collisions (2017, *SHAttered*) → **deprecated**
* **1996** — **RIPEMD-160**: European line; used in Bitcoin address pipeline (HASH160 = RIPEMD-160(SHA-256(pubkey)))
* **2001** — **SHA-2** (SHA-224/256/384/512): current NIST workhorse, widely trusted
* **2008–2012** — NIST competition; winner **Keccak** → standardized as **SHA-3** (2015), sponge construction
* **2013** — **BLAKE2** (fast, practical variants)
* **2020** — **BLAKE3** (tree-sponge, highly parallel and fast)

## Who uses what

* **Git**: historically SHA-1 → migrating to SHA-256
* **Bitcoin**: block hash = double-SHA-256(header); addresses use HASH160
* **Ethereum**: uses **Keccak-256** (pre-SHA-3 padding variant) for many app-layer tasks (addresses, logs). Note: Keccak-256 ≠ NIST SHA-3-256.

## Relation to HashCanon

HashCanon visualizes **given** 160-/256-bit hashes as circular bit-matrices. The visualization is **hash-algorithm agnostic**: any valid `n`-bit input is supported.
In the web generator, text → **SHA-256** via Web Crypto; for blockchain examples, we use the **published** hashes (e.g., Keccak-256 in Ethereum).
For the empirical distributions and feature methodology see the White Paper:
[WhitePaper.ipynb](https://github.com/HashCanon/hashcanon.github.io/blob/main/WhitePaper.ipynb)
