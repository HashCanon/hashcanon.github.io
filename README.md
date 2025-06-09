# HashJing

**HashJing** is a generative-art project that explores the boundary between randomness and order through the visual language of 256-bit hashes. The work is inspired by binary logic, cryptography, and the symbolism of the *Yì Jīng*—the ancient Chinese *Book of Changes*.

<figure markdown>
<img src="pic/hashjing_mandala.png" alt="Mandala generated from the SHA-256 hash of the project name “HashJing”"/>
</figure>

*Mandala generated from the SHA-256 hash of the project name “HashJing”:*
`0x2b054d225d025fc24c58030bda276b16089ae21cc8aff93d2751472a98eab06c`

## What it does

HashJing converts 256-bit strings—for example, Ethereum or Bitcoin hashes, or any cryptographically secure seed—into **mandalas**: circular diagrams with 64 sectors and four concentric rings.
Each sector represents one hex character; each ring, one of that character’s four bits. The visualisation therefore encodes the entire hash, deterministically and unambiguously.

**Address support.** HashJing can also visualise 160-bit Ethereum addresses (wallets or contracts); an address appears as a mandala with 40 sectors.

A mandala becomes a frame for contemplation—of entropy, symmetry, rarity, and the visual face of probability.

## Core ideas

* Binary patterns as a visual language
* Randomness as an observable structure
* Parallels with the 64 hexagrams of the *Yì Jīng*
* The hash as a seal—an imprint of choice
* Complete determinism: one hash → one form

## Project layout

```text
├── index.html                 # Demo page (HTML + Canvas)
├── ArtManifesto.md            # Artistic manifesto
├── README.md                  # You are here
├── WhitePaper.ipynb           # Technical white-paper (Jupyter Notebook)
├── FamousHashes.ipynb         # Visualisation of famous blockchain hashes
├── hash_utils/                # Core Python module
│   ├── __init__.py
│   └── base_hash.py           # Hex→binary conversion, drawing logic
├── pic/                       # Static images & reference diagrams
│   ├── hashjing_mandala.png   # Example mandala
│   └── yi_circle.jpg          # Historical *Yì Jīng* arrangement
```

## Further reading

* [**ArtManifesto.md**](https://github.com/DataSattva/hashjing/blob/main/ArtManifesto.md) – the philosophical core.
  Explains HashJing as a way to embody the “shape of randomness,” fusing cryptography, the *Yì Jīng*, and digital aesthetics. Here the mandala is not ornament but the geometry of entropy.

* [**WhitePaper.ipynb**](https://github.com/DataSattva/hashjing/blob/main/WhitePaper.ipynb) – the technical paper.  
  Details the hash structure, the visual-mapping rules (including the 160-bit address variant), and demonstrates the main drawing function `draw_mandala`.  
  It also introduces higher-level *features of order* such as passage count and bit balance, and explores their statistical distribution in randomly generated hashes.

* [**FamousHashes.ipynb**](https://github.com/DataSattva/hashjing/blob/main/FamousHashes.ipynb) – mandalas of iconic hashes.
  Shows the Genesis block of Bitcoin, the CryptoPunks contract address, and other historically or culturally significant hashes, illustrating how an abstract string becomes a visual signature.

## Demo page

`index.html` is a simple HTML/Canvas page. Each time it loads it generates a random 256-bit hash and draws its mandala automatically.

**No user input required**—just reload the page to see a new mandala.
Perfect for demonstrating deterministic randomness and testing the visual rhythm of hashes.

[**Open the HashJing demo**](https://datasattva.github.io/hashjing/)

You can also supply your own hash value; see the example code in [WhitePaper.ipynb](https://github.com/DataSattva/hashjing/blob/main/WhitePaper.ipynb).

## License

- All source code in this repository is released under the [MIT License](https://github.com/DataSattva/hashjing/blob/main/LICENSE-MIT.md).
- All visual outputs, including SVG mandalas and image-based documentation, are licensed under [CC BY-NC 4.0](https://github.com/DataSattva/hashjing/blob/main/LICENSE-CCBYNC.md).

You're free to use, remix, and redistribute the project — including generated art — for **non-commercial purposes**, with proper attribution.



