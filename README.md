# HashJing

**HashJing** is an open-source toolkit and aesthetic inquiry that sits where generative art meets modern cryptography and East-Asian symbolism.  
By applying binary logic and ideas drawn from the *Yì Jīng*—the ancient *Book of Changes*—it turns any hash into a deterministic visual glyph, revealing the hidden patterns inside cryptographic entropy.


<figure markdown>
<img src="pic/hashjing_mandala.svg" alt="Mandala generated from the SHA‑256 hash of the string “HashJing”"/>
</figure>

*Mandala generated from the SHA‑256 hash of “HashJing”*  
`0x2b054d225d025fc24c58030bda276b16089ae21cc8aff93d2751472a98eab06c`

## What it does

HashJing converts cryptographic hash strings—typically **256‑bit** (SHA‑256) and optionally **160‑bit** (Ethereum‑address format)—into **mandalas**: circular diagrams composed of radial sectors and concentric rings.  
Each sector represents one hex character; each ring represents one of that character’s four bits. Thus a 256‑bit hash yields **64 sectors**, while a 160‑bit hash yields **40**. The mapping is fully deterministic: **one hash → one form**.

Default hashing uses SHA‑256; **Keccak‑256** (the Ethereum variant) is available as an option.

**Address support** – any 160‑bit Ethereum address (wallet or contract) can be visualised as a 40‑sector mandala.

A mandala becomes a frame for contemplating entropy, symmetry, rarity, and the visible face of probability.

## Core ideas

* Binary patterns as a visual language  
* Randomness rendered observable  
* Parallels with the 64 hexagrams of the *Yì Jīng*  
* The hash as a seal—an imprint of choice  
* Complete determinism

## Project layout

```text
├── README.md                    # 1. Entry point
├── WhitePaper.ipynb             # 2. Technical deep-dive
├── ArtManifesto.md              # 3. Artistic statement
├── FamousBlockchainHashes.ipynb # 4. Gallery of landmark hashes
├── index.html                   # 5. HashJing Mandala Demo (live UI)
├── LICENSE-MIT.md               # MIT license for source code
├── LICENSE-CCBYNC.md            # CC BY-NC 4.0 for visuals & docs
├── hash_utils/
│   ├── __init__.py
│   └── base_hash.py             # Hash analysis & mandala construction logic
└── pic/
    ├── hashjing_mandala.svg     # Sample mandala
    └── yi_circle.jpg            # 64-hexagram I Ching diagram
```

## Further reading

* [**ArtManifesto.md**](https://github.com/DataSattva/hashjing/blob/main/ArtManifesto.md) – philosophical background; HashJing as “the shape of randomness.”

* [**WhitePaper.ipynb**](https://github.com/DataSattva/hashjing/blob/main/WhitePaper.ipynb) – technical details, mapping rules, statistical features.

* [**FamousBlockchainHashes.ipynb**](https://github.com/DataSattva/hashjing/blob/main/FamousBlockchainHashes.ipynb) – mandalas of Genesis blocks, landmark contracts, and other culturally significant hashes.

## HashJing Mandala Demo

`index.html` is an interactive viewer and generator for **HashJing** mandalas.  
It lets you explore the visual logic of cryptographic hashes in three ways:

* **Text → hash** Enter any string, get its SHA-256 hash, and see the mandala.  
* **Custom hash** Paste any 256- or 160-bit hexadecimal value (e.g. an Ethereum address).  
* **Random hash** Leave the fields blank and click **Generate** to create a fresh mandala.

Beyond rendering the SVG mandala, the demo also analyses structural features:

* **Bit balance** — whether the hash has an equal number of 0 s and 1 s  
* **Sealedness** — does any path connect the centre to the edge?  
* **Passages** — count of black corridors from core to outer ring  

This tool is ideal for studying how entropy crystallises into form.

**[→ Launch the viewer](https://datasattva.github.io/hashjing-demo/)**  
**[→ See WhitePaper.ipynb for technical details](https://github.com/DataSattva/hashjing/blob/main/WhitePaper.ipynb)**

## License

- Code: [MIT License](https://github.com/DataSattva/hashjing/blob/main/LICENSE-MIT.md).
- Visual outputs and documentation: [CC BY-NC 4.0](https://github.com/DataSattva/hashjing/blob/main/LICENSE-CCBYNC.md).

You're free to use, remix, and redistribute the project — including generated art — for **non-commercial purposes**, with proper attribution.



