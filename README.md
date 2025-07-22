# HashJing  
> *The Canon of Every Hash*

[**▶︎ Launch the demo**](https://datasattva.github.io/hashjing-demo/)

**HashJing** is an open-source toolkit and aesthetic inquiry at the point where generative art meets modern cryptography and East Asian symbolism.  
By applying binary logic and ideas drawn from the *Yì Jīng*—better known in the West as the *I Ching* or *Book of Changes* (literally the classical *Canon of Changes*)—it turns any hash into a deterministic visual glyph, revealing the hidden patterns inside cryptographic entropy.

<figure markdown>
<img src="pic/hashjing_mandala.svg" alt="Mandala generated from the SHA-256 hash of the string “HashJing”" width="512"/>
</figure>

*Mandala generated from the SHA-256 hash of “HashJing”*  
`0x2b054d225d025fc24c58030bda276b16089ae21cc8aff93d2751472a98eab06c`

## What it does

HashJing converts cryptographic hash strings—typically **256-bit** (SHA-256) and optionally **160-bit** (Ethereum-address format)—into **mandalas**: circular diagrams composed of radial sectors and concentric rings.  
Each sector maps to one hex character; each ring maps to one of that character’s four bits. Thus a 256-bit hash yields **64 sectors**, while a 160-bit hash yields **40**. The mapping is fully deterministic: **one hash → one form**.

Default hashing uses SHA-256; **Keccak-256** (the Ethereum variant) is available as an option.  
Any 160-bit Ethereum address (wallet or contract) can be visualised as a 40-sector mandala.

### Why it matters

HashJing is more than a hash visualiser; it is a meditation on chance and order.

One and the same 256-bit string of pure entropy is deterministically transformed into a coherent visual sign, a *canon*.  
*Where is the boundary between entropy and canon?* At what moment does raw data become a cultural or symbolic object—inside the SVG function, in the viewer’s perception, or already in the immutability that cryptography confers?

To frame these questions, HashJing leans on two complementary ideas:

* **The visual principles of the *I Ching*** — a generative grammar of change that inspires the mandala’s radial logic.  
* **The Daoist polarity from Wújí → Tàijí** — the leap from the undivided to the first structured difference, echoed here as entropy crystallising into image.

By merging these insights with modern cryptography, HashJing places contemporary data within a long dialogue on pattern, probability and meaning.

### Core ideas

* Binary patterns as a visual language  
* Randomness rendered observable  
* Parallels with the 64 hexagrams of the *Yì Jīng*  
* The hash as a seal—an imprint of choice  
* Complete determinism  

## Quick start

1. Open the demo: <https://datasattva.github.io/hashjing-demo/>  
2. Paste a hash (or type any text).  
3. Click **Generate** to view an SVG mandala.  
4. Download as **SVG** or **PNG**, or explore the *Features of Order* panel:  
   * Balanced / unbalanced bits  
   * Passage count  
   * Rare sealed states  

## Why NFTs?

The upcoming **HashJing NFT drop** – (Q3 2025)

| Spec   | Value |
|--------|-------|
| Supply | 8 192 tokens |
| Price  | 0.002 ETH (fixed) |
| SVG    | Fully on-chain, gas-efficient |
| Traits | `balanced`, `sealed`, `passages`, `creatorInitialReserve` |

**Each HashJing token is a self-contained, on-chain SVG mandala:** all metadata, image data and traits (`balanced`, `passages`, original 32-byte source hash) live permanently inside the Ethereum contract — no IPFS, no external servers.

Full contract code and specification: <https://github.com/DataSattva/hashjing-nft>

## Learn more

| Resource | Link |
|----------|------|
| White Paper – algorithms and maths | [WhitePaper.ipynb](./WhitePaper.ipynb) |
| Art Manifesto – philosophy | [ArtManifesto.md](./ArtManifesto.md) |
| Famous Blockchain Hashes gallery | [FamousBlockchainHashes.ipynb](./FamousBlockchainHashes.ipynb) |
| Developer guide | [CONTRIBUTING.md](./CONTRIBUTING.md) |

## Contacts and resources

The always-up-to-date contact list lives at  
<https://datasattva.github.io/hashjing-res/>

## License

* Code – MIT  
* Visuals and documentation – CC BY-NC 4.0
