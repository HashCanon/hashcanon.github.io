# HashJing  
> *The Canon of every Hash*

**HashJing** is an open-source toolkit and aesthetic inquiry that sits where generative art meets modern cryptography and East-Asian symbolism.  
By applying binary logic and ideas drawn from the *Yì Jīng*—the ancient *Book of Changes*—it turns any hash into a deterministic visual glyph, revealing the hidden patterns inside cryptographic entropy.

<figure markdown>
<img src="pic/hashjing_mandala.svg" alt="Mandala generated from the SHA-256 hash of the string “HashJing”" width="512"/>
</figure>

*Mandala generated from the SHA-256 hash of “HashJing”*  
`0x2b054d225d025fc24c58030bda276b16089ae21cc8aff93d2751472a98eab06c`

## What it does

HashJing converts cryptographic hash strings—typically **256-bit** (SHA-256) and optionally **160-bit** (Ethereum-address format)—into **mandalas**: circular diagrams composed of radial sectors and concentric rings.  
Each sector represents one hex character; each ring represents one of that character’s four bits. Thus a 256-bit hash yields **64 sectors**, while a 160-bit hash yields **40**. The mapping is fully deterministic: **one hash → one form**.

Default hashing uses SHA-256; **Keccak-256** (the Ethereum variant) is available as an option.  
Any 160-bit Ethereum address (wallet or contract) can be visualised as a 40-sector mandala.

A mandala becomes a frame for contemplating entropy, symmetry, rarity, and the visible face of probability.

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

The upcoming **HashJing NFT drop** (Q3 2025)

| Spec    | Value |
|---------|-------|
| Supply  | 8 192 tokens |
| Price   | 0.002 ETH (fixed) |
| SVG     | Fully on-chain, gas-efficient |
| Traits  | `balanced`, `sealed`, `passages`, `creatorInitialReserve` |

Each HashJing token is a self-contained, on-chain SVG mandala.  
All metadata, image data, and traits (`balanced`, `passages`, original 32-byte source hash) live permanently inside the Ethereum contract—no IPFS, no external servers.  
Full contract code and specification: <https://github.com/DataSattva/hashjing-nft>



## Learn more

| Resource | Link |
|----------|------|
| White Paper – algorithms and maths | [WhitePaper.ipyn](./WhitePaper.ipynb) |
| Art Manifesto – philosophy | [Art Manifesto](./ArtManifesto.md)  |
| Famous Blockchain Hashes gallery | [FamousBlockchainHashes.ipynb](./FamousBlockchainHashes.ipynb) |
| Developer setup and contribution guide | [CONTRIBUTING.md](./CONTRIBUTING.md) |


---

## Contacts and Resources

For a detailed list of HashJing contacts and resources, see the page [Contacts and Resources](https://datasattva.github.io/hashjing-res/)

---

## License

* Code – MIT  
* Visuals and documentation – CC BY-NC 4.0
