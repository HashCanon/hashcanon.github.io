# Contributing to HashJing

Thank you for your interest in **HashJing**.  
The core codebase and visuals are complete and maintained by the project author, so no active development roadmap is advertised.  
However, you are welcome to explore, verify, and extend the analytical notebooks that accompany the project.

## Repository layout

```text
├── README.md                    # Project overview (user-facing)
├── CONTRIBUTING.md              # ← you are here
├── WhitePaper.ipynb             # Technical deep-dive and calculations
├── ArtManifesto.md              # Artistic statement
├── FamousBlockchainHashes.ipynb # Re-creatable gallery of landmark hashes
├── index.html                   # HashJing Mandala Demo (live UI)
├── LICENSE-MIT.md               # MIT license for source code
├── LICENSE-CCBYNC.md            # CC BY-NC 4.0 for visuals & docs
├── hash_utils/
│   └── base_hash.py             # Core hash-to-mandala logic
└── pic/
    └── hashjing_mandala.svg     # Sample image
```

## Reproducible notebooks

Two Jupyter notebooks can be opened locally or on any cloud service (e.g. Google Colab):

| Notebook                         | Purpose                                                                                                                                                      |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **[WhitePaper.ipyn](./WhitePaper.ipynb)**             | Derives the mapping rules, produces statistical tables, and walks through the sealed / balanced / passages metrics.                                          |
| **[FamousBlockchainHashes.ipynb](./FamousBlockchainHashes.ipynb)** | Regenerates mandalas for Genesis blocks, landmark contracts, and other culturally significant hashes. Add your own examples or tweak parameters as you wish. |

Both notebooks rely only on the Python Standard Library plus `matplotlib`.
Simply open, run all cells, and experiment.

## License

Contributions are accepted under the same dual license as the project:

* **Code** — MIT
* **Visuals & documentation** — CC BY-NC 4.0

By submitting content, you agree that it will be released under these terms.

