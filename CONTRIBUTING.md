# Contributing to HashCanon

Thank you for your interest in **HashCanon**.

The core mapping, visuals, and generator are maintained by the project author, so there is no public “feature roadmap” or open call for large refactors. However, you are very welcome to:

- explore and verify the analytical notebooks,
- reproduce the metrics (Balanced, Passages, Symmetries, Crown),
- extend the analysis locally for your own experiments.

If you discover a clear bug, inconsistency, or a mistake in the spec/notebooks, you can open a discussion on GitHub.

---

## Repository layout (high-level)

```text
├── README.md                    # Project overview (user-facing)
├── CONTRIBUTING.md              # This contributing guide
├── HashCanonSpec.ipynb          # Mapping rules, visuals, Balanced / Passages / Symmetries / Crown
├── FamousBlockchainHashes.ipynb # Re-creatable gallery of landmark blockchain hashes
├── ArtManifesto.md              # Artistic statement
├── LICENSE-MIT.md               # MIT license for source code and textual docs
├── LICENSE-CCBYNC.md            # CC BY-NC 4.0 for visual artworks (mandalas)
├── hash_utils/
│   └── base_hash.py             # Python helpers for hashes, traits and symmetry analysis
└── pic/                         # Selected visual outputs (mandalas and related images)
````

---

## Working with notebooks

The two main notebooks are:

| Notebook                       | Purpose                                                                 |
| ------------------------------ | ----------------------------------------------------------------------- |
| `HashCanonSpec.ipynb`          | Canon mapping rules, geometry, feature definitions and basic statistics |
| `FamousBlockchainHashes.ipynb` | Examples and “exhibits”: well-known blockchain hashes and contracts     |

**Requirements (typical):**

* Python 3.11+
* `matplotlib`
* `numpy`
* `pandas`
* `pycryptodome` (for Keccak, if you use it in local experiments)

You can install them into a virtual environment, for example:

```bash
pip install numpy pandas matplotlib pycryptodome
```

Then open the notebook in Jupyter / JupyterLab and run all cells top-to-bottom.

---

## `hash_utils/base_hash.py` (analysis helpers)

The module `hash_utils/base_hash.py` contains the Python helpers used in the notebooks, including:

* hash generation utilities (random hashes, text → hash),
* evenness/bit-balance helpers,
* passage counting in the 4×N circular grid,
* symmetry analysis:

  * `find_symmetries(...)`
  * `symmetry_ranks(...)`
  * `symmetry_metric(...)`
  * `crown_metric(...)`
  * `crown_slices(...)`
  * `symmetry_overlay_segments(...)`
* `draw_mandala(...)` — reference renderer for circular bit-matrices with optional symmetry overlay.

A few ground rules if you touch this file locally for experiments:

* Keep function signatures stable if you reuse the notebooks.
* Do not change the semantics of the **canon** (mapping from bits to rings/sectors).
* If you extend functionality, prefer adding separate helpers instead of changing the core ones in incompatible ways.

At the moment there is no guarantee that external PRs to this module will be reviewed or merged; treat it primarily as a reference implementation for your own analysis.

---

## Reporting issues or inconsistencies

If you notice:

* a mismatch between the spec (`HashCanonSpec.ipynb`) and the implementation,
* a reproducible bug in trait calculations (Balanced, Passages, Symmetries, Crown),
* or a clear mistake in the textual docs,

you can open a GitHub **Discussion** and describe:

1. which file/notebook you are referring to,
2. the exact input (hash, address, or block),
3. the expected vs. observed behaviour.

Link: [https://github.com/HashCanon/hashcanon.github.io/discussions](https://github.com/HashCanon/hashcanon.github.io/discussions)

---

## Community, support & links

* Discussions: [https://github.com/HashCanon/hashcanon.github.io/discussions](https://github.com/HashCanon/hashcanon.github.io/discussions)
* Project resources: [https://hashcanon.github.io/resources/](https://hashcanon.github.io/resources/)

---

## License & attribution

* **Source code and textual documentation** — [MIT](./LICENSE-MIT.md)
* **Visual artworks (mandalas and other visual outputs)** — [CC BY-NC 4.0](./LICENSE-CCBYNC.md)

If you reference HashCanon in research, posts, or derivative analysis, please attribute the project name (**HashCanon**) and the author (**DataSattva**) and link to the main repo or the resources page.
