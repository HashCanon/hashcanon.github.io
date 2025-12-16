# Contributing to HashCanon

Thank you for your interest in **HashCanon**.  
The core codebase and visuals are complete and maintained by the project author, so no active development roadmap is advertised.  
However, you are welcome to explore, verify, and extend the analytical notebooks that accompany the project.

## Repository layout

```text
├── README.md                    # Project overview (user-facing)
├── CONTRIBUTING.md              # Contributing guide (this file)
├── WhitePaper.ipynb             # Technical deep-dive: mapping, visuals, stats (Balanced, Passages, Symmetries, Crown)
├── FamousBlockchainHashes.ipynb # Re-creatable gallery of landmark blockchain hashes
├── ArtManifesto.md              # Artistic statement
├── LICENSE-MIT.md               # MIT license for source code
├── LICENSE-CCBYNC.md            # CC BY-NC 4.0 for visuals & docs
├── hash_utils/
│   ├── __init__.py              # Package marker
│   └── base_hash.py             # Core hash→mandala logic + metrics
└── pic/
    ├── hashcanon_mandala.svg     # Legacy sample image
    └── yi_circle.jpg            # 64-hexagram I Ching diagram
```


**Почему «legacy»?** Файл реально называется `hashjing_mandala.svg`, это историческое имя артефакта из старого названия проекта. Пометка просто предупреждает читателя; переименовывать сейчас не обязательно.

## 2) Блок “Reproducible notebooks” — исправить опечатку и зависимости

Заменить существующий раздел целиком на:

## Reproducible notebooks

Two Jupyter notebooks can be opened locally or on a cloud service:

| Notebook                                   | Purpose                                                                                                     |
|--------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| [WhitePaper.ipynb](./WhitePaper.ipynb)     | Mapping rules, visual construction, and statistical metrics (Balanced, Passages, Symmetries, Crown).       |
| [FamousBlockchainHashes.ipynb](./FamousBlockchainHashes.ipynb) | Re-create mandalas for landmark blockchain hashes; tweak parameters and add examples.                       |

**Requirements:** Python 3.11+, `matplotlib`, `pandas`, `numpy`.  
Open the notebook and run all cells.

## Community, Support & License

- [Discussions](https://github.com/HashCanon/hashcanon.github.io/discussions)
- [Contacts & resources](https://hashcanon.github.io/resources/)

**License:**  
- Code — [MIT](./LICENSE-MIT.md)  
- Visuals & docs — [CC BY-NC 4.0](./LICENSE-CCBYNC.md)
