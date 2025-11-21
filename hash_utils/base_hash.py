import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random, string, hashlib, secrets
from decimal import Decimal
from collections import deque
from Crypto.Hash import keccak

__all__ = [
    "generate_random_hash",
    "explain_hex_to_bin",
    "draw_binary_grid_from_hex_dark",
    "draw_mandala",
    "hash_to_hex",
    "generate_balanced_hash",
    "bit_ratio",
    "is_balanced",
    "count_unique_passages",
    "generate_hash_with_passages",
    "passage_distribution",
    "generate_hash_dataframe"
]


def generate_random_hash(bits: int = 256, algo: str = 'sha256') -> str:
    """
    Generates a cryptographically secure hex string of the specified bit length.
    Supports SHA-256 and Keccak-based entropy sources (using pycryptodome).

    Args:
        bits (int): Bit length. Must be 160 or 256.
        algo (str): Hash algorithm for entropy. Options: 'sha256' (default), 'keccak'.

    Returns:
        str: Hex string with '0x' prefix.
    """
    if bits not in (160, 256):
        raise ValueError("Only 160 or 256 bits supported.")

    if algo == 'keccak':
        entropy = secrets.token_bytes(32)
        k = keccak.new(digest_bits=256)
        k.update(entropy)
        digest = k.digest()
        return '0x' + digest[:bits // 8].hex()

    elif algo == 'sha256':
        return '0x' + secrets.token_hex(bits // 8)

    else:
        raise ValueError("Unsupported algorithm. Use 'sha256' or 'keccak'.")

    
def explain_hex_to_bin(hex_str: str) -> None:
    """
    Prints a per-character view of a 64-character hex string as 4-bit binary chunks.

    Args:
        hex_str (str): A hex string of 64 characters (may start with '0x').

    Example:
        >>> explain_hex_to_bin("0x1a2b3c...")
        00: 1 -> 0001
        01: A -> 1010
        ...
    """
    if hex_str.startswith("0x"):
        hex_str = hex_str[2:]

    assert len(hex_str) == 64, "Expected 64 hex characters (256 bits)"

    for i, char in enumerate(hex_str):
        binary = format(int(char, 16), '04b')
        print(f"{i:02d}: {char.upper()} -> {binary}")


def draw_binary_grid_from_hex(hex_str: str, show_guides: bool = True) -> None:
    """
    Draws a 4×64 binary matrix based on a hex string.

    Args:
        hex_str (str): A 64-character hex string (may start with '0x').
        show_guides (bool): If True, draws red dividers every 16 columns.
    """
    if hex_str.startswith("0x"):
        hex_str = hex_str[2:]

    assert len(hex_str) == 64, "Expected 64 hex characters"

    grid = np.zeros((4, 64), dtype=int)

    for i, char in enumerate(hex_str):
        bin_str = format(int(char, 16), '04b')
        for j, bit in enumerate(bin_str):
            grid[j, i] = 1 - int(bit)

    plt.figure(figsize=(16, 2))
    plt.imshow(grid, cmap="gray_r", interpolation="nearest")

    if show_guides:
        for x in [16, 32, 48]:
            plt.axvline(x=x - 0.5, color='red', linewidth=2)

    plt.xticks([])
    plt.yticks([])
    plt.ylim(-0.5, 3.5)
    plt.xlim(-0.5, 63.5)
    plt.tight_layout()
    plt.show()


def draw_binary_grid_from_hex_dark(hex_str: str, show_guides: bool = True) -> None:
    """
    Draws a 4×64 binary matrix based on a hex string using a dark background.

    Args:
        hex_str (str): A 64-character hex string (may start with '0x').
        show_guides (bool): If True, draws red vertical dividers.
    """
    if hex_str.startswith("0x"):
        hex_str = hex_str[2:]

    assert len(hex_str) == 64, "Expected 64 hex characters"

    grid = np.zeros((4, 64), dtype=int)

    for i, char in enumerate(hex_str):
        bin_str = format(int(char, 16), '04b')
        for j, bit in enumerate(bin_str):
            grid[j, i] = 1 - int(bit)  # 0 → black, 1 → white (inverted)

    fig, ax = plt.subplots(figsize=(16, 2))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    ax.imshow(grid, cmap="gray_r", interpolation="nearest")

    if show_guides:
        for x in [16, 32, 48]:
            ax.axvline(x=x - 0.5, color='red', linewidth=2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, 63.5)
    ax.set_ylim(-0.5, 3.5)
    plt.tight_layout()
    plt.show()


def pad_hex_to_64(hex_string: str) -> str:
    """
    Pads a hex string with trailing zeroes to 64 characters and adds '0x'.

    Args:
        hex_string (str): Hex string (with or without '0x').

    Returns:
        str: A 64-character hex string with '0x' prefix.
    """
    clean = hex_string.lower().replace("0x", "")
    padded = clean.ljust(64, "0")[:64]
    return "0x" + padded


def draw_mandala(hex_string, inner_radius=0.32, show_radial_line=False, sectors=64):
    """
    Draws a circular mandala of bits using hex input and specified number of sectors.

    Args:
        hex_string (str): Full hex string (with or without '0x').
        inner_radius (float): Radius of the central black circle.
        show_radial_line (bool): If True, draws red line at top (12 o'clock).
        sectors (int): Number of sectors (default 64).
    """
    rings = 4
    angle_step = 2 * np.pi / sectors
    radius_step = 0.15
    base_radius = inner_radius
    rotation_offset = angle_step / 2

    total_bits = sectors * rings
    bit_array = hex_to_bit_array(hex_string)[:total_bits]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Center black circle
    ax.bar(
        x=0,
        height=base_radius,
        width=2*np.pi,
        bottom=0,
        color='black',
        edgecolor='black',
        linewidth=0
    )

    # Center hash text, split into 4 lines
    clean_hex = hex_string[2:] if hex_string.startswith("0x") else hex_string
    line_len = len(clean_hex) // 4
    square = [clean_hex[i:i + line_len] for i in range(0, len(clean_hex), line_len)]
    for i, line in enumerate(square):
        fig.text(0.51, 0.535 - i * 0.026, line,
                 ha='center', va='center',
                 color='white', fontsize=8, family='monospace')

    # Sector rendering
    for i in range(sectors):
        bin_segment = bit_array[i * 4:(i + 1) * 4][::-1]
        for j in range(rings):
            if bin_segment[j] == 1:
                theta = i * angle_step + rotation_offset
                r_inner = (3 - j) * radius_step + base_radius
                ax.bar(
                    x=theta,
                    height=radius_step,
                    width=angle_step,
                    bottom=r_inner,
                    color='white',
                    edgecolor='black',
                    linewidth=0.5
                )

    if show_radial_line:
        theta_line = 0
        r_max = base_radius + rings * radius_step
        ax.plot([theta_line, theta_line], [base_radius, r_max], color='red', linewidth=2)

    plt.show()


def hash_to_hex(word: str, bits: int = 256, algo: str = 'sha256') -> str:
    """
    Returns a hash of a string as a hex string with '0x' prefix.
    Supports SHA-256 and Keccak (Ethereum), and both 256-bit and 160-bit lengths.

    Args:
        word (str): Input string.
        bits (int): Desired bit length (160 or 256).
        algo (str): Hash algorithm to use: 'sha256' (default) or 'keccak'.

    Returns:
        str: Hex-encoded hash string with '0x' prefix.
    """
    if bits not in (160, 256):
        raise ValueError("Only 160 and 256 bits supported.")

    if algo == 'keccak':
        k = keccak.new(digest_bits=256)
        k.update(word.encode())
        full_hash = k.digest()
    elif algo == 'sha256':
        full_hash = hashlib.sha256(word.encode()).digest()
    else:
        raise ValueError("Unsupported algorithm. Use 'sha256' or 'keccak'.")

    if bits == 160:
        full_hash = full_hash[-20:]

    return '0x' + full_hash.hex()


# === Features of Order ===

def generate_balanced_hash(bits: int = 256) -> str:
    """
    Generates a balanced hex string (equal number of 0 and 1 bits) of specified bit length.
    Supports 160-bit (e.g., Ethereum addresses) and 256-bit hashes.

    Args:
        bits (int): Total number of bits. Must be even. Common values: 160 or 256.

    Returns:
        str: A balanced hex string with '0x' prefix.
    """
    if bits % 2 != 0:
        raise ValueError("Bit length must be even for balanced generation.")
    if bits not in (160, 256):
        raise ValueError("Only 160-bit and 256-bit supported for now.")

    from secrets import SystemRandom
    random_gen = SystemRandom()
    bit_list = [0] * (bits // 2) + [1] * (bits // 2)
    random_gen.shuffle(bit_list)

    hex_string = ''
    for i in range(0, bits, 4):
        nibble = bit_list[i:i+4]
        hex_digit = hex(int(''.join(map(str, nibble)), 2))[2:]
        hex_string += hex_digit

    return '0x' + hex_string


def bit_ratio(hex_string: str, invert: bool = False, digits: int = 2) -> str:
    """
    Computes the ratio of zeros to ones (or vice versa) for any-length hex string.

    Args:
        hex_string (str): Hex string with '0x' prefix.
        invert (bool): If True, computes 1/0 instead of 0/1.
        digits (int): Decimal precision.

    Returns:
        str: Ratio (e.g., '1.00'), or 'Infinity' if denominator is 0.
    """
    bin_str = bin(int(hex_string[2:], 16))[2:]
    total_bits = len(bin_str)
    padded = bin_str.zfill((len(hex_string) - 2) * 4)  # pad up to nibble count

    ones = padded.count('1')
    zeros = padded.count('0')
    numerator, denominator = (ones, zeros) if invert else (zeros, ones)

    if denominator == 0:
        return 'Infinity'
    else:
        ratio = Decimal(numerator) / Decimal(denominator)
        return f"{ratio:.{digits}f}"


def is_balanced(hex_string: str) -> bool:
    """
    Checks whether the given hex string has an equal number of 0 and 1 bits.

    Args:
        hex_string (str): Hex string with '0x' prefix.

    Returns:
        bool: True if number of 0s == number of 1s, else False.
    """
    bin_str = bin(int(hex_string[2:], 16))[2:]
    padded = bin_str.zfill((len(hex_string) - 2) * 4)
    return padded.count('0') == padded.count('1')


def hex_to_bit_array(hex_string: str) -> list[int]:
    """
    Converts a hex string to a flat list of bits (MSB-first).

    Args:
        hex_string (str): Hex string with '0x' prefix.

    Returns:
        list[int]: Bit list of length 256 or 160, depending on input.
    """
    if hex_string.startswith('0x'):
        hex_string = hex_string[2:]
    return [int(bit) for h in hex_string for bit in bin(int(h, 16))[2:].zfill(4)]


def hex_to_grid(hex_string: str) -> list[list[int]]:
    """
    Converts hex string to a grid representation: [ring][sector].

    - For 256-bit hashes: 4 rings × 64 sectors.
    - For 160-bit hashes: 4 rings × 40 sectors.

    Args:
        hex_string (str): Hex string with '0x' prefix.

    Returns:
        list[list[int]]: Bit grid as [ring][sector].
    """
    bits = hex_to_bit_array(hex_string)
    bit_len = len(bits)
    if bit_len == 256:
        sectors, rings = 64, 4
    elif bit_len == 160:
        sectors, rings = 40, 4
    else:
        raise ValueError("Unsupported bit length. Must be 160 or 256.")

    grid = [[0] * sectors for _ in range(rings)]
    for sector in range(sectors):
        for ring in range(rings):
            bit_index = sector * rings + ring
            grid[ring][sector] = bits[bit_index]
    return grid


def count_unique_passages(hex_string: str) -> int:
    """
    Counts how many unique radial passages exist in the mandala.
    A passage is a connected path of zeros from the center (ring 0)
    to the edge (ring N), using 4-directional adjacency.

    Args:
        hex_string (str): 160- or 256-bit hex string with '0x' prefix.

    Returns:
        int: Number of distinct passages from center to edge.
    """
    grid = hex_to_grid(hex_string)
    rings = len(grid)
    sectors = len(grid[0])
    global_visited = [[False] * sectors for _ in range(rings)]
    passage_count = 0

    for start_sector in range(sectors):
        if grid[0][start_sector] != 0 or global_visited[0][start_sector]:
            continue

        queue = deque()
        local_visited = [[False] * sectors for _ in range(rings)]
        queue.append((0, start_sector))

        reached_edge = False
        path_cells = []

        while queue:
            r, s = queue.popleft()
            if local_visited[r][s] or grid[r][s] != 0:
                continue

            local_visited[r][s] = True
            path_cells.append((r, s))

            if r == rings - 1:
                reached_edge = True

            neighbors = [
                (r + 1, s),                # outward
                (r - 1, s),                # inward
                (r, (s + 1) % sectors),    # clockwise
                (r, (s - 1) % sectors),    # counterclockwise
            ]

            for nr, ns in neighbors:
                if 0 <= nr < rings and not local_visited[nr][ns] and not global_visited[nr][ns] and grid[nr][ns] == 0:
                    queue.append((nr, ns))

        if reached_edge:
            passage_count += 1
            for r, s in path_cells:
                global_visited[r][s] = True

    return passage_count


def generate_hash_with_passages(target_passages: int, bits: int = 256, max_attempts: int = 10000) -> str:
    """
    Attempts to generate a hash with the specified number of radial passages.

    Args:
        target_passages (int): Desired number of passages.
        bits (int): Bit length: 160 or 256.
        max_attempts (int): Maximum attempts before giving up.

    Returns:
        str: Hash with '0x' prefix matching the passage count.

    Raises:
        ValueError: If no matching hash is found.
    """
    for _ in range(max_attempts):
        candidate = generate_random_hash(bits)
        if count_unique_passages(candidate) == target_passages:
            return candidate
    raise ValueError(f"No {bits}-bit hash with {target_passages} passages found in {max_attempts} attempts.")


def passage_distribution(sample_size: int = 10000, bits: int = 256) -> list[int]:
    """
    Generates a sample of random hashes and computes the number of radial passages
    in the corresponding mandala for each hash.

    Args:
        sample_size (int): Number of random hashes to generate.
        bits (int): Bit length of hashes to generate (commonly 160 or 256).

    Returns:
        list[int]: List of passage counts, one per generated hash.
    """
    results = []
    for _ in range(sample_size):
        h = generate_random_hash(bits=bits)
        p = count_unique_passages(h)
        results.append(p)
    return results


def generate_hash_dataframe(n: int = 10000, bits: int = 256) -> pd.DataFrame:
    """
    Generates a DataFrame with random hashes and their associated features:
    - Whether the hash is balanced (equal number of 0s and 1s)
    - How many radial passages the mandala contains

    Args:
        n (int): Number of hashes to generate.
        bits (int): Bit length of each hash (typically 160 or 256).

    Returns:
        pd.DataFrame: DataFrame with columns: hash, is_balanced, num_passages
    """
    data = []
    for _ in range(n):
        h = generate_random_hash(bits=bits)
        balanced = is_balanced(h)
        passages = count_unique_passages(h)
        data.append((h, balanced, passages))

    return pd.DataFrame(data, columns=["hash", "is_balanced", "num_passages"])

# Symmetries for HashCanon (Jupyter-friendly).
# Symmetry analysis and mandala drawing with optional overlay

from typing import List, Dict, Tuple, Optional
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

Symmetry = Tuple[int, int, str]  # (start, length, slice)

# ---------- Bit/grid helpers -------------------------------------------------

def _hex_clean(hex_str: str) -> str:
    return hex_str[2:] if hex_str.startswith("0x") else hex_str

def _bits_from_hex(hex_str: str) -> List[int]:
    clean = _hex_clean(hex_str)
    out: List[int] = []
    for h in clean:
        out.extend(int(b) for b in bin(int(h, 16))[2:].zfill(4))
    return out

def _grid_4xN(hex_str: str, sectors: int) -> List[List[int]]:
    """Return 4 x N grid; ring 0 is LSB (inner ring)."""
    bits = _bits_from_hex(hex_str)
    grid = [[0] * sectors for _ in range(4)]
    for s in range(sectors):
        for r in range(4):
            grid[r][s] = bits[s * 4 + r]
    return grid

# ---------- Symmetries core --------------------------------------------------

def _is_circular_palindrome(grid: List[List[int]], start: int, length: int, sectors: int) -> bool:
    """Check palindrome on all 4 rings over [start, start+length-1] (circular)."""
    half = length // 2
    for r in range(4):
        row = grid[r]
        for k in range(half):
            a = row[(start + k) % sectors]
            b = row[(start + length - 1 - k) % sectors]
            if a != b:
                return False
    return True

def _circular_slice(clean_hex: str, start: int, length: int) -> str:
    """Wrap-aware slice of the hex text (no 0x)."""
    L = len(clean_hex)
    end = (start + length) % L
    if start + length <= L:
        return clean_hex[start:start + length]
    return clean_hex[start:] + clean_hex[:end]

def _covers(a_start: int, a_len: int, b_start: int, b_len: int, sectors: int) -> bool:
    """Return True if segment b is fully covered by segment a on a circle."""
    for k in range(b_len):
        pos = (b_start + k) % sectors
        rel = (pos - a_start + sectors) % sectors
        if rel >= a_len:
            return False
    return True

def _uniq_max(sym_list: List[Symmetry], sectors: int) -> List[Symmetry]:
    """Keep only maximal (non-covered) circular palindromes."""
    sym_sorted = sorted(sym_list, key=lambda t: t[1], reverse=True)
    keep: List[Symmetry] = []
    for cand in sym_sorted:
        s, L, _ = cand
        if not any(_covers(S, L0, s, L, sectors) for (S, L0, _) in keep):
            keep.append(cand)
    return sorted(keep, key=lambda t: t[0])

def find_symmetries(hex_str: str, bits: int = 256) -> List[Symmetry]:
    """Return maximal circular palindromes as (start, length, slice)."""
    sectors = 64 if bits == 256 else 40
    grid = _grid_4xN(hex_str, sectors)
    clean = _hex_clean(hex_str)
    found: List[Symmetry] = []
    for start in range(sectors):
        for length in range(2, sectors + 1):
            if _is_circular_palindrome(grid, start, length, sectors):
                found.append((start, length, _circular_slice(clean, start, length)))
    return _uniq_max(found, sectors)

def symmetry_ranks(sym: List[Symmetry]) -> Dict[int, int]:
    """Aggregate counts by length (rank)."""
    ranks: Dict[int, int] = {}
    for _, L, _ in sym:
        ranks[L] = ranks.get(L, 0) + 1
    return dict(sorted(ranks.items()))

def symmetry_metric(hex_str: str, bits: int = 256) -> str:
    """Return string like: '9 total | Ranks: 2:3, 3:6'."""
    sym = find_symmetries(hex_str, bits=bits)
    ranks = symmetry_ranks(sym)
    ranks_str = ", ".join(f"{L}:{cnt}" for L, cnt in ranks.items())
    return f"{len(sym)} total | Ranks: {ranks_str}"

# ---------- Crown (max symmetry) --------------------------------------------

def crown_from_symmetries(sym: List[Symmetry]) -> Tuple[int, int, List[int], List[str]]:
    """
    Return (max_len, count, starts, slices) for the maximal-rank symmetries.
    Example: (4, 1, [start_idx], [slice_hex])
    """
    if not sym:
        return (0, 0, [], [])
    max_len = max(L for _, L, _ in sym)
    tops = [(s, sl) for (s, L, sl) in sym if L == max_len]
    return (max_len, len(tops), [s for s, _ in tops], [sl for _, sl in tops])

def crown_metric(hex_str: str, bits: int = 256) -> str:
    """Return 'L:K' for the maximal symmetry, e.g. '4:1'."""
    sym = find_symmetries(hex_str, bits=bits)
    L, cnt, _, _ = crown_from_symmetries(sym)
    return f"{L}:{cnt}" if L > 0 else "—"

# ---------- Overlay data for drawing ----------------------------------------

# Backward- and forward-compatible overlay builder
# Supports both legacy and new signatures:
#   1) symmetry_overlay_segments(sym, sectors=64|40)            # legacy
#   2) symmetry_overlay_segments(sym, hex_str=..., bits=...)    # new (no sectors)
#   3) symmetry_overlay_segments(sym)                           # heuristic (40/64)

def symmetry_overlay_segments(
    sym: List[Symmetry],
    sectors: Optional[int] = None,
    *,
    hex_str: Optional[str] = None,
    bits: Optional[int] = None,
) -> Dict[str, List[Tuple[int, int]]]:
    """Return overlay data with both legacy and new calling styles supported."""
    # ---- resolve sectors (keep legacy calls working) ------------------------
    if sectors is None:
        # try to infer from hex_str length
        if isinstance(hex_str, str):
            clean = hex_str[2:] if hex_str.startswith("0x") else hex_str
            if len(clean) in (40, 64):
                sectors = len(clean)
        # try to infer from bits
        if sectors is None and bits in (160, 256):
            sectors = bits // 4
        # fallback heuristic
        if sectors is None:
            fits40 = all((s < 40 and L <= 40) for (s, L, _) in sym) if sym else True
            sectors = 40 if fits40 else 64

    # ---- build lists --------------------------------------------------------
    max_len = max((L for _, L, _ in sym), default=0)
    boundaries = [(s, (s + L) % sectors) for (s, L, _) in sym]
    all_spans  = [(s, L) for (s, L, _) in sym]
    max_spans  = [(s, L) for (s, L, _) in sym if L == max_len]
    return {"boundaries": boundaries, "all_spans": all_spans, "max_spans": max_spans}

# Legacy adapter kept for internal/older calls; delegates to the unified API.
def symmetry_overlay_segments_prepare(sym: List[Symmetry], sectors: int) -> Dict[str, List[Tuple[int, int]]]:
    """Deprecated: use symmetry_overlay_segments(sym, ...) instead."""
    return symmetry_overlay_segments(sym, sectors=sectors)

# -------
def draw_mandala(hex_string, inner_radius=0.32, show_radial_line=False, sectors=64, symmetry_overlay_segments=False):
    """
    Draws a circular mandala of bits using hex input and specified number of sectors.

    Args:
        hex_string (str): Full hex string (with or without '0x').
        inner_radius (float): Radius of the central black circle.
        show_radial_line (bool): If True, draws red line at top (12 o'clock).
        sectors (int): Number of sectors (default 64).
        symmetry_overlay_segments (bool): If True, draws symmetry overlays
            (red boundaries, gray spans for all palindromes, red spans for maximals) exactly like the JS generator.
    """
    # ---- original rendering logic (unchanged) -------------------------------
    rings = 4
    angle_step = 2 * np.pi / sectors
    radius_step = 0.15
    base_radius = inner_radius
    rotation_offset = angle_step / 2

    total_bits = sectors * rings
    bit_array = hex_to_bit_array(hex_string)[:total_bits]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Center black circle
    ax.bar(
        x=0,
        height=base_radius,
        width=2*np.pi,
        bottom=0,
        color='black',
        edgecolor='black',
        linewidth=0
    )

    # Center hash text, split into 4 lines
    clean_hex = hex_string[2:] if hex_string.startswith("0x") else hex_string
    line_len = len(clean_hex) // 4
    square = [clean_hex[i:i + line_len] for i in range(0, len(clean_hex), line_len)]
    for i, line in enumerate(square):
        fig.text(0.51, 0.535 - i * 0.026, line,
                 ha='center', va='center',
                 color='white', fontsize=8, family='monospace')

    # Prepare a white-cell mask to reuse for overlays (avoid double-alpha)
    # grid_white[ring][sector] == 1 if the rendered cell is white
    grid_white = [[0] * sectors for _ in range(rings)]

    # Sector rendering
    for i in range(sectors):
        bin_segment = bit_array[i * 4:(i + 1) * 4][::-1]  # LSB inside; j=0 -> outer ring
        for j in range(rings):
            if bin_segment[j] == 1:
                grid_white[j][i] = 1  # remember white cell for overlay logic
                theta = i * angle_step + rotation_offset
                r_inner = (3 - j) * radius_step + base_radius
                ax.bar(
                    x=theta,
                    height=radius_step,
                    width=angle_step,
                    bottom=r_inner,
                    color='white',
                    edgecolor='black',
                    linewidth=0.5
                )

    # Optional top radial line (reference)
    if show_radial_line:
        theta_line = 0
        r_min = base_radius
        r_max = base_radius + rings * radius_step
        ax.plot([theta_line, theta_line], [r_min, r_max], color='red', linewidth=2)

    # ---- symmetry overlay (added; JS-equivalent semantics) ------------------
    if symmetry_overlay_segments:
        bits = 256 if sectors == 64 else 160
        syms = find_symmetries(hex_string, bits=bits)

        # 1) Red boundaries: draw exactly from base_radius to r_max (no overshoot)
        r_min = base_radius
        r_max = base_radius + rings * radius_step
        for s0, L, _ in syms:
            for edge in (s0, (s0 + L) % sectors):
                theta_edge = edge * angle_step  # sector boundary (no rotation offset)
                ax.plot([theta_edge, theta_edge], [r_min, r_max], color='red', linewidth=1)

        # Build coverage masks per cell to avoid alpha accumulation
        cover_gray = [[0] * sectors for _ in range(rings)]  # any palindrome
        cover_red  = [[0] * sectors for _ in range(rings)]  # maximal palindromes only

        # Mark coverage for all palindromes (gray)
        for s0, L, _ in syms:
            for k in range(L):
                i = (s0 + k) % sectors
                for j in range(rings):
                    if grid_white[j][i] == 1:  # paint only over originally white cells
                        cover_gray[j][i] = 1

        # Mark coverage for maximal palindromes (red)
        max_len = max((L for _, L, _ in syms), default=0)
        if max_len > 0:
            for s0, L, _ in syms:
                if L != max_len:
                    continue
                for k in range(L):
                    i = (s0 + k) % sectors
                    for j in range(rings):
                        cover_red[j][i] = 1

        # 2) Gray translucent fills for all palindromes (one paint per cell)
        for i in range(sectors):
            theta_center = i * angle_step + rotation_offset
            for j in range(rings):
                if cover_gray[j][i] == 1 and cover_red[j][i] == 0:  # ← added mask
                    r_inner = (3 - j) * radius_step + base_radius
                    ax.bar(
                        x=theta_center,
                        height=radius_step,
                        width=angle_step,
                        bottom=r_inner,
                        color='gray',
                        edgecolor=None,
                        linewidth=0,
                        alpha=0.50
                    )

        # 3) Red translucent fills for maximal palindromes (override gray)
        if max_len > 0:
            for i in range(sectors):
                theta_center = i * angle_step + rotation_offset
                for j in range(rings):
                    if cover_red[j][i] == 1:
                        r_inner = (3 - j) * radius_step + base_radius
                        ax.bar(
                            x=theta_center,
                            height=radius_step,
                            width=angle_step,
                            bottom=r_inner,
                            color='red',
                            edgecolor=None,
                            linewidth=0,
                            alpha=0.32
                        )

    plt.show()

