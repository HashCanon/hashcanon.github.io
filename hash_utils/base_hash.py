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
    "hex_to_binary_grid_corrected",
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


def hex_to_bit_array(hex_string):
    """
    Converts a hex string to a flat array of bits.

    Args:
        hex_string (str): Hex string with or without '0x'.

    Returns:
        list[int]: List of 256 bits (64 hex chars × 4 bits).
    """
    if hex_string.startswith('0x'):
        hex_string = hex_string[2:]
    return [int(bit) for h in hex_string for bit in bin(int(h, 16))[2:].zfill(4)]


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