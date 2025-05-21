import matplotlib.pyplot as plt
import numpy as np
import random, string, hashlib

__all__ = [
    "generate_random_hash",
    "explain_hex_to_bin",
    "hex_to_binary_grid_corrected",
    "draw_binary_grid_from_hex_dark",
    "draw_mandala",
    "hash_to_hex256"
]


def generate_random_hash() -> str:
    """
    Generates a random 256-bit hex string (64 characters) with '0x' prefix.

    Returns:
        str: A randomly generated hex string.
    """
    alphabet = string.hexdigits.lower()[:16]   # '0123456789abcdef'
    return '0x' + ''.join(random.choices(alphabet, k=64))


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


def hash_to_hex256(word: str) -> str:
    """
    Returns the SHA-256 hash of a string as a hex string with '0x' prefix.

    Args:
        word (str): Input string.

    Returns:
        str: SHA-256 hash in hex format.
    """
    hash_bytes = hashlib.sha256(word.encode()).digest()
    return '0x' + hash_bytes.hex()
