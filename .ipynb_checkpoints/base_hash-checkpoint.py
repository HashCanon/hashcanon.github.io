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
    alphabet = string.hexdigits.lower()[:16]   # '0123456789abcdef'
    return '0x' + ''.join(random.choices(alphabet, k=64))

    

def explain_hex_to_bin(hex_str: str) -> None:
    """
    Выводит поэлементное представление 64-символьной hex-строки в виде 4-битных бинарных блоков.

    Args:
        hex_str (str): Строка из 64 hex-символов (может начинаться с префикса '0x').

    Example:
        >>> explain_hex_to_bin("0x1a2b3c...")
        00: 1 -> 0001
        01: A -> 1010
        ...
    """
    # Убираем префикс 0x, если есть
    if hex_str.startswith("0x"):
        hex_str = hex_str[2:]

    assert len(hex_str) == 64, "Ожидается 64 hex-символа (256 бит)"

    for i, char in enumerate(hex_str):
        binary = format(int(char, 16), '04b')
        print(f"{i:02d}: {char.upper()} -> {binary}")


def draw_binary_grid_from_hex(hex_str: str, show_guides: bool = True) -> None:
    """
    Отображает бинарную матрицу 4×64 на основе hex-строки.

    Args:
        hex_str (str): Hex-строка длиной 64 символа (может начинаться с '0x').
        show_guides (bool): Если True — рисует красные линии-разделители между блоками по 16 символов.
    """
    if hex_str.startswith("0x"):
        hex_str = hex_str[2:]

    assert len(hex_str) == 64, "Ожидается 64 hex-символа"

    grid = np.zeros((4, 64), dtype=int)

    for i, char in enumerate(hex_str):
        bin_str = format(int(char, 16), '04b')
        for j, bit in enumerate(bin_str):
            grid[j, i] = 1 - int(bit)

    # Визуализация
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
    Визуализирует 4×64 бинарную матрицу по hex-строке на чёрном фоне.

    Args:
        hex_str (str): Hex-строка длиной 64 символа (может начинаться с '0x').
        show_guides (bool): Если True — отображаются красные вертикальные разделители.
    """
    if hex_str.startswith("0x"):
        hex_str = hex_str[2:]

    assert len(hex_str) == 64, "Ожидается 64 hex-символа"

    grid = np.zeros((4, 64), dtype=int)

    for i, char in enumerate(hex_str):
        bin_str = format(int(char, 16), '04b')
        for j, bit in enumerate(bin_str):
            grid[j, i] = 1 - int(bit)  # 0 → чёрный, 1 → белый (инверсия)

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
    Возвращает 
    bits (list): Список из 256 битов (64 hex-символа * 4).
    """
    if hex_string.startswith('0x'):
        hex_string = hex_string[2:]
    return [int(bit) for h in hex_string for bit in bin(int(h, 16))[2:].zfill(4)]


def pad_hex_to_64(hex_string: str) -> str:
    """
    Дополняет hex-строку нулями справа до 64 символов и возвращает с префиксом '0x'.

    Args:
        hex_string (str): Строка hex (включая '0x' или без неё)

    Returns:
        str: Дополненная строка длиной 64 символа с префиксом '0x'
    """
    clean = hex_string.lower().replace("0x", "")
    padded = clean.ljust(64, "0")[:64]
    return "0x" + padded


def draw_mandala(hex_string, inner_radius=0.32, show_radial_line=False, sectors=64):
    """
    Отображает мандалу с заданным числом секторов и 4 кольцами на основе hex-строки.

    Args:
        hex_string (str): Hex-строка.
        inner_radius (float): Радиус внутреннего круга.
        show_radial_line (bool): Если True — рисует красную линию на 12 часов.
        sectors (int): Количество секторов (по умолчанию 64).
    """
    rings = 4
    angle_step = 2 * np.pi / sectors
    radius_step = 0.15
    base_radius = inner_radius
    rotation_offset = angle_step / 2

    # Подготовка данных
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

    # Центральный чёрный круг
    ax.bar(
        x=0,
        height=base_radius,
        width=2*np.pi,
        bottom=0,
        color='black',
        edgecolor='black',
        linewidth=0
    )

    # Хэш в центре (разбиение на 4 строки по длине hex_string // 4)
    clean_hex = hex_string[2:] if hex_string.startswith("0x") else hex_string
    line_len = len(clean_hex) // 4
    square = [clean_hex[i:i + line_len] for i in range(0, len(clean_hex), line_len)]
    for i, line in enumerate(square):
        fig.text(0.51, 0.535 - i * 0.026, line,
                 ha='center', va='center',
                 color='white', fontsize=8, family='monospace')

    # Отрисовка секторов
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
    hash_bytes = hashlib.sha256(word.encode()).digest()
    return '0x' + hash_bytes.hex()