function generateRandomHash(bits = 256) {
  const hex = "0123456789abcdef";
  const length = bits / 4;
  let result = "0x";
  for (let i = 0; i < length; i++) {
    result += hex[Math.floor(Math.random() * 16)];
  }
  return result;
}

function renderFeatures(hex, bits) {
  const featuresContent = document.getElementById("features-content");
  const balanced = isBalanced(hex);
  const passages = countPassagesFromHex(hex);

  const balanceText = balanced
    ? `<span style="color: green"><strong>Balanced</strong></span> — equal number of 0s and 1s in the hash, which occurs naturally in only ~5–6% of cases`
    : `<span style="color: black"><strong>Unbalanced</strong></span> — unequal number of 0s and 1s; the more common state for hashes generated from entropy`;

  const passageText = passages === 0
    ? `<span style="color: green"><strong>Sealed</strong></span> — no valid passage connects the center to the outer ring; all paths are blocked by “white walls”, forming a fully enclosed mandala. Such sealed structures are extremely rare.`
    : `<span style="color: black"><strong>${passages} passage${passages > 1 ? 's' : ''}</strong></span> — continuous black-bit paths lead from the inner core to the edge, forming clear “exits” through the structure. These passages are isolated by white barriers that prevent crossing between them.`;


  featuresContent.innerHTML = `
    <ul style="padding-left: 1em; margin: 0;">
      <li>${balanceText}</li>
      <li>${passageText}</li>
    </ul>
  `;
}

function generate() {
  const hashInput = document.getElementById("hashInput").value.trim();
  const textInput = document.getElementById("textInput").value.trim();
  const status = document.getElementById("status");

  const bits = parseInt(document.querySelector('input[name="hashBits"]:checked').value, 10);
  const expectedLength = bits / 4;

  const isValidHash = /^0x[0-9a-fA-F]+$/.test(hashInput) && hashInput.length === 2 + expectedLength;
  const useText = !!textInput;

  let hex = "";

  if (useText) {
    const buffer = new TextEncoder().encode(textInput);
    crypto.subtle.digest("SHA-256", buffer).then(hashBuffer => {
      const hexFull = Array.from(new Uint8Array(hashBuffer))
        .map(b => b.toString(16).padStart(2, '0')).join('');
      hex = "0x" + hexFull.slice(-expectedLength);
      status.textContent = `Text input → SHA-256 (${bits} bits) → ${hex}`;
      status.style.color = "green";
      drawMandala(hex, bits);
      renderFeatures(hex, bits);
    });
  } else if (isValidHash) {
    hex = hashInput;
    status.textContent = `Using input hash: ${hex}`;
    status.style.color = "green";
    drawMandala(hex, bits);
    renderFeatures(hex, bits);
  } else {
    hex = generateRandomHash(bits);
    status.textContent = `No valid input → using random: ${hex}`;
    status.style.color = "red";
    drawMandala(hex, bits);
    renderFeatures(hex, bits);
  }
}

function updateHashPlaceholder() {
  const bits = parseInt(document.querySelector('input[name="hashBits"]:checked').value, 10);
  const placeholder = `Enter 0x... custom hash (${bits / 4} hex chars)`;
  document.getElementById("hashInput").placeholder = placeholder;
}

generate();

function adjustSvgSizeToViewport() {
  const container = document.getElementById("svg-container");
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const availableHeight = vh - 120;
  const targetSize = Math.min(vw, availableHeight, 800);
  container.style.width = `${targetSize}px`;
  container.style.height = `${targetSize}px`;
}

window.addEventListener("resize", adjustSvgSizeToViewport);
window.addEventListener("load", adjustSvgSizeToViewport);

document.querySelectorAll('input[name="hashBits"]').forEach(radio => {
  radio.addEventListener("change", updateHashPlaceholder);
});

function downloadSVG() {
  const svg = document.querySelector("#svg-container svg");
  if (!svg) return;

  const serializer = new XMLSerializer();
  const source = serializer.serializeToString(svg);

  const blob = new Blob([source], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "hashjing-mandala.svg";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

document.getElementById("textInput").addEventListener("input", () => {
  if (document.getElementById("textInput").value.trim()) {
    document.getElementById("hashInput").value = "";
  }
});

document.getElementById("hashInput").addEventListener("input", () => {
  if (document.getElementById("hashInput").value.trim()) {
    document.getElementById("textInput").value = "";
  }
});

function isBalanced(hexString) {
  const bin = BigInt(hexString).toString(2).padStart((hexString.length - 2) * 4, '0');
  const zeros = [...bin].filter(b => b === '0').length;
  const ones = [...bin].filter(b => b === '1').length;
  return zeros === ones;
}

// Accurate implementation of count_unique_passages from base_hash.py
function hexToGrid(hexString, bits) {
  const clean = hexString.startsWith("0x") ? hexString.slice(2) : hexString;
  const binary = clean.split('').flatMap(h =>
    parseInt(h, 16).toString(2).padStart(4, '0').split('').map(Number)
  );

  const sectors = bits === 256 ? 64 : 40;
  const rings = 4;
  const grid = Array.from({ length: rings }, () => Array(sectors).fill(0));

  for (let sector = 0; sector < sectors; sector++) {
    for (let ring = 0; ring < rings; ring++) {
      const bitIndex = sector * rings + ring;
      grid[ring][sector] = binary[bitIndex];
    }
  }
  return grid;
}

function countPassagesFromHex(hexString) {
  const bits = hexString.length === 42 ? 160 : 256;
  const grid = hexToGrid(hexString, bits);
  const rings = 4;
  const sectors = grid[0].length;

  const globalVisited = Array.from({ length: rings }, () => Array(sectors).fill(false));
  let passageCount = 0;

  for (let startSector = 0; startSector < sectors; startSector++) {
    if (grid[0][startSector] !== 0 || globalVisited[0][startSector]) continue;

    const queue = [[0, startSector]];
    const localVisited = Array.from({ length: rings }, () => Array(sectors).fill(false));
    const pathCells = [];
    let reachedEdge = false;

    while (queue.length > 0) {
      const [r, s] = queue.shift();
      if (localVisited[r][s] || grid[r][s] !== 0) continue;

      localVisited[r][s] = true;
      pathCells.push([r, s]);
      if (r === rings - 1) reachedEdge = true;

      const neighbors = [
        [r + 1, s],
        [r - 1, s],
        [r, (s + 1) % sectors],
        [r, (s - 1 + sectors) % sectors]
      ];

      for (const [nr, ns] of neighbors) {
        if (
          nr >= 0 && nr < rings &&
          !localVisited[nr][ns] &&
          !globalVisited[nr][ns] &&
          grid[nr][ns] === 0
        ) {
          queue.push([nr, ns]);
        }
      }
    }

    if (reachedEdge) {
      passageCount += 1;
      for (const [r, s] of pathCells) {
        globalVisited[r][s] = true;
      }
    }
  }

  return passageCount;
}

function downloadPNG() {
  const svgElement = document.querySelector("#svg-container svg");
  if (!svgElement) return;

  const serializer = new XMLSerializer();
  const svgString = serializer.serializeToString(svgElement);
  const svgBlob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(svgBlob);

  const canvasSize = 1024;
  const canvas = document.createElement("canvas");
  canvas.width = canvasSize;
  canvas.height = canvasSize;
  const ctx = canvas.getContext("2d");

  const img = new Image();
  img.onload = () => {
    // полностью очистим и отрисуем изображение точно по центру
    ctx.clearRect(0, 0, canvasSize, canvasSize);
    ctx.drawImage(img, 0, 0, canvasSize, canvasSize);

    URL.revokeObjectURL(url);

    const a = document.createElement("a");
    a.download = "hashjing-mandala.png";
    a.href = canvas.toDataURL("image/png");
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  // Важно: задать CORS для корректной отрисовки
  img.crossOrigin = "anonymous";
  img.src = url;
}


