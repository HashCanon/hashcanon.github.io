// web/main.js

/* ---------- constants ---------- */
const RESIZE_THROTTLE_MS = 100;       // ≈10 fps (1000 ms / 10)
const visitedCache = {                // reusable “globalVisited” matrices
  256: Array.from({ length: 4 }, () => Array(64).fill(false)),
  160: Array.from({ length: 4 }, () => Array(40).fill(false))
};

/* ---------- random hash generator ---------- */
function generateRandomHash(bits = 256) {
  const hex = "0123456789abcdef";
  const length = bits / 4;
  let result = "0x";
  for (let i = 0; i < length; i++) {
    result += hex[Math.floor(Math.random() * 16)];
  }
  return result;
}

/* ---------- feature analysis & UI write-back ---------- */
function renderFeatures(hex, bits) {
  const featuresContent = document.getElementById("features-content");
  const balanced = isBalanced(hex);
  const passages = countPassagesFromHex(hex);

  const balanceText = balanced
    ? `<span style="color: green"><strong>Balanced</strong></span> — equal number of 0 s and 1 s in the hash, which occurs naturally in only ~5–6 % of cases`
    : `<span style="color: black"><strong>Unbalanced</strong></span> — unequal number of 0 s and 1 s; the more common state for hashes generated from entropy`;

  const passageText = passages === 0
    ? `<span style="color: green"><strong>Sealed</strong></span> — no valid passage connects the center to the outer ring; all paths are blocked by “white walls”, forming a fully enclosed mandala (extremely rare).`
    : `<span style="color: black"><strong>${passages} passage${passages > 1 ? 's' : ''}</strong></span> — continuous black-bit paths lead from the inner core to the edge, forming clear “exits”. These passages are isolated by white barriers preventing cross-travel.`;

  featuresContent.innerHTML = `
    <ul style="padding-left: 1em; margin: 0;">
      <li>${balanceText}</li>
      <li>${passageText}</li>
    </ul>
  `;
}

/* ---------- main “Generate” workflow ---------- */
function generate() {
  const hashInput = document.getElementById("hashInput").value.trim();
  const textInput = document.getElementById("textInput").value.trim();
  const status    = document.getElementById("status");

  const bits           = parseInt(document.querySelector('input[name="hashBits"]:checked').value, 10);
  const expectedLength = bits / 4;

  const isValidHash = /^0x[0-9a-fA-F]+$/.test(hashInput) &&
                      hashInput.length === 2 + expectedLength;
  const useText     = !!textInput;

  let hex = "";

  if (useText) {
    /* text → SHA-256 → hex */
    const buffer = new TextEncoder().encode(textInput);
    crypto.subtle.digest("SHA-256", buffer).then(hashBuffer => {
      const hexFull = Array.from(new Uint8Array(hashBuffer))
        .map(b => b.toString(16).padStart(2, '0'))
        .join('');
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
  document.getElementById("hashInput").placeholder =
    `Enter 0x... custom hash (${bits / 4} hex chars)`;
}

/* initial render */
generate();

/* ---------- responsive SVG (resize throttled) ---------- */
function adjustSvgSizeToViewport() {
  const container       = document.getElementById("svg-container");
  const vw              = window.innerWidth;
  const vh              = window.innerHeight;
  const availableHeight = vh - 120;                   // leave space for UI
  const targetSize      = Math.min(vw, availableHeight, 800);
  container.style.width  = `${targetSize}px`;
  container.style.height = `${targetSize}px`;
}

/* throttle wrapper */
let resizeTimer = null;
function onResizeThrottled() {
  if (resizeTimer !== null) return;
  resizeTimer = setTimeout(() => {
    adjustSvgSizeToViewport();
    resizeTimer = null;
  }, RESIZE_THROTTLE_MS);
}

window.addEventListener("resize", onResizeThrottled);
window.addEventListener("load",   adjustSvgSizeToViewport);
document.querySelectorAll('input[name="hashBits"]').forEach(radio =>
  radio.addEventListener("change", updateHashPlaceholder)
);

/* ---------- SVG download ---------- */
function downloadSVG() {
  const svg = document.querySelector("#svg-container svg");
  if (!svg) return;

  const source = new XMLSerializer().serializeToString(svg);
  const blob   = new Blob([source], { type: "image/svg+xml;charset=utf-8" });
  const url    = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "hashjing-mandala.svg";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/* ---------- PNG download ---------- */
function downloadPNG() {
  const svgElement = document.querySelector("#svg-container svg");
  if (!svgElement) return;

  const serializer = new XMLSerializer();
  const svgString  = serializer.serializeToString(svgElement);
  const svgBlob    = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
  const url        = URL.createObjectURL(svgBlob);

  const canvasSize = 1024;
  const canvas = document.createElement("canvas");
  canvas.width  = canvasSize;
  canvas.height = canvasSize;
  const ctx = canvas.getContext("2d");

  const img = new Image();
  img.onload = () => {
    /* clear and draw the image pixel-perfect in the centre */
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

  /* Important: set CORS so rendering works */
  img.crossOrigin = "anonymous";
  img.src = url;
}

/* ---------- input field sync ---------- */
document.getElementById("textInput").addEventListener("input", () => {
  if (document.getElementById("textInput").value.trim())
    document.getElementById("hashInput").value = "";
});
document.getElementById("hashInput").addEventListener("input", () => {
  if (document.getElementById("hashInput").value.trim())
    document.getElementById("textInput").value = "";
});

/* ---------- bit-level helpers ---------- */
function isBalanced(hexString) {
  const bin = BigInt(hexString).toString(2)
               .padStart((hexString.length - 2) * 4, '0');
  const zeros = [...bin].filter(b => b === '0').length;
  const ones  = bin.length - zeros;
  return zeros === ones;
}

function hexToGrid(hexString, bits) {
  const clean = hexString.startsWith("0x") ? hexString.slice(2) : hexString;
  const binary = clean.split('').flatMap(h =>
    parseInt(h, 16).toString(2).padStart(4, '0').split('').map(Number)
  );

  const sectors = bits === 256 ? 64 : 40;
  const rings   = 4;
  const grid    = Array.from({ length: rings }, () => Array(sectors).fill(0));

  for (let s = 0; s < sectors; s++) {
    for (let r = 0; r < rings; r++) {
      grid[r][s] = binary[s * rings + r];
    }
  }
  return grid;
}

function countPassagesFromHex(hexString) {
  const bits    = hexString.length === 42 ? 160 : 256;
  const grid    = hexToGrid(hexString, bits);
  const rings   = 4;
  const sectors = grid[0].length;

  /* reuse the cached matrix instead of reallocating */
  const globalVisited = visitedCache[bits];
  for (let r = 0; r < rings; r++) globalVisited[r].fill(false);

  let passageCount = 0;

  for (let startSector = 0; startSector < sectors; startSector++) {
    if (grid[0][startSector] !== 0 || globalVisited[0][startSector]) continue;

    const queue        = [[0, startSector]];
    const localVisited = Array.from({ length: rings }, () => Array(sectors).fill(false));
    const pathCells    = [];
    let reachedEdge    = false;

    while (queue.length) {
      const [r, s] = queue.shift();
      if (localVisited[r][s] || grid[r][s] !== 0) continue;

      localVisited[r][s] = true;
      pathCells.push([r, s]);
      if (r === rings - 1) reachedEdge = true;

      const nei = [
        [r + 1, s],
        [r - 1, s],
        [r, (s + 1) % sectors],
        [r, (s - 1 + sectors) % sectors]
      ];

      for (const [nr, ns] of nei) {
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
      passageCount++;
      for (const [r, s] of pathCells) globalVisited[r][s] = true;
    }
  }

  return passageCount;
}
