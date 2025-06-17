// web/mandala.js

const now = new Date().toISOString().split('T')[0]; // format YYYY-MM-DD

const metadata = `
  <metadata>
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
             xmlns:dc="http://purl.org/dc/elements/1.1/">
      <dc:title>HashJing Mandala</dc:title>
      <dc:creator>DataSattva</dc:creator>
      <dc:description>Generative art based on cryptographic hash entropy and symbolic geometry.</dc:description>
      <dc:date>2025</dc:date>
      <dc:rights>MIT License (code); CC BY-NC 4.0 (visuals)</dc:rights>
      <dc:identifier>https://github.com/DataSattva/HashJing</dc:identifier>
      <dc:language>en</dc:language>
      <dc:format>image/svg+xml</dc:format>
      <dc:subject>generative art, mandala, hash, entropy, cryptography, I Ching</dc:subject>
    </rdf:RDF>
  </metadata>
  <desc>
    Generated with HashJing by DataSattva on ${now}
  </desc>
`;

function hexToBitArray(hex) {
  return hex.slice(2).split('').flatMap(h =>
    parseInt(h, 16).toString(2).padStart(4, '0').split('').map(Number)
  );
}

function polarToCartesian(cx, cy, r, angle) {
  return {
    x: cx + r * Math.cos(angle),
    y: cy + r * Math.sin(angle)
  };
}

function generateSectorPath(cx, cy, i, j, angleStep, radiusStep, baseRadius) {
  const angleStart = i * angleStep - Math.PI / 2;
  const angleEnd   = angleStart + angleStep;
  const rInner = (3 - j) * radiusStep + baseRadius;
  const rOuter = rInner + radiusStep;

  const a0 = polarToCartesian(cx, cy, rInner, angleStart);
  const a1 = polarToCartesian(cx, cy, rInner, angleEnd);
  const b1 = polarToCartesian(cx, cy, rOuter, angleEnd);
  const b0 = polarToCartesian(cx, cy, rOuter, angleStart);

  return [
    `M ${a0.x} ${a0.y}`,
    `A ${rInner} ${rInner} 0 0 1 ${a1.x} ${a1.y}`,
    `L ${b1.x} ${b1.y}`,
    `A ${rOuter} ${rOuter} 0 0 0 ${b0.x} ${b0.y}`,
    'Z'
  ].join(' ');
}

function drawMandala(hex, bits = 256) {
  const canvasSize  = 1024;
  const cx          = canvasSize / 2;
  const cy          = canvasSize / 2;
  const rings       = 4;
  const sectors     = bits / 4;
  const angleStep   = (2 * Math.PI) / sectors;
  const radiusStep  = 80;
  const baseRadius  = 160;

  const signature = `<text x="${canvasSize - 10}" y="${canvasSize - 10}" font-size="18" fill="white" text-anchor="end" font-family="monospace" opacity="0.5">github.com/DataSattva/HashJing</text>`;
  const bitsArray = hexToBitArray(hex);
  const bgRect    = `<rect x="0" y="0" width="${canvasSize}" height="${canvasSize}" fill="black"/>`;
  const paths     = [];

  for (let i = 0; i < sectors; i++) {
    const bin = bitsArray.slice(i * 4, (i + 1) * 4).reverse();
    for (let j = 0; j < rings; j++) {
      const path = generateSectorPath(cx, cy, i, j, angleStep, radiusStep, baseRadius);
      const fill = bin[j] === 1 ? 'white' : 'black';
      paths.push(`<path d="${path}" fill="${fill}" stroke="black" stroke-width="1"/>`);
    }
  }

  const lineLength   = bits === 160 ? 10 : 16;
  const square       = hex.slice(2).match(new RegExp(`.{1,${lineLength}}`, 'g'));
  const textElements = square.map((line, i) =>
    `<text x="${cx}" y="${cy - 45 + i * 36}" font-size="22" fill="white" text-anchor="middle" font-family="monospace">${line}</text>`
  );

  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg"
         viewBox="0 0 ${canvasSize} ${canvasSize}"
         preserveAspectRatio="xMidYMid meet">
      ${metadata}
      ${[bgRect, ...paths, ...textElements, signature].join('\n')}
    </svg>
  `;

  document.getElementById('svg-container').innerHTML = svg;
}
