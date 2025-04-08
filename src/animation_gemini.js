const canvas = document.getElementById('peAnimationCanvas');
const ctx = canvas.getContext('2d');
const width = canvas.width;
const height = canvas.height;

// --- Parameters ---
const numBits = 4;
const bitDisplayY = 60;       // Vertical position of the '0'/'1' digits
const waveTopY = bitDisplayY + 15; // Top of the sine wave drawing area
const waveBottomY = height - 30;  // Bottom of the sine wave drawing area
const waveViewHeight = waveBottomY - waveTopY;
const amplitude = 20;         // Horizontal oscillation amplitude of waves
const basePeriod = 80;        // Period 'T' (in 'pos' units) for the leftmost (fastest) wave
const maxPosHistory = basePeriod * (2**(numBits - 1)); // How much 'pos' history to show vertically (e.g., one full cycle of slowest wave)

let currentPos = 0;
let posIncrement = 0.5; // Initial Speed: Controls how much 'pos' advances each frame

// --- Speed Control ---
const speedSlider = document.getElementById('speedControl');
const speedValueSpan = document.getElementById('speedValue');
speedSlider.oninput = function() {
    posIncrement = parseFloat(this.value);
    speedValueSpan.textContent = posIncrement.toFixed(1);
}
speedValueSpan.textContent = posIncrement.toFixed(1); // Initial display

// --- Helper Functions ---
function getBitCenterX(bitIndex) {
    // Calculate horizontal center for each bit display and wave
    return (width / (numBits + 1)) * (bitIndex + 1);
}

function mapPosToY(pos, viewStart, viewEnd) {
    // Maps a position value 'pos' within the viewable range [viewStart, viewEnd]
    // to a vertical coordinate 'y' within the wave drawing area.
    if (viewEnd === viewStart) return waveBottomY; // Avoid division by zero
    const relativePos = (pos - viewStart) / (viewEnd - viewStart); // Normalize pos within view [0, 1]
    return waveBottomY - (relativePos * waveViewHeight); // Map to Y (inverted: 0=bottom, 1=top)
}

// --- Drawing Function ---
function draw() {
    // 1. Clear Canvas
    ctx.clearRect(0, 0, width, height);

    // 2. Update State
    currentPos += posIncrement;

    // 3. Determine Position Range to Draw
    const posEnd = currentPos; // The 'present' time is at the top
    const posStart = Math.max(0, posEnd - maxPosHistory); // How far back in 'pos' to draw

    // Draw axis label (optional)
    ctx.fillStyle = 'grey';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText('Pos ≈ 0', getBitCenterX(0) - amplitude - 35 , waveBottomY + 5);
    ctx.fillText(`Pos ≈ ${maxPosHistory.toFixed(0)}`, getBitCenterX(0) - amplitude - 35, waveTopY - 5);


    // 4. Draw Each Bit and its Wave
    for (let i = 0; i < numBits; i++) {
        const bitX = getBitCenterX(i);
        const period = basePeriod * (2**i); // T, 2T, 4T, 8T for i=0, 1, 2, 3
        const omega = (2 * Math.PI) / period;

        // --- Draw the Sine Wave Segment ---
        ctx.beginPath();
        let firstPoint = true;
        const waveResolution = 2; // Draw line segment every N pos units

        for (let p = posStart; p <= posEnd; p += waveResolution) {
            const y = mapPosToY(p, posStart, posEnd);
            const xOffset = amplitude * Math.sin(omega * p);
            const x = bitX + xOffset;

            // Ensure drawing stays within vertical bounds visually
            if (y < waveTopY - 2 || y > waveBottomY + 2) continue; // Small tolerance

            if (firstPoint) {
                ctx.moveTo(x, y);
                firstPoint = false;
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.strokeStyle = `hsl(${200 + i*20}, 70%, 60%)`; // Vary color slightly
        ctx.lineWidth = 2;
        ctx.stroke();

        // --- Determine and Draw the Bit Value ---
        // Value is determined by the sine wave at the *current* position
        const currentSineValue = Math.sin(omega * currentPos);
        const bitValue = (currentSineValue >= 0) ? '1' : '0';

        ctx.fillStyle = 'black';
        ctx.font = 'bold 30px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(bitValue, bitX, bitDisplayY);

         // Optional: Draw a marker for the current position on the wave
         const markerY = waveTopY; // Current pos is always at the top edge of the view
         const markerXOffset = amplitude * Math.sin(omega * currentPos);
         ctx.fillStyle = 'red';
         ctx.beginPath();
         ctx.arc(bitX + markerXOffset, markerY, 5, 0, 2 * Math.PI);
         ctx.fill();
         ctx.strokeStyle = 'black';
         ctx.lineWidth = 1;
         ctx.stroke();


        // Draw Period Label (optional)
        ctx.fillStyle = 'grey';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`P=${period/basePeriod}T`, bitX, waveBottomY + 20);

    }

    // 5. Request Next Frame
    requestAnimationFrame(draw);
}

// --- Start Animation ---
draw();