// Independent per-browser canvas state. Only Generate/Guess sends data to Python.
const host = element.closest('#ggf-inpaint-canvas') || element;
const canvas = element.querySelector('.paint-canvas');
const viewport = element.querySelector('.paint-viewport');
const info = element.querySelector('.paint-info');
const context = canvas.getContext('2d');
const overlay = document.createElement('canvas');
const marks = overlay.getContext('2d');
let photo = null, source = null, strokes = [], redo = [], active = null;
let tool = 'add', hidden = false, zoom = 1, uploading = false, loadVersion = 0;
const q = (selector) => element.querySelector(selector);
const brushSize = () => Number(q('.paint-size').value);

function setTool(next) {
  tool = next;
  element.querySelectorAll('[data-tool]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.tool === tool)));
  q('.paint-color').disabled = tool !== 'color';
  q('.paint-color').title = tool === 'color' ? 'Desired color for Color strokes' : 'Select Color to recolor an object. Add and Remove use fixed guide colors.';
  q('.paint-tool-status').textContent = ({add:'ADD: sketch a new shape', remove:'REMOVE: paint the object to erase', color:'COLOR: paint the desired color', erase:'ERASER: clear brush marks'})[tool];
}

function controls() {
  q('.paint-undo').disabled = !strokes.length;
  q('.paint-redo').disabled = !redo.length;
  if (photo) info.textContent = `${photo.width} × ${photo.height} · ${strokes.length} stroke${strokes.length === 1 ? '' : 's'}`;
}
function drawStroke(ctx, stroke) {
  const points = stroke.points;
  if (!points.length) return;
  ctx.save();
  ctx.globalCompositeOperation = stroke.tool === 'erase' ? 'destination-out' : 'source-over';
  ctx.strokeStyle = stroke.tool === 'add' ? '#00e676' : stroke.tool === 'remove' ? '#ff2d55' : stroke.color;
  ctx.fillStyle = ctx.strokeStyle;
  ctx.globalAlpha = stroke.tool === 'remove' ? .8 : 1;
  ctx.lineWidth = stroke.size;
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.beginPath();
  points.forEach((p, i) => i ? ctx.lineTo(p[0] * (canvas.width - 1), p[1] * (canvas.height - 1)) : ctx.moveTo(p[0] * (canvas.width - 1), p[1] * (canvas.height - 1)));
  ctx.stroke();
  if (points.length === 1) {
    ctx.beginPath(); ctx.arc(points[0][0] * (canvas.width - 1), points[0][1] * (canvas.height - 1), stroke.size / 2, 0, Math.PI * 2); ctx.fill();
  }
  ctx.restore();
}
function render() {
  if (!photo) return;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(photo, 0, 0);
  marks.clearRect(0, 0, overlay.width, overlay.height);
  if (!hidden) {strokes.forEach(s => drawStroke(marks, s)); if (active) drawStroke(marks, active);}
  context.drawImage(overlay, 0, 0);
  controls();
}
function fit() {
  if (!photo || viewport.clientWidth < 10) return;
  const scale = Math.min(viewport.clientWidth / photo.width, viewport.clientHeight / photo.height) * zoom;
  canvas.style.width = `${Math.max(1, photo.width * scale)}px`;
  canvas.style.height = `${Math.max(1, photo.height * scale)}px`;
}
async function loadImage(url) {
  const version = ++loadVersion;
  uploading = true;
  info.textContent = 'Loading image…';
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    if (version !== loadVersion) return;
    if (image.width * image.height > 16777216) throw new Error('Use an image up to 4096 × 4096 pixels.');
    // Normalize orientation and format in browser, preserving original pixel size.
    const normalized = document.createElement('canvas');
    normalized.width = image.width; normalized.height = image.height;
    const ctx = normalized.getContext('2d');
    ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, image.width, image.height); ctx.drawImage(image, 0, 0);
    source = normalized.toDataURL('image/png');
    photo = normalized;
    canvas.width = overlay.width = image.width; canvas.height = overlay.height = image.height;
    strokes = []; redo = []; active = null; hidden = false;
    zoom = 1; q('.paint-zoom').value = '1';
    q('.paint-hide').textContent = 'Hide marks'; q('.paint-hide').setAttribute('aria-pressed', 'false');
    canvas.hidden = false; q('.paint-empty').style.display = 'none';
    fit(); render();
    trigger('upload');
  } catch (error) {info.textContent = `Could not load image: ${error.message}`;}
  finally {if (version === loadVersion) uploading = false;}
}
function loadFile(file) {
  if (!file || !/^image\/(png|jpeg|webp)$/.test(file.type)) {info.textContent = 'Choose a PNG, JPEG, or WebP image.'; return;}
  if (file.size > 40 * 1024 * 1024) {info.textContent = 'Use an image smaller than 40 MB.'; return;}
  const url = URL.createObjectURL(file);
  loadImage(url).finally(() => URL.revokeObjectURL(url));
}
q('.paint-upload').addEventListener('change', event => {loadFile(event.target.files[0]); event.target.value = '';});
viewport.addEventListener('dragover', event => {event.preventDefault();});
viewport.addEventListener('drop', event => {event.preventDefault(); loadFile(event.dataTransfer.files[0]);});
element.querySelectorAll('[data-tool]').forEach(button => button.addEventListener('click', () => {
  finish();
  setTool(button.dataset.tool);
}));
// Picking a color must never turn Remove into Color behind the user's back.
q('.paint-color').addEventListener('input', () => controls());
setTool('add');
q('.paint-size').addEventListener('input', () => {q('.size-value').textContent = brushSize();});
const point = (event) => {const rect = canvas.getBoundingClientRect(); return [Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)), Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))];};
canvas.addEventListener('pointerdown', event => {
  if (!photo || uploading || event.button !== 0) return;
  if (strokes.length >= 2000) {info.textContent = 'Drawing limit reached. Accept a result or clear marks.'; return;}
  event.preventDefault(); canvas.setPointerCapture(event.pointerId); viewport.focus({preventScroll:true});
  hidden = false; q('.paint-hide').textContent = 'Hide marks'; q('.paint-hide').setAttribute('aria-pressed','false');
  active = {tool, size:brushSize(), color:q('.paint-color').value, points:[point(event)]}; render();
});
canvas.addEventListener('pointermove', event => {
  if (!active) return;
  event.preventDefault();
  const next = point(event), last = active.points[active.points.length - 1];
  if (Math.hypot((next[0]-last[0])*canvas.width,(next[1]-last[1])*canvas.height) < 1) return;
  if (active.points.length < 10000) active.points.push(next);
  render();
});
function finish() {if(active) {strokes.push(active); active=null; redo=[]; render();}}
canvas.addEventListener('pointerup', finish);
canvas.addEventListener('pointercancel', finish);
canvas.addEventListener('lostpointercapture', finish);
function undoStroke() {finish(); if (strokes.length) {redo.push(strokes.pop()); render();}}
function redoStroke() {if (redo.length) {strokes.push(redo.pop()); render();}}
q('.paint-undo').addEventListener('click', undoStroke);
q('.paint-redo').addEventListener('click', redoStroke);
q('.paint-clear').addEventListener('click', () => {strokes=[]; redo=[]; active=null; render();});
q('.paint-hide').addEventListener('click', event => {hidden=!hidden; event.target.textContent=hidden?'Show marks':'Hide marks';event.target.setAttribute('aria-pressed',String(hidden));render();});
q('.paint-zoom').addEventListener('change', event => {zoom=Number(event.target.value);fit();});
viewport.addEventListener('keydown', event => {
  if (!(event.ctrlKey || event.metaKey)) return;
  if(event.key.toLowerCase()==='z') {event.preventDefault(); event.shiftKey?redoStroke():undoStroke();}
  if(event.key.toLowerCase()==='y') {event.preventDefault(); redoStroke();}
});
const observer = new ResizeObserver(fit); observer.observe(viewport);
host.getDrawing = () => {if (uploading) throw new Error('Wait for the image to finish loading.'); finish(); return {image:source,strokes:structuredClone(strokes)};};
watch('value', () => {if(props.value && props.value.load_image) loadImage(props.value.load_image);});
if (props.value && props.value.load_image) loadImage(props.value.load_image);
