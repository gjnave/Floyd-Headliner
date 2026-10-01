// Exercise the actual canvas event handlers without loading model weights.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const drawing = new Proxy({}, {get: () => () => {}});
function widget() {
  return {events:{}, style:{}, value:'', dataset:{},
    addEventListener(name, fn) {this.events[name] = fn;},
    setAttribute(name, value) {this[name] = value;},
    getContext() {return drawing;}, setPointerCapture() {}, focus() {},
    getBoundingClientRect() {return {left:0,top:0,width:100,height:100};}};
}
const nodes = Object.fromEntries(['.paint-canvas','.paint-viewport','.paint-info','.paint-size',
  '.paint-color','.paint-tool-status','.paint-undo','.paint-redo','.paint-upload',
  '.size-value','.paint-clear','.paint-hide','.paint-zoom','.paint-empty'].map(key => [key,widget()]));
nodes['.paint-size'].value = '20'; nodes['.paint-color'].value = '#42a5f5';
const buttons = ['add','remove','color','erase'].map(tool => Object.assign(widget(),{dataset:{tool}}));
const host = {};
const sandbox = vm.createContext({
  element:{closest:()=>host, querySelector:s=>nodes[s], querySelectorAll:()=>buttons},
  document:{createElement:()=>widget()},
  ResizeObserver:class{observe(){}}, watch(){}, trigger(){}, props:{}, structuredClone,
});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../inpaint_assets/canvas.js'),'utf8'),sandbox);
vm.runInContext("photo={width:100,height:100}; source='fixture'; canvas.width=canvas.height=100",sandbox);
function stroke() {
  nodes['.paint-canvas'].events.pointerdown({button:0,pointerId:1,clientX:50,clientY:50,preventDefault(){}});
  nodes['.paint-canvas'].events.pointerup();
  return host.getDrawing().strokes.at(-1);
}
assert.equal(nodes['.paint-color'].value,'#00e676');
buttons.find(b=>b.dataset.tool==='remove').events.click();
assert.equal(nodes['.paint-color'].disabled,true);
assert.equal(nodes['.paint-color'].value,'#ff2d55');
// Even a programmatic color event must not change a Remove stroke into Color.
nodes['.paint-color'].value='#6633ff'; nodes['.paint-color'].events.input();
assert.equal(stroke().tool,'remove');
assert.equal(nodes['.paint-color'].value,'#ff2d55');
buttons.find(b=>b.dataset.tool==='color').events.click();
assert.equal(nodes['.paint-color'].disabled,false);
assert.equal(nodes['.paint-color'].value,'#42a5f5');
nodes['.paint-color'].value='#6633ff'; nodes['.paint-color'].events.input();
assert.equal(stroke().tool,'color');
assert.equal(stroke().color,'#6633ff');
buttons.find(b=>b.dataset.tool==='add').events.click();
assert.equal(stroke().tool,'add');
assert.equal(nodes['.paint-color'].value,'#00e676');
assert.match(nodes['.paint-tool-status'].textContent,/ADD/);
buttons.find(b=>b.dataset.tool==='color').events.click();
assert.equal(nodes['.paint-color'].value,'#6633ff');
buttons.find(b=>b.dataset.tool==='erase').events.click();
assert.equal(nodes['.paint-color'].disabled,true);
assert.equal(stroke().tool,'erase');
// Symbols are distinct geometry, and only decorate the browser overlay.
for (const [tool, expected] of [['add',2],['remove',1],['color',0],['erase',0]]) {
  sandbox.markerTool = tool;
  assert.equal(vm.runInContext(`(() => {
    let lines=0;
    const ctx={beginPath(){},moveTo(){},lineTo(){lines++;},stroke(){}};
    drawToolMarkers(ctx,{tool:markerTool,size:20,points:[[.5,.5],[.51,.51]]});
    return lines;
  })()`,sandbox),expected);
}
assert.equal(host.getDrawing().image,'fixture');
assert.deepEqual(Object.keys(host.getDrawing()).sort(),['image','strokes']);
console.log('PASS: active swatches, remembered color, Add/Remove symbols, clean drawing payload.');
