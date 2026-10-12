const test = require('node:test');
const assert = require('node:assert/strict');
const {createController,bindDrag,draggedIds} = require('../recipe_sorter/frontend/sorting.js');
function fixture(){
 const state={selected:new Set([1,2]),clips:[{id:1,manual_revision:0},{id:2,manual_revision:2}],lastMove:{id:7}};
 const calls=[],messages=[];let refreshes=0;
 const controller=createController({getState:()=>state,api:async(...args)=>{calls.push(args);return {saved:true};},refresh:async()=>{refreshes++;},announce:m=>messages.push(m)});
 return {state,calls,messages,controller,get refreshes(){return refreshes;}};
}
test('a multi-selection is moved together, an unselected drag moves only itself',()=>{
 assert.deepEqual(draggedIds(1,new Set([1,2])),[1,2]);assert.deepEqual(draggedIds(3,new Set([1,2])),[3]);
});
test('accessible move submits revisions, clears selection after saving, refreshes and announces',async()=>{
 const f=fixture();await f.controller.move([1,2],4);
 assert.deepEqual(f.calls[0],['/api/assign','POST',{clip_ids:[1,2],group_id:4,expected_revisions:{1:0,2:2}}]);
 assert.equal(f.state.selected.size,0);assert.equal(f.refreshes,1);assert.match(f.messages[0],/2 clips moved/);
});
test('persistent undo uses the move returned by the library after restart',async()=>{
 const f=fixture();await f.controller.undo();assert.deepEqual(f.calls[0],['/api/moves/7/undo','POST']);assert.equal(f.refreshes,1);
});
test('failed save retains selection and never reports success',async()=>{
 const state={selected:new Set([1]),clips:[{id:1,manual_revision:1}]};let announced=false;
 const c=createController({getState:()=>state,api:async()=>{throw Error('newer move');},refresh:async()=>{},announce:()=>announced=true});
 await assert.rejects(c.move([1],2),/newer move/);assert.equal(state.selected.size,1);assert.equal(announced,false);assert.equal(c.saving,false);
});
test('duplicate interactions while saving are suppressed',async()=>{
 const f=fixture();let release;let calls=0;
 const c=createController({getState:()=>f.state,api:()=>{calls++;return new Promise(r=>release=r);},refresh:async()=>{},announce:()=>{}});
 const first=c.move([1],3);assert.equal(await c.move([1],4),false);release();await first;assert.equal(calls,1);
});
test('accept, reject and skip each persist their own decision',async()=>{
 const f=fixture();for(const decision of ['accepted','rejected','skipped'])await f.controller.decide(12,decision);
 assert.deepEqual(f.calls.map(c=>c[2].decision),['accepted','rejected','skipped']);assert.equal(f.refreshes,3);
});
function classList(){const s=new Set();return {add:x=>s.add(x),remove:x=>s.delete(x),contains:x=>s.has(x)};}
function dragRoot(){const handlers={};return {handlers,body:{classList:classList()},addEventListener:(type,fn)=>handlers[type]=fn};}
function node(attrs){return {dataset:attrs,classList:classList(),contains:x=>false,closest(selector){
 if(selector==='[data-drag-clip]'&&this.dataset.dragClip)return this;
 if(selector==='[data-drop-group]'&&this.dataset.dropGroup)return this;
 return null;
}};}
function event(target){return {target,relatedTarget:null,prevented:false,preventDefault(){this.prevented=true;},dataTransfer:{values:{},setData(type,data){this.values[type]=data;}}};}
test('drag handlers highlight a sidebar/group target and drop the whole selection',async()=>{
 const f=fixture(),root=dragRoot();let error;
 bindDrag(root,{getState:()=>f.state,controller:f.controller,announce:()=>{},onError:e=>error=e});
 const start=event(node({dragClip:'1'}));root.handlers.dragstart(start);
 assert.equal(root.body.classList.contains('is-dragging'),true);
 assert.deepEqual(JSON.parse(start.dataTransfer.values['application/x-recipe-clips']),[1,2]);
 const group=node({dropGroup:'4'}),over=event(group);root.handlers.dragover(over);
 assert.equal(over.prevented,true);assert.equal(group.classList.contains('drop-ready'),true);
 await root.handlers.drop(event(group));assert.equal(f.calls[0][2].group_id,4);assert.deepEqual(f.calls[0][2].clip_ids,[1,2]);
 assert.equal(group.classList.contains('drop-ready'),false);assert.equal(root.body.classList.contains('is-dragging'),false);assert.equal(error,undefined);
});
test('drop onto Uncertain is a normal saved move',async()=>{
 const f=fixture(),root=dragRoot();bindDrag(root,{getState:()=>f.state,controller:f.controller,announce:()=>{},onError:e=>{throw e;}});
 root.handlers.dragstart(event(node({dragClip:'1'})));await root.handlers.drop(event(node({dropGroup:'1'})));
 assert.equal(f.calls[0][2].group_id,1);
});
test('external file drop is ignored',async()=>{
 const f=fixture(),root=dragRoot();bindDrag(root,{getState:()=>f.state,controller:f.controller,announce:()=>{},onError:()=>{}});
 await root.handlers.drop(event(node({dropGroup:'3'})));assert.equal(f.calls.length,0);
});
test('drag cancellation removes feedback and prevents a later accidental drop',async()=>{
 const f=fixture(),root=dragRoot();bindDrag(root,{getState:()=>f.state,controller:f.controller,announce:()=>{},onError:()=>{}});
 root.handlers.dragstart(event(node({dragClip:'1'})));root.handlers.dragend();
 await root.handlers.drop(event(node({dropGroup:'3'})));assert.equal(f.calls.length,0);assert.equal(root.body.classList.contains('is-dragging'),false);
});
test('same-group no-op retains selection and announces no change',async()=>{
 const f=fixture();const c=createController({getState:()=>f.state,api:async()=>({moved:0,move_id:null}),refresh:async()=>{},announce:m=>f.messages.push(m)});
 await c.move([1,2],3);assert.equal(f.state.selected.size,2);assert.match(f.messages[0],/Nothing changed/);
});
test('explicit same-group confirmation is separate from a drag move',async()=>{
 const f=fixture();await f.controller.move([1],3,true);assert.equal(f.calls[0][2].confirm,true);assert.match(f.messages[0],/confirmed/);
});
test('drag count and hidden selections are announced and shown in the native ghost',()=>{
 const f=fixture(),root=dragRoot();f.state.visibleIds=[1];let ghost;
 root.createElement=()=>({remove(){}});root.body.append=e=>ghost=e;
 bindDrag(root,{getState:()=>f.state,controller:f.controller,announce:m=>f.messages.push(m),onError:()=>{}});
 const start=event(node({dragClip:'1'}));start.dataTransfer.setDragImage=()=>{};root.handlers.dragstart(start);
 assert.match(f.messages[0],/2 clips.*1 hidden/);assert.equal(ghost.textContent,'2 clips · 1 hidden');root.handlers.dragend();
});
test('a deleted destination cannot receive a drop',async()=>{
 const f=fixture(),root=dragRoot();f.state.groups=[{id:1},{id:2}];bindDrag(root,{getState:()=>f.state,controller:f.controller,announce:()=>{},onError:()=>{}});
 root.handlers.dragstart(event(node({dragClip:'1'})));await root.handlers.drop(event(node({dropGroup:'4'})));assert.equal(f.calls.length,0);
});
test('refresh failure reports saved state without pretending the view succeeded',async()=>{
 const f=fixture();const c=createController({getState:()=>f.state,api:async()=>({moved:1}),refresh:async()=>{throw Error('network');},announce:m=>f.messages.push(m)});
 await assert.rejects(c.move([1],3),/change was saved.*Reload/);assert.equal(f.messages.length,0);
});
