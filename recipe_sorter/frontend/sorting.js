/* One controller for dragging, accessible moves, confirmation, and persistent undo. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.RCSSorting=api;})(typeof globalThis==='object'?globalThis:this,function(){
'use strict';
function draggedIds(id,selected){return selected.has(id)?[...selected]:[id];}
function revisions(ids,clips){const result={};for(const id of ids){const c=clips.find(c=>c.id===id);if(!c)throw Error('A selected clip is no longer in this view. Refresh and try again.');result[id]=c.manual_revision;}return result;}
function createController({api,getState,refresh,announce}){
 let saving=false;
 async function mutate(action,message){if(saving)return false;saving=true;try{const result=await action();try{await refresh();}catch(e){throw Error('The change was saved, but the view could not refresh. Reload before continuing.');}announce(typeof message==='function'?message(result):message);return true;}finally{saving=false;}}
 return {
  move(ids,groupId,confirm=false){if(!ids.length||!Number.isInteger(groupId)||groupId<1)return Promise.resolve(false);const state=getState(),expected=revisions(ids,state.clips);
   return mutate(async()=>{const body={clip_ids:ids,group_id:groupId,expected_revisions:expected};if(confirm)body.confirm=true;const result=(await api('/api/assign','POST',body))||{};if(result.moved!==0)state.selected.clear();return result;},r=>r.moved===0?'Already in this group. Nothing changed.':`${r.moved??ids.length} clip${(r.moved??ids.length)===1?'':'s'} ${confirm?'confirmed':'moved'}. Undo is available.`);},
  undo(){const move=getState().lastMove;if(!move)return Promise.resolve(false);return mutate(()=>api(`/api/moves/${move.id}/undo`,'POST'),'Move undone. Your earlier assignment is restored.');},
  decide(id,decision){if(!['accepted','rejected','skipped'].includes(decision))throw Error('Unknown review action');return mutate(()=>api(`/api/suggestions/${id}/decision`,'POST',{decision}),decision==='accepted'?'Match accepted and saved. Undo is available.':decision==='rejected'?'Rejected for this group. Your decision is saved.':'Skipped for now. Find it in Skipped.');},
  get saving(){return saving;}
 };
}
function bindDrag(root,{getState,controller,announce,onError}){
 let ids=[],target=null,ghost=null,pointer=null,scrollFrame=null;
 const win=root.defaultView;
 function clearTarget(){if(target)target.classList.remove('drop-ready');target=null;}
 function finish(){ids=[];clearTarget();root.body?.classList.remove('is-dragging');ghost?.remove();ghost=null;pointer=null;if(scrollFrame&&win)win.cancelAnimationFrame(scrollFrame);scrollFrame=null;}
 function scroll(){
  if(!ids.length||!pointer||!win)return;
  let el=pointer.target?.parentElement,scrolled=false;
  while(el&&el!==root.body){const style=win.getComputedStyle(el),rect=el.getBoundingClientRect();if(/auto|scroll/.test(style.overflowY)&&el.scrollHeight>el.clientHeight){const speed=pointer.y<rect.top+55?-16:pointer.y>rect.bottom-55?16:0;if(speed){el.scrollBy(0,speed);scrolled=true;break;}}el=el.parentElement;}
  if(!scrolled){const speed=pointer.y<70?-18:pointer.y>win.innerHeight-70?18:0;if(speed)win.scrollBy(0,speed);}
  scrollFrame=win.requestAnimationFrame(scroll);
 }
 root.addEventListener('dragstart',event=>{
  const handle=event.target.closest('[data-drag-clip]');if(!handle||event.target.closest('input,select'))return;
  const state=getState();ids=draggedIds(Number(handle.dataset.dragClip),state.selected);
  event.dataTransfer.setData('application/x-recipe-clips',JSON.stringify(ids));event.dataTransfer.effectAllowed='move';root.body?.classList.add('is-dragging');
  const hidden=state.visibleIds?ids.filter(id=>!state.visibleIds.includes(id)).length:0;
  const message=`Moving ${ids.length} clip${ids.length===1?'':'s'}${hidden?` (${hidden} hidden by filters)`:''}. Drop on a recipe group or Uncertain.`;
  if(root.createElement&&event.dataTransfer.setDragImage){ghost=root.createElement('div');ghost.className='drag-preview';ghost.textContent=`${ids.length} clip${ids.length===1?'':'s'}${hidden?` · ${hidden} hidden`:''}`;root.body.append(ghost);event.dataTransfer.setDragImage(ghost,20,20);}
  announce(message);
 });
 root.addEventListener('dragover',event=>{
  if(!ids.length)return;pointer={x:event.clientX,y:event.clientY,target:event.target};if(win&&!scrollFrame)scrollFrame=win.requestAnimationFrame(scroll);
  const zone=event.target.closest('[data-drop-group]');const groups=getState().groups;
  if(!zone||(groups&&!groups.some(g=>g.id===Number(zone.dataset.dropGroup)))){clearTarget();return;}
  event.preventDefault();event.dataTransfer.dropEffect='move';if(target!==zone){clearTarget();target=zone;target.classList.add('drop-ready');}
 });
 root.addEventListener('dragleave',event=>{if(target&&!target.contains(event.relatedTarget))clearTarget();});
 root.addEventListener('drop',async event=>{
  const zone=event.target.closest('[data-drop-group]'),moving=ids.slice(),groups=getState().groups;
  finish();if(!moving.length){if(event.dataTransfer?.files?.length)event.preventDefault();return;}
  event.preventDefault();if(!zone||(groups&&!groups.some(g=>g.id===Number(zone.dataset.dropGroup))))return;
  try{await controller.move(moving,Number(zone.dataset.dropGroup));}catch(e){onError(e);}
 });
 root.addEventListener('dragend',finish);
 return {cancel:finish};
}
return {draggedIds,revisions,createController,bindDrag};
});
