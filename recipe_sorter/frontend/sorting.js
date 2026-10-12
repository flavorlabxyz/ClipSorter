/* Shared interaction controller. Browser handlers and Node tests use this code. */
(function (root, factory) {
  const exports = factory();
  if (typeof module === 'object' && module.exports) module.exports = exports;
  else root.RCSSorting = exports;
})(typeof globalThis === 'object' ? globalThis : this, function () {
  'use strict';
  function draggedIds(id, selected) {
    return selected.has(id) ? [...selected] : [id];
  }
  function revisions(ids, clips) {
    const result = {};
    for (const id of ids) {
      const clip = clips.find(c => c.id === id);
      if (!clip) throw Error('A selected clip is no longer in this view. Refresh and try again.');
      result[id] = clip.manual_revision;
    }
    return result;
  }
  function createController({api, getState, refresh, announce}) {
    let saving = false;
    async function mutate(action, message) {
      if (saving) return false;
      saving = true;
      try {
        await action();
        await refresh();
        announce(message);
        return true;
      } finally { saving = false; }
    }
    return {
      move(ids, groupId) {
        if (!ids.length || !Number.isInteger(groupId)) return Promise.resolve(false);
        const snapshot = getState();
        const expected = revisions(ids, snapshot.clips);
        return mutate(async () => {
          await api('/api/assign', 'POST', {clip_ids: ids, group_id: groupId, expected_revisions: expected});
          snapshot.selected.clear();
        }, `${ids.length} clip${ids.length === 1 ? '' : 's'} moved. Undo is available.`);
      },
      undo() {
        const move = getState().lastMove;
        if (!move) return Promise.resolve(false);
        return mutate(() => api(`/api/moves/${move.id}/undo`, 'POST'), 'Move undone. Your earlier assignment is restored.');
      },
      decide(id, decision) {
        if (!['accepted', 'rejected', 'skipped'].includes(decision)) throw Error('Unknown review action');
        return mutate(() => api(`/api/suggestions/${id}/decision`, 'POST', {decision}),
          decision === 'accepted' ? 'Match accepted and saved. Undo is available.' : decision === 'rejected' ? 'Rejected for this group. Your decision is saved.' : 'Skipped for now. Find it in Skipped.');
      },
      get saving() { return saving; }
    };
  }
  function bindDrag(root, {getState, controller, announce, onError}) {
    let ids = [], target = null;
    const clearTarget = () => { if (target) target.classList.remove('drop-ready'); target = null; };
    root.addEventListener('dragstart', event => {
      const card = event.target.closest('[data-drag-clip]');
      if (!card || event.target.closest('input,select')) return;
      ids = draggedIds(Number(card.dataset.dragClip), getState().selected);
      event.dataTransfer.setData('application/x-recipe-clips', JSON.stringify(ids));
      event.dataTransfer.effectAllowed = 'move';
      root.body?.classList.add('is-dragging');
      announce(`Moving ${ids.length} clip${ids.length === 1 ? '' : 's'}. Drop on a group or Uncertain.`);
    });
    root.addEventListener('dragover', event => {
      if (!ids.length) return;
      const zone = event.target.closest('[data-drop-group]');
      if (!zone) { clearTarget(); return; }
      event.preventDefault(); event.dataTransfer.dropEffect = 'move';
      if (target !== zone) { clearTarget(); target = zone; target.classList.add('drop-ready'); }
    });
    root.addEventListener('dragleave', event => {
      if (target && !target.contains(event.relatedTarget)) clearTarget();
    });
    root.addEventListener('drop', async event => {
      const zone = event.target.closest('[data-drop-group]');
      const moving = ids.slice();
      clearTarget(); ids = []; root.body?.classList.remove('is-dragging');
      // Ignore external file drops. No paths are ingested or moved.
      if (!zone || !moving.length) return;
      event.preventDefault();
      try { await controller.move(moving, Number(zone.dataset.dropGroup)); }
      catch (e) { onError(e); }
    });
    root.addEventListener('dragend', () => { ids = []; clearTarget(); root.body?.classList.remove('is-dragging'); });
  }
  return {draggedIds, revisions, createController, bindDrag};
});
