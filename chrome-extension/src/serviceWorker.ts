/**
 * MV3 service worker — minimal.
 *
 * Since the switch to direct localhost fetch, the worker no longer
 * routes server traffic (the side panel fetches `bibsync serve`
 * itself). Its only jobs now:
 *   1. Open the side panel when the toolbar icon is clicked.
 *   2. Register the side panel to open on action click after install.
 *
 * The worker is ephemeral (Chrome kills it when idle); it keeps no
 * state, which is fine — both handlers are stateless.
 */

chrome.action.onClicked.addListener(async (tab) => {
  if (tab.windowId !== undefined) {
    await chrome.sidePanel.open({ windowId: tab.windowId });
  }
});

chrome.runtime.onInstalled.addListener(() => {
  chrome.sidePanel
    .setPanelBehavior({ openPanelOnActionClick: true })
    .catch((e) => console.error("[bibsync] setPanelBehavior failed", e));
});
