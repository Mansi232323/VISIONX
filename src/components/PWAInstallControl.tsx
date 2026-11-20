import React, { useState } from 'react';
import { Check, Copy, Download, Smartphone, WifiOff, X } from 'lucide-react';
import { useOnlineStatus, usePWAInstall } from '../hooks/usePWAInstall';

export const PWAInstallControl: React.FC = () => {
  const { isInstallable, isInstalled, isIOS, install } = usePWAInstall();
  const [showGuide, setShowGuide] = useState(false);
  const [copiedCmd, setCopiedCmd] = useState(false);

  const apkBuildCommands = [
    '# Option 1: Build signed Android APK via Bubblewrap (from Shared App URL)',
    'npm install -g @bubblewrap/cli',
    `bubblewrap init --manifest="${typeof window !== 'undefined' ? window.location.origin : ''}/manifest.webmanifest"`,
    'bubblewrap build',
  ].join('\n');

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(apkBuildCommands);
      setCopiedCmd(true);
      setTimeout(() => setCopiedCmd(false), 1800);
    } catch {
      // ignore clipboard errors
    }
  };

  if (isInstalled) {
    return null;
  }

  return (
    <>
      <div className="flex items-center gap-2">
        {isInstallable ? (
          <button
            type="button"
            onClick={() => void install()}
            className="inline-flex items-center gap-1.5 rounded bg-cyan-500 px-3 py-1 text-xs font-semibold text-slate-950 hover:bg-cyan-400 cursor-pointer whitespace-nowrap shrink-0"
          >
            <Download className="h-3.5 w-3.5" />
            Install Android / Desktop App
          </button>
        ) : (
          <button
            type="button"
            onClick={() => setShowGuide(true)}
            className="inline-flex items-center gap-1.5 rounded border border-slate-300 dark:border-slate-700 px-2.5 py-1 text-xs text-slate-700 dark:text-slate-200 hover:border-cyan-500 cursor-pointer whitespace-nowrap shrink-0"
          >
            <Smartphone className="h-3.5 w-3.5 text-cyan-500" />
            {isIOS ? 'Install on iOS' : 'Android Install / APK'}
          </button>
        )}
      </div>

      {showGuide && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="android-install-title"
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
        >
          <div className="w-full max-w-lg rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-5 shadow-xl">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-3">
              <h2
                id="android-install-title"
                className="font-display text-base font-semibold text-slate-900 dark:text-white"
              >
                Install VisionX on Android / Build APK
              </h2>
              <button
                type="button"
                onClick={() => setShowGuide(false)}
                aria-label="Close install guide"
                className="rounded p-1 text-slate-400 hover:text-slate-200 cursor-pointer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="mt-4 space-y-4 text-xs leading-relaxed text-slate-600 dark:text-slate-300">
              {isIOS ? (
                <div className="rounded-md border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-[#090C12] p-3">
                  <div className="font-semibold text-slate-900 dark:text-slate-100">
                    iPhone / iPad Installation
                  </div>
                  <p className="mt-1">
                    1. Tap the <strong>Share</strong> icon in Safari.<br />
                    2. Scroll down and select <strong>Add to Home Screen</strong>.
                  </p>
                </div>
              ) : (
                <div className="rounded-md border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-[#090C12] p-3">
                  <div className="font-semibold text-slate-900 dark:text-slate-100">
                    1. Instant Android Standalone Install (No Cable Needed)
                  </div>
                  <p className="mt-1">
                    VisionX is configured as a standalone offline-ready PWA with cached MediaPipe WASM and model weights. Open the Shared App URL in <strong>Chrome on Android</strong>, tap the <strong>⋮ menu</strong>, and choose <strong>Install app</strong> (or <strong>Add to Home screen</strong>) to run it full-screen with camera access.
                  </p>
                </div>
              )}

              <div className="rounded-md border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-[#090C12] p-3">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-slate-900 dark:text-slate-100">
                    2. Package a Signed .apk Locally (Bubblewrap TWA)
                  </span>
                  <button
                    type="button"
                    onClick={() => void handleCopy()}
                    className="inline-flex items-center gap-1 rounded border border-slate-700 px-2 py-0.5 text-[11px] text-slate-300 hover:border-cyan-500 cursor-pointer"
                  >
                    {copiedCmd ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
                    {copiedCmd ? 'Copied' : 'Copy'}
                  </button>
                </div>
                <pre className="mt-2 overflow-x-auto rounded bg-black/60 p-2.5 font-mono text-[11px] text-cyan-300">
                  {apkBuildCommands}
                </pre>
              </div>
            </div>

            <div className="mt-4 flex justify-end">
              <button
                type="button"
                onClick={() => setShowGuide(false)}
                className="rounded-md bg-cyan-500 px-4 py-1.5 text-xs font-semibold text-slate-950 hover:bg-cyan-400 cursor-pointer"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export const OfflineIndicator: React.FC = () => {
  const isOnline = useOnlineStatus();
  if (isOnline) return null;

  return (
    <div className="fixed bottom-4 left-4 z-50 flex items-center gap-2 rounded-md border border-amber-500/40 bg-amber-600 px-3 py-1.5 text-xs font-medium text-white shadow-lg">
      <WifiOff className="h-3.5 w-3.5" />
      <span>Offline Mode — Cached CV models active</span>
    </div>
  );
};
