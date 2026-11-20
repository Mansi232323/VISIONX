import React, { useEffect, useState } from 'react';
import { Camera, CameraOff, Moon, Sun } from 'lucide-react';
import { AppTab } from './types/vision';
import { VisionProvider, useVision } from './vision/VisionContext';
import { OverviewView } from './components/OverviewView';
import { AirStudioView } from './components/AirStudioView';
import { YouTubeControlView } from './components/YouTubeControlView';
import { VirtualSearchView } from './components/VirtualSearchView';
import { MLLabView } from './components/MLLabView';
import { OfflineIndicator, PWAInstallControl } from './components/PWAInstallControl';

const NAV_ITEMS: Array<{ id: AppTab; label: string }> = [
  { id: 'overview', label: 'Overview' },
  { id: 'air-studio', label: 'Air Studio' },
  { id: 'youtube', label: 'YouTube' },
  { id: 'virtual-search', label: 'Virtual Search' },
  { id: 'ml-lab', label: 'ML Lab' },
];

const VisionXShell: React.FC = () => {
  const [activeTab, setActiveTab] = useState<AppTab>('overview');
  const [darkMode, setDarkMode] = useState(true);

  const { cameraActive, modelsLoading, startCamera, stopCamera } = useVision();

  useEffect(() => {
    const root = document.documentElement;
    if (darkMode) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [darkMode]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-[#07090E] dark:text-slate-100 flex flex-col">
      {/* Strict 3-Zone Top Bar Contract */}
      <header className="sticky top-0 z-30 border-b border-slate-200 dark:border-slate-800 bg-white/95 dark:bg-[#090C12]/95 backdrop-blur-sm">
        <div className="mx-auto flex max-w-[1440px] items-center justify-between gap-8 px-6 py-3.5">
          {/* Zone 1: Single-element Brand Wordmark */}
          <button
            type="button"
            onClick={() => setActiveTab('overview')}
            className="font-display text-lg font-bold tracking-tight text-slate-900 dark:text-white whitespace-nowrap shrink-0 cursor-pointer"
          >
            VisionX
          </button>

          {/* Zone 2: 5 concise single-line text nav links */}
          <nav
            aria-label="Primary"
            className="flex items-center gap-6 overflow-x-auto text-sm font-medium"
          >
            {NAV_ITEMS.map((item) => {
              const active = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setActiveTab(item.id)}
                  className={`relative py-1 transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                    active
                      ? 'text-cyan-600 dark:text-cyan-400'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100'
                  }`}
                >
                  {item.label}
                  {active && (
                    <span className="absolute inset-x-0 -bottom-3.5 h-0.5 bg-cyan-500" />
                  )}
                </button>
              );
            })}
          </nav>

          {/* Zone 3: 1 Primary Action */}
          <div className="shrink-0">
            <button
              type="button"
              onClick={() => (cameraActive ? stopCamera() : void startCamera())}
              disabled={modelsLoading}
              className={`inline-flex items-center gap-2 rounded-md px-4 py-2 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:opacity-60 ${
                cameraActive
                  ? 'border border-rose-500/50 bg-rose-500/15 text-rose-500 hover:bg-rose-500/25'
                  : 'bg-cyan-500 text-slate-950 hover:bg-cyan-400'
              }`}
            >
              {cameraActive ? (
                <>
                  <CameraOff className="h-3.5 w-3.5" />
                  Stop Camera
                </>
              ) : (
                <>
                  <Camera className="h-3.5 w-3.5" />
                  {modelsLoading ? 'Loading Models…' : 'Start Camera'}
                </>
              )}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Stage */}
      <main className="mx-auto w-full max-w-[1440px] flex-1 px-6 py-6">
        {activeTab === 'overview' && <OverviewView />}
        {activeTab === 'air-studio' && <AirStudioView />}
        {activeTab === 'youtube' && <YouTubeControlView />}
        {activeTab === 'virtual-search' && <VirtualSearchView />}
        {activeTab === 'ml-lab' && <MLLabView />}
      </main>

      {/* Quiet Laboratory Footer */}
      <footer className="border-t border-slate-200 dark:border-slate-800/80 px-6 py-3.5">
        <div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-4 text-xs text-slate-500 dark:text-slate-400">
          <span>
            VisionX Local Computer Vision & ML Workspace · On-device MediaPipe inference
          </span>
          <div className="flex items-center gap-2.5">
            <PWAInstallControl />
            <button
              type="button"
              onClick={() => setDarkMode((d) => !d)}
              className="inline-flex items-center gap-1.5 rounded border border-slate-200 dark:border-slate-800 px-2.5 py-1 text-xs text-slate-600 dark:text-slate-300 hover:border-slate-400 dark:hover:border-slate-600 cursor-pointer"
            >
              {darkMode ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
              {darkMode ? 'Light Lab Theme' : 'Obsidian Dark Theme'}
            </button>
          </div>
        </div>
      </footer>
      <OfflineIndicator />
    </div>
  );
};

export default function App() {
  return (
    <VisionProvider>
      <VisionXShell />
    </VisionProvider>
  );
}
