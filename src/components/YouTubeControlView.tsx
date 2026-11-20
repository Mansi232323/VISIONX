import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ChevronDown,
  ChevronUp,
  FastForward,
  Maximize2,
  Minimize2,
  Pause,
  Play,
  Rewind,
  ShieldAlert,
  ShieldCheck,
  Volume2,
  VolumeX,
} from 'lucide-react';
import { useVision } from '../vision/VisionContext';
import { VisionViewport } from './VisionViewport';
import { YouTubeAction } from '../types/vision';

interface VideoItem {
  id: string;
  title: string;
  channel: string;
  duration: string;
  topic: string;
}

const CURATED_VIDEOS: VideoItem[] = [
  {
    id: 'aircAruvnKk',
    title: 'But what is a neural network? | Deep Learning Chapter 1',
    channel: '3Blue1Brown',
    duration: '19:13',
    topic: 'Deep Learning',
  },
  {
    id: 'OkmNXy7er84',
    title: 'Visualizing Transformers & Attention Mechanisms',
    channel: '3Blue1Brown',
    duration: '27:14',
    topic: 'AI Architecture',
  },
  {
    id: 'WXuK6gekU1Y',
    title: 'AlphaGo — The Movie (Full Documentary)',
    channel: 'Google DeepMind',
    duration: '1:30:27',
    topic: 'Reinforcement Learning',
  },
  {
    id: 'zjkBMFhNj_g',
    title: 'Intro to Large Language Models & Computer Vision Agents',
    channel: 'Andrej Karpathy',
    duration: '59:47',
    topic: 'Machine Learning',
  },
  {
    id: 'kYB8IZa5AuE',
    title: 'Linear transformations and matrices | Computer Vision Math',
    channel: '3Blue1Brown',
    duration: '10:58',
    topic: 'Linear Algebra',
  },
  {
    id: 'fNk_zzaMoSs',
    title: 'Vectors, Coordinate Geometry & Spatial Landmarks',
    channel: '3Blue1Brown',
    duration: '9:52',
    topic: 'Geometry',
  },
];

const ACTION_LABELS: Record<YouTubeAction, string> = {
  play_pause: 'Play / Pause Video',
  scroll_down: 'Scroll Feed Down',
  scroll_up: 'Scroll Feed Up',
  volume_up: 'Volume +15%',
  volume_down: 'Volume -15%',
  mute_toggle: 'Mute / Unmute',
  seek_forward: 'Seek Forward +10s',
  seek_backward: 'Seek Backward -10s',
  fullscreen_toggle: 'Toggle Theater Mode',
};

const DEFAULT_MAPPINGS: Record<string, YouTubeAction> = {
  'Open Palm': 'play_pause',
  'Peace Sign': 'scroll_down',
  'Pointing Index': 'scroll_up',
  'Thumbs Up': 'volume_up',
  'Thumbs Down': 'volume_down',
  'Closed Fist': 'mute_toggle',
  'Three Fingers': 'seek_forward',
  'Four Fingers': 'seek_backward',
  'OK Sign': 'fullscreen_toggle',
};

function parseYouTubeId(input: string): string | null {
  const trimmed = input.trim();
  if (/^[a-zA-Z0-9_-]{11}$/.test(trimmed)) return trimmed;
  try {
    const url = new URL(trimmed);
    if (url.searchParams.get('v')) return url.searchParams.get('v');
    if (url.hostname.includes('youtu.be')) return url.pathname.slice(1);
    const parts = url.pathname.split('/');
    const embedIdx = parts.indexOf('embed');
    if (embedIdx >= 0 && parts[embedIdx + 1]) return parts[embedIdx + 1];
  } catch {
    return null;
  }
  return null;
}

export const YouTubeControlView: React.FC = () => {
  const {
    hands,
    inputMode,
    cameraActive,
    simConfig,
    updateSimConfig,
    addLog,
  } = useVision();

  const [armed, setArmed] = useState(false);
  const [cooldownMs, setCooldownMs] = useState(900);
  const [mappings, setMappings] = useState<Record<string, YouTubeAction>>(DEFAULT_MAPPINGS);

  const [activeVideo, setActiveVideo] = useState<VideoItem>(CURATED_VIDEOS[0]);
  const [urlInput, setUrlInput] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(70);
  const [isMuted, setIsMuted] = useState(false);
  const [seekOffset, setSeekOffset] = useState(0);
  const [theaterMode, setTheaterMode] = useState(false);
  const [lastFeedback, setLastFeedback] = useState<string | null>(null);

  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  const feedRef = useRef<HTMLDivElement | null>(null);
  const lastTriggerRef = useRef(0);
  const prevMotionRef = useRef<string>('Idle');

  const sendYTCommand = useCallback((func: string, args: unknown[] = []) => {
    const win = iframeRef.current?.contentWindow;
    if (!win) return;
    win.postMessage(
      JSON.stringify({ event: 'command', func, args }),
      '*'
    );
  }, []);

  const executeAction = useCallback(
    (action: YouTubeAction, sourceLabel: string) => {
      const now = performance.now();
      if (now - lastTriggerRef.current < cooldownMs) return;
      lastTriggerRef.current = now;

      switch (action) {
        case 'play_pause':
          setIsPlaying((prev) => {
            const next = !prev;
            sendYTCommand(next ? 'playVideo' : 'pauseVideo');
            setLastFeedback(`${next ? 'Playing' : 'Paused'} (${sourceLabel})`);
            return next;
          });
          break;
        case 'scroll_down':
          feedRef.current?.scrollBy({ top: 180, behavior: 'smooth' });
          setLastFeedback(`Scrolled Feed Down (${sourceLabel})`);
          break;
        case 'scroll_up':
          feedRef.current?.scrollBy({ top: -180, behavior: 'smooth' });
          setLastFeedback(`Scrolled Feed Up (${sourceLabel})`);
          break;
        case 'volume_up':
          setVolume((v) => {
            const next = Math.min(100, v + 15);
            sendYTCommand('setVolume', [next]);
            setIsMuted(false);
            sendYTCommand('unMute');
            setLastFeedback(`Volume ${next}% (${sourceLabel})`);
            return next;
          });
          break;
        case 'volume_down':
          setVolume((v) => {
            const next = Math.max(0, v - 15);
            sendYTCommand('setVolume', [next]);
            setLastFeedback(`Volume ${next}% (${sourceLabel})`);
            return next;
          });
          break;
        case 'mute_toggle':
          setIsMuted((m) => {
            const next = !m;
            sendYTCommand(next ? 'mute' : 'unMute');
            setLastFeedback(`${next ? 'Muted' : 'Unmuted'} (${sourceLabel})`);
            return next;
          });
          break;
        case 'seek_forward':
          setSeekOffset((s) => {
            const next = s + 10;
            sendYTCommand('seekTo', [Math.max(0, next), true]);
            setLastFeedback(`Seek +10s (${sourceLabel})`);
            return next;
          });
          break;
        case 'seek_backward':
          setSeekOffset((s) => {
            const next = Math.max(0, s - 10);
            sendYTCommand('seekTo', [next, true]);
            setLastFeedback(`Seek -10s (${sourceLabel})`);
            return next;
          });
          break;
        case 'fullscreen_toggle':
          setTheaterMode((t) => {
            const next = !t;
            setLastFeedback(`${next ? 'Theater Expanded' : 'Standard View'} (${sourceLabel})`);
            return next;
          });
          break;
      }

      addLog('YouTube', ACTION_LABELS[action], `Triggered via ${sourceLabel}`);
    },
    [addLog, cooldownMs, sendYTCommand]
  );

  // Emergency Stop on Escape key
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && armed) {
        setArmed(false);
        setLastFeedback('Emergency Stop Activated (ESC)');
        addLog('YouTube', 'Emergency Stop', 'Touchless YouTube automation disarmed via ESC');
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [addLog, armed]);

  // Process live hand gestures + vertical swipes when armed
  useEffect(() => {
    if (!armed || (!cameraActive && inputMode !== 'simulation')) return;
    const hand = hands[0];
    if (!hand) return;

    // Two-hand Open Palm emergency stop
    if (
      hands.length === 2 &&
      hands[0].gesture === 'Open Palm' &&
      hands[1].gesture === 'Open Palm'
    ) {
      setArmed(false);
      setLastFeedback('Two-Hand Emergency Stop');
      return;
    }

    // Dynamic vertical swipe scrolling
    if (hand.motion !== prevMotionRef.current) {
      prevMotionRef.current = hand.motion;
      if (hand.motion === 'Swipe Up') {
        executeAction('scroll_up', 'Swipe Up');
        return;
      }
      if (hand.motion === 'Swipe Down') {
        executeAction('scroll_down', 'Swipe Down');
        return;
      }
    }

    const mapped = mappings[hand.gesture];
    if (mapped && hand.gestureConfidence >= 0.72) {
      executeAction(mapped, hand.gesture);
    }
  }, [armed, cameraActive, executeAction, hands, inputMode, mappings]);

  const handleLoadCustomUrl = (e: React.FormEvent) => {
    e.preventDefault();
    const id = parseYouTubeId(urlInput);
    if (!id) {
      setLastFeedback('Invalid YouTube URL or 11-character video ID');
      return;
    }
    setActiveVideo({
      id,
      title: `Custom YouTube Video (${id})`,
      channel: 'Loaded URL',
      duration: 'Live',
      topic: 'Custom',
    });
    setSeekOffset(0);
    setUrlInput('');
  };

  const primaryHand = hands[0];

  return (
    <div className="space-y-6">
      {/* Safety & Control Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] px-4 py-3">
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => setArmed((v) => !v)}
            className={`inline-flex items-center gap-2 rounded-md px-4 py-2 text-xs font-semibold transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
              armed
                ? 'bg-emerald-500 text-slate-950 hover:bg-emerald-400'
                : 'border border-slate-300 dark:border-slate-700 bg-slate-100 dark:bg-[#090C12] text-slate-800 dark:text-slate-200 hover:border-cyan-500'
            }`}
          >
            {armed ? <ShieldCheck className="h-4 w-4" /> : <ShieldAlert className="h-4 w-4" />}
            {armed ? 'Touchless Control Armed' : 'Arm Touchless Control'}
          </button>

          {armed && (
            <button
              type="button"
              onClick={() => {
                setArmed(false);
                setLastFeedback('Emergency Stop Engaged');
              }}
              className="rounded-md bg-rose-600 px-3 py-2 text-xs font-semibold text-white hover:bg-rose-500 cursor-pointer whitespace-nowrap shrink-0"
            >
              Emergency Stop (ESC)
            </button>
          )}

          <label className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-300">
            <span>Cooldown</span>
            <input
              type="range"
              min={400}
              max={2000}
              step={100}
              value={cooldownMs}
              onChange={(e) => setCooldownMs(Number(e.target.value))}
              className="w-24 accent-cyan-500"
            />
            <span className="font-mono tabular-nums">{cooldownMs}ms</span>
          </label>
        </div>

        <div className="text-xs font-mono text-slate-500 dark:text-slate-400">
          {lastFeedback ? (
            <span className="text-cyan-500">{lastFeedback}</span>
          ) : (
            <span>Hold gesture or swipe vertically to control player & feed</span>
          )}
        </div>
      </div>

      {/* Main Layout */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 items-start">
        {/* Video Stage + Scrollable Feed */}
        <div className={`${theaterMode ? 'lg:col-span-12' : 'lg:col-span-7'} space-y-4`}>
          <div className="overflow-hidden rounded-lg border border-slate-200 dark:border-slate-800 bg-[#090C12]">
            <div className="aspect-video w-full bg-black">
              <iframe
                ref={iframeRef}
                title={activeVideo.title}
                src={`https://www.youtube.com/embed/${activeVideo.id}?enablejsapi=1&rel=0`}
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                className="h-full w-full border-0"
              />
            </div>

            {/* Synchronized Player State & Touchless Bar */}
            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-800 bg-[#0E131D] px-4 py-3 text-xs text-slate-200">
              <div className="min-w-0 flex-1">
                <div className="truncate font-display font-semibold text-white">
                  {activeVideo.title}
                </div>
                <div className="mt-0.5 text-[11px] text-slate-400">
                  {activeVideo.channel} · {activeVideo.topic} · Offset {seekOffset}s
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => executeAction('seek_backward', 'Button')}
                  aria-label="Seek backward 10 seconds"
                  className="rounded border border-slate-700 p-1.5 hover:bg-slate-800 cursor-pointer"
                >
                  <Rewind className="h-3.5 w-3.5" />
                </button>
                <button
                  type="button"
                  onClick={() => executeAction('play_pause', 'Button')}
                  aria-label="Play or Pause"
                  className="rounded bg-cyan-500 p-1.5 text-slate-950 hover:bg-cyan-400 cursor-pointer"
                >
                  {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
                </button>
                <button
                  type="button"
                  onClick={() => executeAction('seek_forward', 'Button')}
                  aria-label="Seek forward 10 seconds"
                  className="rounded border border-slate-700 p-1.5 hover:bg-slate-800 cursor-pointer"
                >
                  <FastForward className="h-3.5 w-3.5" />
                </button>
                <button
                  type="button"
                  onClick={() => executeAction('mute_toggle', 'Button')}
                  aria-label="Toggle mute"
                  className="rounded border border-slate-700 p-1.5 hover:bg-slate-800 cursor-pointer"
                >
                  {isMuted ? <VolumeX className="h-3.5 w-3.5 text-rose-400" /> : <Volume2 className="h-3.5 w-3.5" />}
                </button>
                <span className="font-mono text-[11px] w-9 tabular-nums text-slate-300">
                  {isMuted ? '0%' : `${volume}%`}
                </span>
                <button
                  type="button"
                  onClick={() => executeAction('fullscreen_toggle', 'Button')}
                  aria-label="Toggle theater view"
                  className="rounded border border-slate-700 p-1.5 hover:bg-slate-800 cursor-pointer"
                >
                  {theaterMode ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
                </button>
              </div>
            </div>
          </div>

          {/* Load Any YouTube URL */}
          <form onSubmit={handleLoadCustomUrl} className="flex gap-2">
            <input
              type="text"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              placeholder="Paste any YouTube URL or Video ID to load in player…"
              className="min-w-0 flex-1 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-[#0E131D] px-3 py-2 text-xs text-slate-900 dark:text-slate-100 focus:border-cyan-500 focus:outline-none"
            />
            <button
              type="submit"
              className="rounded-md border border-slate-300 dark:border-slate-700 bg-slate-100 dark:bg-slate-800 px-4 py-2 text-xs font-medium text-slate-900 dark:text-slate-100 hover:border-cyan-500 cursor-pointer whitespace-nowrap shrink-0"
            >
              Load Video
            </button>
          </form>

          {/* Gesture-Scrollable YouTube Video Feed */}
          <section className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-4">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-2.5">
              <h2 className="font-display text-sm font-semibold text-slate-900 dark:text-slate-100">
                Scrollable Video Queue (Peace Sign = Scroll Down · Index = Scroll Up)
              </h2>
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => executeAction('scroll_up', 'Scroll Button')}
                  aria-label="Scroll feed up"
                  className="rounded border border-slate-200 dark:border-slate-700 p-1 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer"
                >
                  <ChevronUp className="h-3.5 w-3.5" />
                </button>
                <button
                  type="button"
                  onClick={() => executeAction('scroll_down', 'Scroll Button')}
                  aria-label="Scroll feed down"
                  className="rounded border border-slate-200 dark:border-slate-700 p-1 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer"
                >
                  <ChevronDown className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>

            <div
              ref={feedRef}
              className="mt-3 max-h-60 space-y-2 overflow-y-auto pr-1"
            >
              {CURATED_VIDEOS.map((vid) => {
                const selected = vid.id === activeVideo.id;
                return (
                  <button
                    key={vid.id}
                    type="button"
                    onClick={() => {
                      setActiveVideo(vid);
                      setSeekOffset(0);
                    }}
                    className={`flex w-full items-center justify-between gap-3 rounded-md border p-3 text-left transition-colors cursor-pointer ${
                      selected
                        ? 'border-cyan-500/60 bg-cyan-500/10'
                        : 'border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-[#090C12] hover:border-slate-300 dark:hover:border-slate-700'
                    }`}
                  >
                    <div className="min-w-0">
                      <div className="truncate text-xs font-semibold text-slate-900 dark:text-slate-100">
                        {vid.title}
                      </div>
                      <div className="mt-0.5 text-[11px] text-slate-500 dark:text-slate-400">
                        {vid.channel} · {vid.topic}
                      </div>
                    </div>
                    <span className="font-mono text-[11px] text-slate-400 tabular-nums shrink-0">
                      {vid.duration}
                    </span>
                  </button>
                );
              })}
            </div>
          </section>
        </div>

        {/* Right Column: Live Camera Feed + Gesture Mapping Table */}
        <div className={`${theaterMode ? 'lg:col-span-12 grid grid-cols-1 md:grid-cols-2 gap-6' : 'lg:col-span-5 space-y-4'}`}>
          <div className="space-y-3">
            <VisionViewport aspectClass="aspect-video" compact />
            <div className="flex items-center justify-between rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] px-3.5 py-2.5 text-xs">
              <span className="text-slate-500 dark:text-slate-400">Detected Pose</span>
              <span className="font-mono font-semibold text-cyan-600 dark:text-cyan-400">
                {primaryHand ? `${primaryHand.gesture} (${primaryHand.motion})` : 'No Hand'}
              </span>
            </div>

            {inputMode === 'simulation' && (
              <div className="rounded-lg border border-cyan-500/30 bg-cyan-950/15 p-3 text-xs">
                <div className="font-mono font-semibold text-cyan-400 mb-2">
                  Simulator Quick Gestures
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {Object.keys(DEFAULT_MAPPINGS).map((g) => (
                    <button
                      key={g}
                      type="button"
                      onClick={() => updateSimConfig({ gesture: g, shapeTrajectory: 'none' })}
                      className={`rounded border px-2 py-1 text-[11px] cursor-pointer ${
                        simConfig.gesture === g
                          ? 'border-cyan-500 bg-cyan-500/20 text-cyan-200'
                          : 'border-slate-700 text-slate-300 hover:bg-slate-800'
                      }`}
                    >
                      {g}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          <section className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-4">
            <h2 className="font-display text-sm font-semibold text-slate-900 dark:text-slate-100 border-b border-slate-100 dark:border-slate-800 pb-2.5">
              Gesture-to-YouTube Action Mapping
            </h2>
            <div className="mt-3 space-y-2">
              {Object.entries(mappings).map(([gesture, action]) => {
                const isActive = primaryHand?.gesture === gesture && armed;
                return (
                  <div
                    key={gesture}
                    className={`flex items-center justify-between gap-2 rounded px-2 py-1.5 text-xs ${
                      isActive ? 'bg-cyan-500/10 ring-1 ring-cyan-500/40' : ''
                    }`}
                  >
                    <span className="font-medium text-slate-700 dark:text-slate-300 truncate">
                      {gesture}
                    </span>
                    <select
                      value={action}
                      onChange={(e) =>
                        setMappings((prev) => ({
                          ...prev,
                          [gesture]: e.target.value as YouTubeAction,
                        }))
                      }
                      aria-label={`Action for ${gesture}`}
                      className="rounded border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-[#090C12] px-2 py-1 text-xs text-slate-800 dark:text-slate-200"
                    >
                      {(Object.keys(ACTION_LABELS) as YouTubeAction[]).map((act) => (
                        <option key={act} value={act}>
                          {ACTION_LABELS[act]}
                        </option>
                      ))}
                    </select>
                  </div>
                );
              })}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};
