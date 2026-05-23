"""
G.T.A.P 2.0 - Terminal Audio Player with YouTube support
Fixed version: non-blocking input, stable visualizer, proper time tracking
"""

import os, random, time, subprocess, json, sys, threading, tempfile, signal
import numpy as np
import pygame
from pydub import AudioSegment
from mutagen import File as MutagenFile
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
import tty, termios, select

SUPPORTED_FORMATS = (".mp3", ".flac", ".ogg", ".wav", ".m4a")
BAR_HEIGHT = 30
REFRESH_HZ = 24
VOLUME_STEP = 0.05
SMOOTH_FACTOR = 0.55
DECAY_FACTOR = 0.80

ASCII_HEADER = r"""
S.Y.A.G.T.A.P 2.0
 ____  _  _  __    ___  ____  __   ____
/ ___)( \/ )/ _\  / __)(_  _)/ _\ (  _ \
\___ \ )  //    \( (_ \  )( /    \ ) __/
(____/(__/ \_/\_/ \___/ (__)\_/\_/(__)
"""

console = Console()

# ─── Thread-safe keyboard reader ─────────────────────────────────────────────

class KeyReader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._key = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._active = threading.Event()
        self._fd = sys.stdin.fileno()
        self._normal_settings = termios.tcgetattr(self._fd)

    def run(self):
        while not self._stop.is_set():
            if not self._active.wait(timeout=0.1):
                continue
            try:
                tty.setcbreak(self._fd)
            except Exception:
                pass
            while self._active.is_set() and not self._stop.is_set():
                try:
                    r, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if r:
                        ch = sys.stdin.read(1)
                        if ch:
                            with self._lock:
                                self._key = ch.lower()
                except Exception:
                    break
            self._restore()

    def _restore(self):
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._normal_settings)
        except Exception:
            pass

    def activate(self):
        with self._lock:
            self._key = None
        self._active.set()

    def deactivate(self):
        self._active.clear()
        time.sleep(0.12)

    def get(self):
        with self._lock:
            k = self._key
            self._key = None
            return k

    def stop(self):
        self._stop.set()
        self._active.clear()
        self._restore()


# ─── Audio loading ────────────────────────────────────────────────────────────

def load_audio_samples(path):
    audio = AudioSegment.from_file(path)
    sr = audio.frame_rate
    arr = np.array(audio.get_array_of_samples())
    if audio.channels > 1:
        arr = arr.reshape((-1, audio.channels)).mean(axis=1)
    max_val = float(2 ** (8 * audio.sample_width - 1))
    return arr.astype(np.float32) / max_val, sr


# ─── Visualizer ───────────────────────────────────────────────────────────────

def compute_fft_bars(samples, sr, elapsed, num_bars, max_height, window_ms=80):
    window_size = int(sr * window_ms / 1000)
    start = int(max(0.0, elapsed) * sr)
    end = min(start + window_size, len(samples))
    segment = samples[start:end]

    if len(segment) < 32:
        return [0] * num_bars

    win = np.hanning(len(segment))
    fft_vals = np.abs(np.fft.rfft(segment * win))
    freqs = np.fft.rfftfreq(len(segment), 1.0 / sr)

    band_edges = np.logspace(np.log10(40), np.log10(16000), num_bars + 1)
    band_indices = np.searchsorted(freqs, band_edges)

    band_vals = []
    for i in range(num_bars):
        a = band_indices[i]
        b = band_indices[i + 1]
        if b <= a:
            b = a + 1
        b = min(b, len(fft_vals))
        a = min(a, len(fft_vals) - 1)
        chunk = fft_vals[a:b]
        rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0
        band_vals.append(rms)

    peak = max(band_vals) if any(v > 0 for v in band_vals) else 1.0
    result = []
    for v in band_vals:
        norm = (v / peak) ** 0.45
        h = int(round(norm * max_height))
        result.append(max(0, min(max_height, h)))
    return result


def smooth_bars(old_bars, new_bars, num_bars):
    if len(old_bars) != num_bars:
        old_bars = new_bars[:]
    result = []
    for old, new in zip(old_bars, new_bars):
        if new >= old:
            result.append(int(old * SMOOTH_FACTOR + new * (1 - SMOOTH_FACTOR)))
        else:
            result.append(int(old * DECAY_FACTOR + new * (1 - DECAY_FACTOR)))
    return result


def render_bars(heights, max_height):
    num_bars = len(heights)
    lines = []
    for row in range(max_height, 0, -1):
        line_parts = []
        for b in heights:
            if b >= row:
                ratio = b / max_height
                if ratio < 0.33:
                    color = "green"
                elif ratio < 0.66:
                    color = "yellow"
                else:
                    color = "red"
                line_parts.append(f"[{color}]█[/]")
            else:
                line_parts.append(" ")
        lines.append("".join(line_parts))

    bass_end = num_bars // 3
    mid_end  = 2 * num_bars // 3
    label_line = "".join(
        "B" if i < bass_end else ("M" if i < mid_end else "T")
        for i in range(num_bars)
    )
    lines.append(f"[dim]{label_line}[/]")
    return "\n".join(lines)


# ─── UI layout ────────────────────────────────────────────────────────────────

def fmt_time(sec):
    sec = max(0, int(sec))
    return f"{sec // 60}:{sec % 60:02d}"

def fmt_vol(vol, width=18):
    filled = int(round(vol * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {int(vol * 100)}%"

def build_layout(track, nxt, elapsed, length, vol, paused, vis):
    state = "⏸ PAUSED" if paused else "▶ PLAYING"
    progress_width = 32
    filled = int(round((elapsed / length) * progress_width)) if length > 0 else 0
    prog_bar = f"[{'█' * filled}{'░' * (progress_width - filled)}] {fmt_time(elapsed)}/{fmt_time(length)}"

    controls = "[dim]SPC[/]=pause  [dim]+/-[/]=vol  [dim]f/b[/]=seek±10s  [dim]n[/]=next  [dim]q[/]=quit"

    layout = Layout()
    layout.split_column(
        Layout(
            Panel(
                ASCII_HEADER.strip() + "\n  loop: one",
                style="bold yellow",
            ),
            name="header",
            size=9,
        ),
        Layout(
            Panel(vis, title="[ visualizer ]", padding=(0, 1), style="bold white"),
            name="vis",
            ratio=3,
        ),
        Layout(name="footer", size=7),
    )

    layout["footer"].split_row(
        Layout(
            Panel(
                f"{state}\n[bold cyan]{track}[/bold cyan]\n{prog_bar}\n{controls}",
                title="now playing",
            ),
            ratio=3,
        ),
        Layout(name="footer_right", ratio=1),
    )

    layout["footer_right"].split_column(
        Layout(Panel(f"[green]{fmt_vol(vol)}[/green]", title="volume")),
        Layout(Panel(f"[magenta]{nxt}[/magenta]", title="next up")),
    )

    return layout


# ─── YouTube helpers ──────────────────────────────────────────────────────────

def search_youtube(query, n=7):
    try:
        cmd = ["yt-dlp", "--no-warnings", "--flat-playlist", "-j", f"ytsearch{n}:{query}"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode != 0:
            return []
        results = []
        for line in result.stdout.strip().splitlines():
            try:
                data = json.loads(line)
                vid_id = data.get("id", "")
                url = data.get("webpage_url") or data.get("url") or (
                    f"https://www.youtube.com/watch?v={vid_id}" if vid_id else ""
                )
                dur = int(data.get("duration") or 0)
                if url:
                    results.append({"title": data.get("title", "Unknown")[:72], "url": url, "duration": dur})
            except json.JSONDecodeError:
                pass
        return results
    except Exception as e:
        return []


def download_audio(url, temp_dir):
    try:
        out_tpl = os.path.join(temp_dir, "%(title)s.%(ext)s")
        cmd = [
            "yt-dlp", "--no-warnings",
            "-x", "--audio-format", "mp3", "--audio-quality", "192",
            "-o", out_tpl, url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            console.print(f"[red]yt-dlp error:[/red] {result.stderr[:300]}")
            return None
        audio_files = [
            f for f in os.listdir(temp_dir)
            if f.lower().endswith((".mp3", ".flac", ".ogg", ".wav", ".m4a"))
        ]
        if audio_files:
            audio_files.sort(key=lambda f: os.path.getmtime(os.path.join(temp_dir, f)), reverse=True)
            return os.path.join(temp_dir, audio_files[0])
        return None
    except Exception as e:
        console.print(f"[red]Download exception:[/red] {e}")
        return None


def search_interface():
    console.clear()
    console.print(Panel("[bold yellow]G.T.A.P 2.0[/bold yellow]  –  YouTube Audio Player", style="yellow"))
    console.print()
    query = console.input("[bold green]Search for a song / artist: [/]")
    if not query.strip():
        return None

    console.print("[yellow]🔍 Searching YouTube…[/]")
    results = search_youtube(query.strip())
    if not results:
        console.print("[red]No results found.[/red]")
        time.sleep(2)
        return None

    console.clear()
    console.print("[bold green]Search results[/bold green]\n")
    for i, r in enumerate(results, 1):
        mins, secs = divmod(r["duration"], 60)
        console.print(f"  [bold green]{i}[/bold green]  {r['title']:<72} [dim]{mins}:{secs:02d}[/dim]")

    console.print()
    choice = console.input("[bold green]Select (1-7, or q to quit): [/]")
    if choice.strip().lower() == "q":
        return None
    try:
        idx = int(choice.strip()) - 1
        if 0 <= idx < len(results):
            return results[idx]["url"]
    except ValueError:
        pass
    return None


# ─── Main player loop ─────────────────────────────────────────────────────────

def play_track(track_path, key_reader):
    """Load and play a single track. Returns 'quit', 'next', or 'done'."""

    console.print("[yellow]🎵 Loading audio data…[/]")
    try:
        samples, sr = load_audio_samples(track_path)
    except Exception as e:
        console.print(f"[red]Failed to load audio: {e}[/red]")
        return "done"

    meta = MutagenFile(track_path)
    length = meta.info.length if meta and meta.info else len(samples) / sr

    pygame.mixer.quit()
    pygame.mixer.init(frequency=sr, size=-16, channels=2, buffer=2048)
    pygame.mixer.music.load(track_path)

    volume = 0.8
    paused = False

    play_start_wall = time.monotonic()
    elapsed_at_pause = 0.0

    bars = [0] * 8
    tw, th = console.size
    num_bars = max(8, tw - 6)
    bar_height = max(4, th - (9 + 7 + 3 + 4))

    pygame.mixer.music.set_volume(volume)
    pygame.mixer.music.play()

    track_name = os.path.basename(track_path)
    nxt_name = "—"

    refresh_delay = 1.0 / REFRESH_HZ

    with Live(
        build_layout(track_name, nxt_name, 0, length, volume, paused,
                     render_bars(bars, bar_height)),
        refresh_per_second=REFRESH_HZ,
        console=console,
        screen=False,
    ) as live:
        while True:
            new_tw, new_th = console.size
            if new_tw != tw or new_th != th:
                tw, th = new_tw, new_th
                num_bars = max(8, tw - 6)
                bar_height = max(4, th - (9 + 7 + 3 + 4))
                bars = [0] * num_bars

            if paused:
                elapsed = elapsed_at_pause
            else:
                elapsed = elapsed_at_pause + (time.monotonic() - play_start_wall)
                elapsed = min(elapsed, length)

            new_bars = compute_fft_bars(samples, sr, elapsed, num_bars, bar_height)
            bars = smooth_bars(bars, new_bars, num_bars)
            vis = render_bars(bars, bar_height)

            live.update(
                build_layout(track_name, nxt_name, elapsed, length,
                             volume, paused, vis)
            )

            # Loop one: restart automatically when the track ends
            if not pygame.mixer.music.get_busy() and not paused:
                pygame.mixer.music.play()
                play_start_wall = time.monotonic()
                elapsed_at_pause = 0.0
                continue

            key = key_reader.get()

            if key == " ":
                if not paused:
                    pygame.mixer.music.pause()
                    elapsed_at_pause = elapsed_at_pause + (time.monotonic() - play_start_wall)
                    paused = True
                else:
                    pygame.mixer.music.unpause()
                    play_start_wall = time.monotonic()
                    paused = False

            elif key == "q":
                pygame.mixer.music.stop()
                return "quit"

            elif key == "n":
                pygame.mixer.music.stop()
                return "next"

            elif key in ("+", "="):
                volume = min(1.0, volume + VOLUME_STEP)
                pygame.mixer.music.set_volume(volume)

            elif key == "-":
                volume = max(0.0, volume - VOLUME_STEP)
                pygame.mixer.music.set_volume(volume)

            elif key == "f":
                if not paused:
                    elapsed_at_pause = min(length - 1, elapsed_at_pause + (time.monotonic() - play_start_wall) + 10)
                    play_start_wall = time.monotonic()
                    new_pos = elapsed_at_pause
                else:
                    elapsed_at_pause = min(length - 1, elapsed_at_pause + 10)
                    new_pos = elapsed_at_pause
                pygame.mixer.music.stop()
                try:
                    pygame.mixer.music.play(start=new_pos)
                    if paused:
                        pygame.mixer.music.pause()
                except Exception:
                    pygame.mixer.music.play()
                    elapsed_at_pause = 0.0
                    play_start_wall = time.monotonic()

            elif key == "b":
                if not paused:
                    elapsed_at_pause = max(0.0, elapsed_at_pause + (time.monotonic() - play_start_wall) - 10)
                    play_start_wall = time.monotonic()
                    new_pos = elapsed_at_pause
                else:
                    elapsed_at_pause = max(0.0, elapsed_at_pause - 10)
                    new_pos = elapsed_at_pause
                pygame.mixer.music.stop()
                try:
                    pygame.mixer.music.play(start=new_pos)
                    if paused:
                        pygame.mixer.music.pause()
                except Exception:
                    pygame.mixer.music.play()
                    elapsed_at_pause = 0.0
                    play_start_wall = time.monotonic()

            time.sleep(refresh_delay)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    pygame.mixer.pre_init(44100, -16, 2, 2048)
    pygame.mixer.init()

    temp_dir = tempfile.mkdtemp(prefix="syagtap_")
    key_reader = KeyReader()
    key_reader.start()

    def cleanup(sig=None, frame=None):
        key_reader.stop()
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        console.clear()
        sys.exit(0)

    signal.signal(signal.SIGINT,  cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        while True:
            url = search_interface()
            if not url:
                console.print("[red]Goodbye.[/red]")
                break

            console.print("[yellow]📥 Downloading audio…[/]")
            track_path = download_audio(url, temp_dir)

            if not track_path:
                console.print("[red]Download failed. Check yt-dlp is installed and try again.[/red]")
                time.sleep(3)
                continue

            console.print(f"[green]✓ Ready: {os.path.basename(track_path)}[/green]")
            time.sleep(0.6)

            key_reader.activate()
            result = play_track(track_path, key_reader)
            key_reader.deactivate()

            try:
                os.remove(track_path)
            except Exception:
                pass

            if result == "quit":
                console.print("\n[red]Quit.[/red]")
                break
            # result == "done" or "next" → loop back to search

    finally:
        key_reader.stop()
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
