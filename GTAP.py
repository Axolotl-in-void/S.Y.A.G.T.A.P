import os, random, time
import numpy as np
import pygame
from pydub import AudioSegment
from mutagen import File as MutagenFile
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live

MUSIC_FOLDER = os.path.dirname(os.path.abspath(__file__))
SUPPORTED_FORMATS = (".mp3", ".flac", ".ogg", ".wav", ".m4a")
NUM_BARS = 205
BAR_HEIGHT = 35
REFRESH_HZ = 30
VOLUME_STEP = 0.05
SMOOTH_FACTOR = 0.6
DECAY_FACTOR = 0.85
ASCII_HEADER = r"""
G.T.A.P 2.0
  ___    ____     __      ____ 
 / __)  (_  _)   / _\    (  _ \
( (_ \ _  )(  _ /    \ _  ) __/
 \___/(_)(__)(_)\_/\_/(_)(__)  
"""
console = Console()

if os.name == "nt":
    import msvcrt
    def kb_hit(): return msvcrt.kbhit()
    def kb_get(): return msvcrt.getch().decode("utf-8", errors="ignore").lower()
else:
    def kb_hit(): return False
    def kb_get(): return None

def get_audio_files(folder):
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(SUPPORTED_FORMATS)]

def load_audio_samples(path):
    audio = AudioSegment.from_file(path)
    sr = audio.frame_rate
    arr = np.array(audio.get_array_of_samples())
    if audio.channels > 1:
        arr = arr.reshape((-1, audio.channels)).mean(axis=1)
    return arr.astype(np.float32)/(2**(8*audio.sample_width-1)), sr

def compute_fft_bars(samples, sr, elapsed, num_bars, max_height, window_ms=100):
    window_size = int(sr * window_ms / 1000)
    start = int(max(0, elapsed * sr))
    end = min(start + window_size, len(samples))

    segment = samples[start:end]
    if len(segment) < 32:
        return [0] * num_bars

    segment = segment * np.hanning(len(segment))

    fft_vals = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(len(segment), 1 / sr)

    band_edges = np.logspace(np.log10(20), np.log10(8000), num_bars + 1)
    band_indices = np.searchsorted(freqs, band_edges)

    band_vals = []
    for i in range(num_bars):
        a, b = band_indices[i], band_indices[i + 1]
        if b <= a:
            a = max(0, a - 1)
            b = min(len(fft_vals), a + 2)
            
        rms = np.sqrt(np.mean(fft_vals[a:b] ** 2))
        band_vals.append(rms)

    peak = max(band_vals) if max(band_vals) > 0 else 1.0
    
    return [int(min(max_height, round(((v / peak) ** 0.4) * max_height))) for v in band_vals]



def render_vertical_bars_with_labels(heights, max_height):
    lines = []
    for row in range(max_height, 0, -1):
        line = ""
        for b in heights:
            if b >= row:
                ratio = b / max_height
                if ratio < 0.33:
                    color = "green"
                elif ratio < 0.66:
                    color = "yellow"
                else:
                    color = "red"
                line += f"[{color}]█[/]"
            else:
                line += " "
        lines.append(line)
    
    num_bars = len(heights)
    bass_end = num_bars // 3
    mid_end = 2 * num_bars // 3
    labels = ""
    for i in range(num_bars):
        if i < bass_end:
            labels += "B"
        elif i < mid_end:
            labels += "M"
        else:
            labels += "T"
    lines.append(labels)
    return "\n".join(lines)

def format_volume(vol, width=20):
    filled = int(round(vol*width))
    return f"[{'█'*filled}{'░'*(width-filled)}] {int(vol*100)}%"

def make_layout(track, nxt, elapsed, length, vol, paused, vis, shuffle, loop_mode):
    state = "⏸" if paused else "▶"
    loop_names = ["no loop", "loop one", "loop all"]
    
    layout = Layout()
    layout.split_column(
        Layout(Panel(ASCII_HEADER.strip(), title=f"mode: {'shuffle' if shuffle else 'normal'} | loop: {loop_names[loop_mode]}", style="bold yellow"), name="header", size=8),
        Layout(Panel(vis, title="visualizer", padding=(0, 1), style="bold white"), name="vis", ratio=3),
        Layout(name="footer", size=6)
    )
    
    progress_width = 30
    filled_prog = int(round((elapsed / length) * progress_width)) if length > 0 else 0
    prog_bar = f"[{'█' * filled_prog}{'░' * (progress_width - filled_prog)}]"

    layout["footer"].split_row(
        Layout(Panel(f"{state} [bold cyan]{track}[/bold cyan]\n{prog_bar} {int(elapsed)}/{int(length)} sec", title="current track"), ratio=2),
        Layout(name="footer_right", ratio=1)
    )
    
    layout["footer_right"].split_column(
        Layout(Panel(f"[green]volume:[/green] {format_volume(vol)}", title="volume")),
        Layout(Panel(f"next: [magenta]{nxt}[/magenta]", title="next"))
    )

    return layout

def main():
    pygame.mixer.init()
    files = get_audio_files(MUSIC_FOLDER)
    if not files:
        console.print(f"[red]FAULT 1: NO MUSIC FOUND IN {MUSIC_FOLDER}[/red]")
        return

    volume = 0.8
    shuffle = False
    loop_mode = 0
    current_index = 0

    def get_next_index(idx):
        if shuffle:
            return random.randint(0, len(files)-1)
        else:
            return (idx + 1) % len(files)

    def get_prev_index(idx):
        if shuffle:
            return random.randint(0, len(files)-1)
        else:
            return (idx - 1) % len(files)

    while True:
        track_path = files[current_index]
        track_name = os.path.basename(track_path)
        nxt_name = os.path.basename(files[get_next_index(current_index)]) if len(files)>1 else "None"

        try:
            samples, sr = load_audio_samples(track_path)
        except Exception as e:
            console.print(f"[red]FAULT 2: FAILED TO LOAD {track_path}: {e}[/red]")
            current_index = get_next_index(current_index)
            continue

        meta = MutagenFile(track_path)
        length = meta.info.length if meta and meta.info else 0

        pygame.mixer.quit()
        pygame.mixer.init(frequency=sr)
        pygame.mixer.music.load(track_path)
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play()

        paused = False
        paused_time = 0
        bars = [0]*NUM_BARS
        refresh_delay = 1/REFRESH_HZ

        with Live(make_layout(track_name, nxt_name, 0, length, volume, paused, render_vertical_bars_with_labels(bars, BAR_HEIGHT), shuffle, loop_mode),
                  refresh_per_second=REFRESH_HZ, console=console) as live:
            while True:
                tw, th = console.size
                dynamic_num_bars = max(8, tw - 4)
                dynamic_bar_height = max(4, th - (8 + 6 + 3 + 5))
                
                if paused:
                    elapsed = paused_time
                else:
                    elapsed = pygame.mixer.music.get_pos() / 1000

                new_bars = compute_fft_bars(samples, sr, elapsed, dynamic_num_bars, dynamic_bar_height)
                bars = [
                    int(old * SMOOTH_FACTOR + new * (1 - SMOOTH_FACTOR)) if new >= old
                    else int(old * DECAY_FACTOR + new * (1 - DECAY_FACTOR))
                    for old, new in zip(bars, new_bars)
                ]
                
                if len(bars) != dynamic_num_bars:
                    bars = new_bars

                vis = render_vertical_bars_with_labels(bars, dynamic_bar_height)
                live.update(make_layout(track_name, nxt_name, elapsed, length, volume, paused, vis, shuffle, loop_mode))

                if not pygame.mixer.music.get_busy() and not paused:
                    if loop_mode == 1:
                        pygame.mixer.music.play()
                        continue
                    elif loop_mode == 2:
                        current_index = get_next_index(current_index)
                        break
                    else:
                        if current_index + 1 >= len(files):
                            console.clear()
                            console.print("[magenta]played all[/magenta]")
                            return
                        else:
                            current_index = get_next_index(current_index)
                            break

                if kb_hit():
                    key = kb_get()
                    if key == " ":
                        if not paused:
                            pygame.mixer.music.pause()
                            paused = True
                            paused_time = pygame.mixer.music.get_pos() / 1000
                        else:
                            pygame.mixer.music.unpause()
                            paused = False
                    elif key == "n":
                        pygame.mixer.music.stop()
                        current_index = get_next_index(current_index)
                        break
                    elif key == "p":
                        pygame.mixer.music.stop()
                        current_index = get_prev_index(current_index)
                        break
                    elif key == "s":
                        shuffle = not shuffle
                    elif key == "l":
                        loop_mode = (loop_mode + 1) % 3
                    elif key == "+":
                        volume = min(1.0, volume + VOLUME_STEP)
                        pygame.mixer.music.set_volume(volume)
                    elif key == "-":
                        volume = max(0.0, volume - VOLUME_STEP)
                        pygame.mixer.music.set_volume(volume)
                    elif key == "q":
                        pygame.mixer.music.stop()
                        console.clear()
                        console.print("[red]quit[/red]")
                        return

                time.sleep(refresh_delay)

if __name__ == "__main__":
    main()
