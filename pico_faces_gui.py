"""Virtual LCD and button panel for the pico-faces Pico 2 firmware.

The PC provides only the controls, serial transport, and display. Image
generation remains on the RP2350.
"""

from __future__ import annotations

import argparse
import binascii
import queue
import random
import struct
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import serial
from PIL import Image, ImageTk
from serial.tools import list_ports


BAUD_RATE = 115_200
SERIAL_TIMEOUT_SECONDS = 120
MAX_IMAGE_BYTES = 4 * 1024 * 1024
CLASS_LABELS = (
    "Female · neutral",
    "Female · smiling",
    "Male · neutral",
    "Male · smiling",
    "Unconditional",
)
STEP_OPTIONS = (1, 2, 4, 8)
CFG_OPTIONS = ("Plain", "CFG 4", "CFG 6", "CFG 8")


@dataclass(frozen=True)
class GenerationRequest:
    port: str
    seed: int
    steps: int
    condition: int
    cfg: int


@dataclass(frozen=True)
class GenerationResult:
    seed: int
    width: int
    height: int
    channels: int
    condition: int
    pixels: bytes
    device_crc: int
    local_crc: int
    generation_ms: int

    @property
    def crc_matches(self) -> bool:
        return self.device_crc == self.local_crc


def read_exact(connection: serial.Serial, byte_count: int) -> bytes:
    """Read exactly byte_count bytes or raise a helpful timeout error."""
    chunks = bytearray()
    while len(chunks) < byte_count:
        part = connection.read(byte_count - len(chunks))
        if not part:
            raise TimeoutError(
                f"Timed out after receiving {len(chunks):,} of "
                f"{byte_count:,} expected bytes."
            )
        chunks.extend(part)
    return bytes(chunks)


def read_magic(connection: serial.Serial) -> bytes:
    """Synchronize on either supported four-byte response marker."""
    window = b""
    while window not in (b"RFIM", b"RFI2"):
        next_byte = connection.read(1)
        if not next_byte:
            raise TimeoutError("Timed out waiting for the Pico image response.")
        window = (window + next_byte)[-4:]
    return window


def generate_on_pico(request: GenerationRequest) -> GenerationResult:
    """Send one request and receive one complete image from the Pico."""
    with serial.Serial(
        request.port, BAUD_RATE, timeout=SERIAL_TIMEOUT_SECONDS
    ) as connection:
        connection.reset_input_buffer()
        command = (
            f"G {request.seed} {request.steps} "
            f"{request.condition} {request.cfg}\n"
        )
        connection.write(command.encode("ascii"))
        connection.flush()

        magic = read_magic(connection)
        if magic == b"RFIM":
            seed, width, height = struct.unpack("<IHH", read_exact(connection, 8))
            channels, condition = 1, 0
        else:
            seed, width, height, channels, condition = struct.unpack(
                "<IHHHH", read_exact(connection, 12)
            )

        image_size = width * height * channels
        if width <= 0 or height <= 0 or channels not in (1, 3):
            raise ValueError(
                f"Invalid image header: {width}x{height}x{channels}."
            )
        if image_size > MAX_IMAGE_BYTES:
            raise ValueError(f"Refusing unexpected {image_size:,}-byte image.")

        pixels = read_exact(connection, image_size)
        device_crc, generation_ms = struct.unpack(
            "<II", read_exact(connection, 8)
        )
        local_crc = binascii.crc32(pixels) & 0xFFFFFFFF

    return GenerationResult(
        seed=seed,
        width=width,
        height=height,
        channels=channels,
        condition=condition,
        pixels=pixels,
        device_crc=device_crc,
        local_crc=local_crc,
        generation_ms=generation_ms,
    )


class PicoFacesApp:
    BG = "#17191f"
    PANEL = "#232731"
    LCD_BEZEL = "#0c0e12"
    TEXT = "#edf0f5"
    MUTED = "#aab0bd"
    ACCENT = "#66d49a"
    ERROR = "#ff7c86"

    def __init__(self, root: tk.Tk, initial_port: str | None = None) -> None:
        self.root = root
        self.root.title("Pico Faces · Virtual Device")
        self.root.configure(bg=self.BG)
        self.root.minsize(920, 690)

        self.port_var = tk.StringVar(value=initial_port or "COM3")
        self.seed_var = tk.StringVar(value="42")
        self.steps_var = tk.IntVar(value=4)
        self.class_var = tk.IntVar(value=1)
        self.cfg_var = tk.StringVar(value="CFG 4")
        self.status_var = tk.StringVar(value="Ready — Pico performs all AI inference")
        self.detail_var = tk.StringVar(value="No image generated yet")

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.busy = False
        self.last_result: GenerationResult | None = None
        self.last_image: Image.Image | None = None
        self.display_photo: ImageTk.PhotoImage | None = None

        self._configure_styles()
        self._build_layout()
        self.refresh_ports()
        self._draw_lcd_placeholder()
        self.root.after(100, self._poll_events)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background=self.BG)
        style.configure("Panel.TFrame", background=self.PANEL)
        style.configure(
            "TLabel", background=self.BG, foreground=self.TEXT, font=("Segoe UI", 10)
        )
        style.configure(
            "Panel.TLabel",
            background=self.PANEL,
            foreground=self.TEXT,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Title.TLabel",
            background=self.BG,
            foreground=self.TEXT,
            font=("Segoe UI Semibold", 20),
        )
        style.configure(
            "Subtitle.TLabel",
            background=self.BG,
            foreground=self.MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Accent.TButton",
            font=("Segoe UI Semibold", 12),
            padding=(18, 12),
            background=self.ACCENT,
            foreground="#101713",
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#83e2ae"), ("disabled", "#48554e")],
        )
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 7))
        style.configure(
            "Choice.TRadiobutton",
            background=self.PANEL,
            foreground=self.TEXT,
            font=("Segoe UI", 10),
            padding=(7, 5),
        )
        style.map(
            "Choice.TRadiobutton",
            background=[("active", "#303642"), ("selected", "#354a42")],
            foreground=[("selected", self.ACCENT)],
        )
        style.configure(
            "TCombobox", fieldbackground="#f4f5f7", foreground="#111318"
        )

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=22)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 16))
        ttk.Label(header, text="Pico Faces", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Virtual controls and LCD · inference runs on the RP2350",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0)
        body.rowconfigure(0, weight=1)

        display_panel = ttk.Frame(body, style="Panel.TFrame", padding=18)
        display_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        display_panel.columnconfigure(0, weight=1)
        display_panel.rowconfigure(1, weight=1)
        ttk.Label(
            display_panel, text="VIRTUAL 128 × 128 LCD", style="Panel.TLabel"
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        self.lcd = tk.Canvas(
            display_panel,
            width=512,
            height=512,
            bg=self.LCD_BEZEL,
            highlightthickness=10,
            highlightbackground=self.LCD_BEZEL,
        )
        self.lcd.grid(row=1, column=0, sticky="n", padx=4)

        controls = ttk.Frame(body, style="Panel.TFrame", padding=18, width=300)
        controls.grid(row=0, column=1, sticky="ns")
        controls.grid_propagate(False)
        controls.columnconfigure(0, weight=1)

        ttk.Label(controls, text="CONNECTION", style="Panel.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        port_row = ttk.Frame(controls, style="Panel.TFrame")
        port_row.grid(row=1, column=0, sticky="ew", pady=(6, 14))
        port_row.columnconfigure(0, weight=1)
        self.port_box = ttk.Combobox(port_row, textvariable=self.port_var, width=15)
        self.port_box.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.refresh_button = ttk.Button(port_row, text="↻", width=3, command=self.refresh_ports)
        self.refresh_button.grid(row=0, column=1)

        ttk.Label(controls, text="FACE CLASS", style="Panel.TLabel").grid(
            row=2, column=0, sticky="w"
        )
        class_frame = ttk.Frame(controls, style="Panel.TFrame")
        class_frame.grid(row=3, column=0, sticky="ew", pady=(5, 13))
        self.class_buttons: list[ttk.Radiobutton] = []
        for index, label in enumerate(CLASS_LABELS):
            button = ttk.Radiobutton(
                class_frame,
                text=label,
                value=index,
                variable=self.class_var,
                style="Choice.TRadiobutton",
            )
            button.pack(fill="x", anchor="w")
            self.class_buttons.append(button)

        settings = ttk.Frame(controls, style="Panel.TFrame")
        settings.grid(row=4, column=0, sticky="ew")
        settings.columnconfigure(1, weight=1)
        ttk.Label(settings, text="Seed", style="Panel.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 9), pady=4
        )
        self.seed_entry = ttk.Entry(settings, textvariable=self.seed_var, width=13)
        self.seed_entry.grid(row=0, column=1, sticky="ew", pady=4)
        self.random_button = ttk.Button(settings, text="New", command=self.new_seed)
        self.random_button.grid(row=0, column=2, padx=(6, 0), pady=4)

        ttk.Label(settings, text="Steps", style="Panel.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 9), pady=4
        )
        self.steps_box = ttk.Combobox(
            settings,
            textvariable=self.steps_var,
            values=STEP_OPTIONS,
            state="readonly",
            width=8,
        )
        self.steps_box.grid(row=1, column=1, columnspan=2, sticky="ew", pady=4)

        ttk.Label(settings, text="Guidance", style="Panel.TLabel").grid(
            row=2, column=0, sticky="w", padx=(0, 9), pady=4
        )
        self.cfg_box = ttk.Combobox(
            settings,
            textvariable=self.cfg_var,
            values=CFG_OPTIONS,
            state="readonly",
            width=10,
        )
        self.cfg_box.grid(row=2, column=1, columnspan=2, sticky="ew", pady=4)

        self.generate_button = ttk.Button(
            controls,
            text="GENERATE",
            style="Accent.TButton",
            command=self.start_generation,
        )
        self.generate_button.grid(row=5, column=0, sticky="ew", pady=(18, 8))
        self.save_button = ttk.Button(
            controls, text="Save image…", command=self.save_image, state="disabled"
        )
        self.save_button.grid(row=6, column=0, sticky="ew")

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(14, 0))
        self.status_label = ttk.Label(footer, textvariable=self.status_var)
        self.status_label.pack(anchor="w")
        ttk.Label(footer, textvariable=self.detail_var, style="Subtitle.TLabel").pack(
            anchor="w", pady=(2, 0)
        )

        self.interactive_widgets = (
            self.port_box,
            self.refresh_button,
            self.seed_entry,
            self.random_button,
            self.steps_box,
            self.cfg_box,
            self.generate_button,
            *self.class_buttons,
        )

    def _draw_lcd_placeholder(self) -> None:
        self.lcd.delete("all")
        self.lcd.create_rectangle(0, 0, 512, 512, fill="#11141a", outline="")
        self.lcd.create_text(
            256,
            238,
            text="PICO FACES",
            fill=self.ACCENT,
            font=("Consolas", 25, "bold"),
        )
        self.lcd.create_text(
            256,
            280,
            text="PRESS GENERATE",
            fill="#717987",
            font=("Consolas", 13),
        )

    def refresh_ports(self) -> None:
        ports = [port.device for port in list_ports.comports()]
        self.port_box["values"] = ports
        if self.port_var.get() not in ports and ports:
            preferred = next((p for p in ports if p.upper() == "COM3"), ports[0])
            self.port_var.set(preferred)
        if ports:
            self.status_var.set(f"Ready — found {len(ports)} serial port(s)")
        else:
            self.status_var.set("No serial ports found — connect the Pico and refresh")

    def new_seed(self) -> None:
        self.seed_var.set(str(random.SystemRandom().randrange(0, 2**32)))

    def _request_from_controls(self) -> GenerationRequest:
        port = self.port_var.get().strip()
        if not port:
            raise ValueError("Choose a serial port.")
        try:
            seed = int(self.seed_var.get().strip(), 0)
        except ValueError as exc:
            raise ValueError("Seed must be a whole number.") from exc
        if not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("Seed must be between 0 and 4,294,967,295.")
        steps = int(self.steps_var.get())
        if steps not in STEP_OPTIONS:
            raise ValueError("Steps must be 1, 2, 4, or 8.")
        cfg_text = self.cfg_var.get()
        cfg = 0 if cfg_text == "Plain" else int(cfg_text.split()[-1])
        return GenerationRequest(
            port=port,
            seed=seed,
            steps=steps,
            condition=int(self.class_var.get()),
            cfg=cfg,
        )

    def start_generation(self) -> None:
        if self.busy:
            return
        try:
            request = self._request_from_controls()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc), parent=self.root)
            return

        self.busy = True
        self._set_controls_enabled(False)
        mode = "plain" if request.cfg == 0 else f"CFG {request.cfg}"
        self.status_var.set(
            f"Generating on Pico… {request.steps} step(s), {mode}"
        )
        self.detail_var.set("The window remains responsive while the RP2350 works.")
        thread = threading.Thread(
            target=self._generation_worker, args=(request,), daemon=True
        )
        thread.start()

    def _generation_worker(self, request: GenerationRequest) -> None:
        try:
            result = generate_on_pico(request)
            self.events.put(("result", result))
        except Exception as exc:  # surfaced in the GUI with context
            self.events.put(("error", exc))

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                self.busy = False
                self._set_controls_enabled(True)
                if kind == "result":
                    self._show_result(payload)  # type: ignore[arg-type]
                else:
                    self._show_error(payload)  # type: ignore[arg-type]
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in self.interactive_widgets:
            widget.configure(state=state)
        if enabled:
            self.steps_box.configure(state="readonly")
            self.cfg_box.configure(state="readonly")

    def _show_result(self, result: GenerationResult) -> None:
        self.last_result = result
        mode = "L" if result.channels == 1 else "RGB"
        image = Image.frombytes(mode, (result.width, result.height), result.pixels)
        if mode == "L":
            image = image.convert("RGB")
        self.last_image = image
        display = image.resize((512, 512), Image.Resampling.NEAREST)
        self.display_photo = ImageTk.PhotoImage(display)
        self.lcd.delete("all")
        self.lcd.create_image(0, 0, anchor="nw", image=self.display_photo)
        self.save_button.configure(state="normal")

        if result.crc_matches:
            self.status_var.set("Complete — Pico frame received and CRC verified")
            crc_text = f"CRC {result.device_crc:08x} ✓"
        else:
            self.status_var.set("Warning — image received with a CRC mismatch")
            crc_text = (
                f"device CRC {result.device_crc:08x}; "
                f"received CRC {result.local_crc:08x}"
            )
        class_name = (
            CLASS_LABELS[result.condition]
            if 0 <= result.condition < len(CLASS_LABELS)
            else f"class {result.condition}"
        )
        self.detail_var.set(
            f"Seed {result.seed} · {class_name} · "
            f"{result.generation_ms:,} ms · {crc_text}"
        )

    def _show_error(self, error: Exception) -> None:
        self.status_var.set("Generation failed — check the connection and settings")
        self.detail_var.set(str(error))
        messagebox.showerror("Pico communication error", str(error), parent=self.root)

    def save_image(self) -> None:
        if self.last_image is None or self.last_result is None:
            return
        default_name = f"seed_{self.last_result.seed}.png"
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save generated image",
            initialfile=default_name,
            defaultextension=".png",
            filetypes=(("PNG image", "*.png"),),
        )
        if path:
            self.last_image.save(Path(path))
            self.status_var.set(f"Saved {Path(path).name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="initial serial port (default: COM3)")
    parser.add_argument(
        "--protocol-self-test",
        action="store_true",
        help="validate command-independent protocol helpers and exit",
    )
    parser.add_argument(
        "--device-self-test",
        action="store_true",
        help="generate the default image on a connected Pico and exit",
    )
    return parser.parse_args()


def protocol_self_test() -> None:
    pixels = bytes(range(12))
    assert (binascii.crc32(pixels) & 0xFFFFFFFF) == 0x9270C965
    request = GenerationRequest("COM3", 42, 4, 1, 4)
    command = (
        f"G {request.seed} {request.steps} {request.condition} {request.cfg}\n"
    )
    assert command == "G 42 4 1 4\n"
    print("Protocol self-test passed.")


def main() -> None:
    args = parse_args()
    if args.protocol_self_test:
        protocol_self_test()
        return
    if args.device_self_test:
        result = generate_on_pico(
            GenerationRequest(args.port or "COM3", 42, 4, 1, 4)
        )
        status = "OK" if result.crc_matches else "CRC MISMATCH"
        print(
            f"seed {result.seed} cls {result.condition}: "
            f"{result.width}x{result.height}x{result.channels}, "
            f"{result.generation_ms} ms, crc {result.device_crc:08x} ({status})"
        )
        if not result.crc_matches:
            raise SystemExit(1)
        return
    root = tk.Tk()
    PicoFacesApp(root, args.port)
    root.mainloop()


if __name__ == "__main__":
    main()
