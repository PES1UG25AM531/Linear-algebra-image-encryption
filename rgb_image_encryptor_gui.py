"""RGB Image Encryptor GUI — educational modular-matrix demonstration.

Requirements: Python 3.9+, NumPy, Pillow, and Tkinter (usually included with Python).
Run: python rgb_image_encryptor_gui.py

Each RGB pixel is a row vector P=[R,G,B]. Encryption mixes channel values modulo 256,
then permutes pixel positions. Decryption reverses the permutation and channel transform.
This is for education, not secure cryptography.
"""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageTk, ImageOps

MOD = 256
PIXEL_SCRAMBLE_SEED = 20261009  # Fixed demonstration key; same seed is required to decrypt.
K = np.array([[1, 2, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int64)


def inverse_mod_matrix(matrix: np.ndarray, modulus: int = MOD) -> np.ndarray:
    """Calculate a matrix inverse using Gauss–Jordan elimination mod modulus."""
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The key matrix must be square.")
    if modulus < 2:
        raise ValueError("Modulus must be at least 2.")
    n = matrix.shape[0]
    aug = np.concatenate((matrix % modulus, np.eye(n, dtype=np.int64)), axis=1)
    for col in range(n):
        pivot_row = next(
            (r for r in range(col, n)
             if math.gcd(int(aug[r, col]), modulus) == 1),
            None,
        )
        if pivot_row is None:
            raise ValueError(
                f"The key matrix has no invertible pivot in column {col}; "
                f"it is not invertible by this modular elimination method."
            )
        if pivot_row != col:
            aug[[col, pivot_row]] = aug[[pivot_row, col]]
        pivot_inv = pow(int(aug[col, col]), -1, modulus)
        aug[col, :] = (aug[col, :] * pivot_inv) % modulus
        for row in range(n):
            if row != col:
                factor = int(aug[row, col])
                aug[row, :] = (aug[row, :] - factor * aug[col, :]) % modulus
    left, right = aug[:, :n] % modulus, aug[:, n:] % modulus
    identity = np.eye(n, dtype=np.int64) % modulus
    if not np.array_equal(left, identity):
        raise ValueError("Gauss–Jordan elimination failed to produce identity.")
    if not (np.array_equal(matrix @ right % modulus, identity)
            and np.array_equal(right @ matrix % modulus, identity)):
        raise ValueError("The computed inverse did not pass verification.")
    return right


K_INV = inverse_mod_matrix(K, MOD)


def _validate_rgb_array(array: np.ndarray) -> None:
    if not isinstance(array, np.ndarray) or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Image data must be an RGB array of shape (height, width, 3).")
    if array.dtype != np.uint8:
        raise ValueError("RGB image data must use 8-bit channel values (0–255).")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("Image width and height must be greater than zero.")


def _pixel_permutation(pixel_count: int, seed: int = PIXEL_SCRAMBLE_SEED) -> np.ndarray:
    """Build a repeatable permutation of pixel positions from the scramble seed."""
    if pixel_count < 1:
        raise ValueError("The image must contain at least one pixel.")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("The pixel-scramble key must be an integer.")
    return np.random.default_rng(int(seed)).permutation(pixel_count)


def encrypt_rgb(image: np.ndarray, scramble_seed: int = PIXEL_SCRAMBLE_SEED) -> np.ndarray:
    """Mix RGB channels modulo 256, then shuffle pixel positions reproducibly."""
    _validate_rgb_array(image)
    pixels = image.reshape(-1, 3).astype(np.int64)
    mixed = (pixels @ K) % MOD
    order = _pixel_permutation(len(mixed), scramble_seed)
    scrambled = mixed[order]
    return scrambled.reshape(image.shape).astype(np.uint8)


def decrypt_rgb(image: np.ndarray, scramble_seed: int = PIXEL_SCRAMBLE_SEED) -> np.ndarray:
    """Undo pixel-position shuffling, then invert the RGB channel transform."""
    _validate_rgb_array(image)
    scrambled = image.reshape(-1, 3).astype(np.int64)
    order = _pixel_permutation(len(scrambled), scramble_seed)
    inverse_order = np.argsort(order)
    mixed = scrambled[inverse_order]
    pixels = (mixed @ K_INV) % MOD
    return pixels.reshape(image.shape).astype(np.uint8)


class RGBEncryptorApp:
    PREVIEW_SIZE = (310, 240)

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("RGB Image Encryptor | Linear Algebra Project")
        self.root.geometry("1120x760")
        self.root.minsize(900, 650)
        self.root.configure(bg="#f3f6fb")

        self.original: Optional[np.ndarray] = None
        self.encrypted: Optional[np.ndarray] = None
        self.decrypted: Optional[np.ndarray] = None
        self.input_path: Optional[Path] = None
        self._photo_refs: dict[str, ImageTk.PhotoImage] = {}
        self._preview_labels: dict[str, ttk.Label] = {}
        self._preview_info: dict[str, ttk.Label] = {}

        self._setup_style()
        self._build_ui()
        self._update_buttons()

    def _setup_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background="#f3f6fb")
        style.configure("Header.TLabel", background="#f3f6fb", foreground="#14213d",
                        font=("Segoe UI", 21, "bold"))
        style.configure("Sub.TLabel", background="#f3f6fb", foreground="#566176",
                        font=("Segoe UI", 10))
        style.configure("Card.TFrame", background="#ffffff", relief="flat")
        style.configure("CardTitle.TLabel", background="#ffffff", foreground="#17233b",
                        font=("Segoe UI", 11, "bold"))
        style.configure("Preview.TLabel", background="#edf1f7", foreground="#778196",
                        font=("Segoe UI", 10))
        style.configure("Info.TLabel", background="#ffffff", foreground="#667085",
                        font=("Segoe UI", 9))
        style.configure("Status.TLabel", background="#e8eef7", foreground="#24324b",
                        font=("Segoe UI", 9))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8))
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 7))

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, style="App.TFrame", padding=(22, 18, 22, 14))
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="RGB Image Encryptor", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="A visual demo of linear algebra, modular arithmetic, and reversible pixel transforms",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(3, 14))

        top = ttk.Frame(outer, style="App.TFrame")
        top.pack(fill="x", pady=(0, 12))
        ttk.Button(top, text="📂  Load Image", command=self.load_image,
                   style="Primary.TButton").pack(side="left")
        self.path_label = ttk.Label(top, text="No image selected", style="Sub.TLabel")
        self.path_label.pack(side="left", padx=(12, 0), fill="x", expand=True)

        actions = ttk.Frame(outer, style="App.TFrame")
        actions.pack(fill="x", pady=(0, 15))
        self.encrypt_button = ttk.Button(actions, text="🔐  Encrypt", command=self.encrypt_image,
                                         style="Primary.TButton")
        self.encrypt_button.pack(side="left", padx=(0, 8))
        self.decrypt_button = ttk.Button(actions, text="🔓  Decrypt", command=self.decrypt_image,
                                         style="Primary.TButton")
        self.decrypt_button.pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Load Encrypted PNG", command=self.load_encrypted_image).pack(
            side="left", padx=(0, 8)
        )
        self.save_encrypted_button = ttk.Button(actions, text="Save Encrypted", command=self.save_encrypted)
        self.save_encrypted_button.pack(side="left", padx=(0, 8))
        self.save_decrypted_button = ttk.Button(actions, text="Save Decrypted", command=self.save_decrypted)
        self.save_decrypted_button.pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Reset", command=self.reset).pack(side="right")

        previews = ttk.Frame(outer, style="App.TFrame")
        previews.pack(fill="both", expand=True)
        cards = [
            ("original", "1. Original Image", "Load any image; it is converted to RGB."),
            ("encrypted", "2. Encrypted Image", "RGB values mixed + pixel positions scrambled."),
            ("decrypted", "3. Decrypted Image", "Should match the original pixel for pixel."),
        ]
        for col, (key, title, description) in enumerate(cards):
            previews.columnconfigure(col, weight=1, uniform="preview")
            card = ttk.Frame(previews, style="Card.TFrame", padding=12)
            card.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 7, 0 if col == 2 else 7))
            ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w", pady=(0, 8))
            # Frame fixes the preview area size and provides a calm neutral background.
            image_area = tk.Frame(card, bg="#edf1f7", width=310, height=240,
                                  highlightthickness=1, highlightbackground="#dfe5ee")
            image_area.pack(fill="both", expand=True)
            image_area.pack_propagate(False)
            image_label = ttk.Label(image_area, text="No image", style="Preview.TLabel", anchor="center")
            image_label.pack(fill="both", expand=True)
            self._preview_labels[key] = image_label
            info = ttk.Label(card, text="—", style="Info.TLabel", wraplength=300)
            info.pack(anchor="w", pady=(8, 2))
            self._preview_info[key] = info
            ttk.Label(card, text=description, style="Info.TLabel", wraplength=300).pack(anchor="w", pady=(3, 0))

        formula = ttk.Frame(outer, style="App.TFrame", padding=(12, 10))
        formula.pack(fill="x", pady=(14, 0))
        formula.configure(relief="solid", borderwidth=1)
        ttk.Label(formula, text="Encryption: RGB channel mixing mod 256 + pixel-position permutation",
                  style="Sub.TLabel").pack(anchor="w")
        ttk.Label(formula, text="Decryption: inverse permutation + C × K⁻¹ (mod 256)  •  K⁻¹ found by modular Gauss–Jordan elimination",
                  style="Sub.TLabel").pack(anchor="w", pady=(4, 0))
        ttk.Label(formula, text=f"Scramble seed: {PIXEL_SCRAMBLE_SEED} (must stay the same for recovery)",
                  style="Sub.TLabel").pack(anchor="w", pady=(4, 0))

        self.status_var = tk.StringVar(value="Ready. Load an image to begin.")
        status = ttk.Label(outer, textvariable=self.status_var, style="Status.TLabel", padding=(10, 8))
        status.pack(fill="x", pady=(12, 0))
        self.root.bind("<Control-o>", lambda _event: self.load_image())
        self.root.bind("<Control-s>", lambda _event: self.save_encrypted())

    def _update_buttons(self) -> None:
        has_original = self.original is not None
        has_encrypted = self.encrypted is not None
        has_decrypted = self.decrypted is not None
        self.encrypt_button.configure(state="normal" if has_original else "disabled")
        self.decrypt_button.configure(state="normal" if has_encrypted else "disabled")
        self.save_encrypted_button.configure(state="normal" if has_encrypted else "disabled")
        self.save_decrypted_button.configure(state="normal" if has_decrypted else "disabled")

    def _set_preview(self, key: str, array: Optional[np.ndarray], placeholder: str = "No image") -> None:
        label = self._preview_labels[key]
        if array is None:
            label.configure(image="", text=placeholder)
            self._photo_refs.pop(key, None)
            self._preview_info[key].configure(text="—")
            return
        height, width, _channels = array.shape
        preview = Image.fromarray(array, mode="RGB")
        preview.thumbnail(self.PREVIEW_SIZE, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(preview)
        self._photo_refs[key] = photo  # Keep a reference so Tkinter does not clear the image.
        label.configure(image=photo, text="")
        self._preview_info[key].configure(text=f"{width} × {height} px  |  RGB  |  {width * height:,} pixels")

    def load_image(self) -> None:
        filename = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp"),
                       ("All files", "*.*")],
        )
        if not filename:
            return
        try:
            with Image.open(filename) as source:
                # Respect camera/phone orientation metadata before conversion.
                original = np.asarray(ImageOps.exif_transpose(source).convert("RGB"), dtype=np.uint8)
            _validate_rgb_array(original)
        except Exception as exc:
            messagebox.showerror("Could not load image", f"The selected file could not be opened as an image.\n\nDetails: {exc}")
            return
        self.original = original.copy()
        self.encrypted = None
        self.decrypted = None
        self.input_path = Path(filename)
        self.path_label.configure(text=f"{self.input_path.name}  •  {original.shape[1]} × {original.shape[0]} px")
        self._set_preview("original", self.original)
        self._set_preview("encrypted", None, "Click Encrypt")
        self._set_preview("decrypted", None, "Click Decrypt")
        self.status_var.set("Image loaded. Encrypt will mix RGB channels and scramble pixel positions.")
        self._update_buttons()

    def load_encrypted_image(self) -> None:
        """Load an encrypted PNG created by this application in an earlier session."""
        filename = filedialog.askopenfilename(
            title="Open a previously saved encrypted PNG",
            filetypes=[("PNG images", "*.png")],
        )
        if not filename:
            return
        try:
            with Image.open(filename) as source:
                encrypted = np.asarray(ImageOps.exif_transpose(source).convert("RGB"), dtype=np.uint8).copy()
            _validate_rgb_array(encrypted)
        except Exception as exc:
            messagebox.showerror(
                "Could not load encrypted image",
                f"Please select a valid encrypted PNG created by this app.\n\nDetails: {exc}",
            )
            return

        # Deliberately clear the original: decryption must work from the encrypted PNG alone.
        self.original = None
        self.encrypted = encrypted
        self.decrypted = None
        self.input_path = Path(filename)
        self.path_label.configure(text=f"Encrypted file: {self.input_path.name}  •  {encrypted.shape[1]} × {encrypted.shape[0]} px")
        self._set_preview("original", None, "Original not loaded\n(load only for comparison)")
        self._set_preview("encrypted", self.encrypted)
        self._set_preview("decrypted", None, "Click Decrypt")
        self.status_var.set(
            "Encrypted PNG loaded. Click Decrypt. Use a PNG made by this app; lossy JPEG files cannot be recovered exactly."
        )
        self._update_buttons()

    def encrypt_image(self) -> None:
        if self.original is None:
            messagebox.showinfo("No original image", "Click Load Image first.")
            return
        try:
            self.encrypted = encrypt_rgb(self.original)
            self.decrypted = None
            self._set_preview("encrypted", self.encrypted)
            self._set_preview("decrypted", None, "Click Decrypt")
            self.status_var.set("Encryption complete. RGB channels were mixed and pixel positions scrambled.")
            self._update_buttons()
        except Exception as exc:
            messagebox.showerror("Encryption failed", str(exc))
            self.status_var.set("Encryption failed.")

    def decrypt_image(self) -> None:
        if self.encrypted is None:
            messagebox.showinfo(
                "No encrypted image",
                "First encrypt an image, or click 'Load Encrypted PNG' to open a previously saved encrypted image.",
            )
            return
        try:
            self.decrypted = decrypt_rgb(self.encrypted)
            self._set_preview("decrypted", self.decrypted)
            if self.original is not None:
                exact = np.array_equal(self.original, self.decrypted)
                if exact:
                    self.status_var.set("SUCCESS — decrypted pixels match the original exactly.")
                    messagebox.showinfo(
                        "Verification successful",
                        "The decrypted image matches the original pixel by pixel.",
                    )
                else:
                    self.status_var.set("WARNING — decrypted pixels differ from the original.")
                    messagebox.showwarning(
                        "Verification failed",
                        "The decrypted pixels do not exactly match the original. Make sure the encrypted file was saved as PNG.",
                    )
            else:
                self.status_var.set(
                    "Decryption complete from the loaded encrypted PNG. Load the original separately if you want pixel-by-pixel comparison."
                )
                messagebox.showinfo(
                    "Decryption complete",
                    "The encrypted PNG has been decrypted. The original was not loaded, so automatic pixel-by-pixel comparison was not performed.",
                )
            self._update_buttons()
        except Exception as exc:
            messagebox.showerror("Decryption failed", str(exc))
            self.status_var.set("Decryption failed.")

    def _save_array(self, array: Optional[np.ndarray], default_name: str, dialog_title: str) -> None:
        if array is None:
            return
        path = filedialog.asksaveasfilename(
            title=dialog_title,
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG image (lossless)", "*.png")],
        )
        if not path:
            return
        save_path = Path(path)
        if save_path.suffix.lower() != ".png":
            save_path = save_path.with_suffix(".png")
        try:
            Image.fromarray(array, mode="RGB").save(save_path, format="PNG")
            # Verify that saving and reopening kept the RGB pixels unchanged.
            with Image.open(save_path) as saved:
                saved_array = np.asarray(saved.convert("RGB"), dtype=np.uint8)
            if not np.array_equal(array, saved_array):
                raise OSError("The saved PNG did not preserve the image pixels exactly.")
            self.status_var.set(f"Saved and verified: {save_path}")
            messagebox.showinfo("Image saved", f"Saved lossless PNG successfully:\n{save_path}")
        except Exception as exc:
            messagebox.showerror("Save failed", f"Could not save the image.\n\nDetails: {exc}")

    def save_encrypted(self) -> None:
        name = (self.input_path.stem if self.input_path else "image") + "_encrypted.png"
        self._save_array(self.encrypted, name, "Save encrypted image")

    def save_decrypted(self) -> None:
        name = (self.input_path.stem if self.input_path else "image") + "_decrypted.png"
        self._save_array(self.decrypted, name, "Save decrypted image")

    def reset(self) -> None:
        self.original = None
        self.encrypted = None
        self.decrypted = None
        self.input_path = None
        self.path_label.configure(text="No image selected")
        self._set_preview("original", None)
        self._set_preview("encrypted", None)
        self._set_preview("decrypted", None)
        self.status_var.set("Reset complete. Load an original image or a previously saved encrypted PNG.")
        self._update_buttons()


def main() -> None:
    root = tk.Tk()
    RGBEncryptorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
