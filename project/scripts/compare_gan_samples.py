
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = ROOT / "results"
OUTPUT_DIR = SAMPLE_ROOT / "synthetic_evaluation"
OUTPUT_FILE = OUTPUT_DIR / "checkpoint_sample_comparison.png"

epochs = [50, 100, 150, 200]
gan_dirs = [
    ("DCGAN", SAMPLE_ROOT / "GAN_DCGAN" / "seed_0" / "samples"),
    ("WGAN-GP", SAMPLE_ROOT / "GAN_WGANGP" / "seed_0" / "samples"),
]

cell_width = 600
header_height = 36
row_height = 350
margin = 16

canvas = Image.new(
    "RGB",
    (margin * 2 + cell_width * 2, margin * 2 + header_height + row_height * len(epochs)),
    "white",
)
draw = ImageDraw.Draw(canvas)

for col, (gan_name, sample_dir) in enumerate(gan_dirs):
    x = margin + col * cell_width
    draw.text((x + 8, margin + 8), gan_name, fill="black")

for row, epoch in enumerate(epochs):
    y = margin + header_height + row * row_height
    draw.text((margin, y + 4), f"Epoch {epoch}", fill="black")

    for col, (gan_name, sample_dir) in enumerate(gan_dirs):
        path = sample_dir / f"epoch_{epoch:03d}.png"
        if not path.exists():
            raise FileNotFoundError(f"Missing sample grid: {path}")

        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((cell_width - 20, row_height - 40))
            x = margin + col * cell_width
            canvas.paste(im, (x + 8, y + 30))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
canvas.save(OUTPUT_FILE)
print(f"Saved comparison sheet: {OUTPUT_FILE}")
print(f"Image dimensions: {canvas.width} x {canvas.height}")
