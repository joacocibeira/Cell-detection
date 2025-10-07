from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

def plot_yolo_annotations(image_path, label_path=None, class_names=None, figsize=(8,8)):
    """
    Draw YOLO-format labels on an image.
    - image_path: path to the image file
    - label_path: path to the .txt labels (auto-derive from image if None)
    - class_names: list or dict mapping class id -> name (optional)
    """
    image_path = Path(image_path)
    if label_path is None:
        if "images" in image_path.parts:
            i = image_path.parts.index("images")
            label_path = Path(*image_path.parts[:i], "labels", *image_path.parts[i+1:-1]) / (image_path.stem + ".txt")
        else:
            label_path = image_path.with_suffix(".txt")

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis("off")

    lp = Path(label_path)
    if lp.is_file():
        with lp.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                cls, xc, yc, bw, bh = map(float, line.split()[:5])
                cls = int(cls)
                x1 = (xc - bw/2) * w
                y1 = (yc - bh/2) * h
                ww = bw * w
                hh = bh * h
                ax.add_patch(Rectangle((x1, y1), ww, hh, fill=False, edgecolor="lime", linewidth=2))
                if class_names is not None:
                    name = class_names[cls] if not isinstance(class_names, dict) else class_names.get(cls, str(cls))
                    ax.text(x1, max(0, y1 - 2), name, fontsize=9,
                            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
    else:
        ax.set_title("No label file found", fontsize=10)

    plt.show()
