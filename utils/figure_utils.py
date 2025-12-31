import io
import zipfile


def collect_task_figures(task_result: dict) -> dict[str, object]:
    """
    Return {filename: matplotlib_figure} for a task result.
    Supports:
      - Task 4 style: fig1, fig2
      - Task 5 style: figures dict
    """
    name = task_result.get("name", "Task").replace("/", "-")
    out: dict[str, object] = {}

    # Task 5 style: {"figures": {"fig0": fig0, "fig1": fig1, ...}}
    figs = task_result.get("figures")
    if isinstance(figs, dict):
        for k, fig in figs.items():
            if fig is not None:
                out[f"{name}_{k}.png"] = fig

    # Task 4 style: fig1, fig2 (single figures)
    for k in ("fig1", "fig2", "fig0"):
        fig = task_result.get(k)
        if fig is not None:
            out[f"{name}_{k}.png"] = fig

    return out


def figures_zip_bytes(figures: dict[str, object], dpi: int = 200) -> bytes:
    """
    Build a zip (in-memory) containing PNG files from matplotlib figures.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, fig in figures.items():
            img = io.BytesIO()
            # matplotlib Figure has savefig()
            fig.savefig(img, format="png", dpi=dpi, bbox_inches="tight")
            img.seek(0)
            zf.writestr(filename, img.read())
    buf.seek(0)
    return buf.read()
