"""Generate self-contained PNG/PDF manuscript figures using Pillow only."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


OUT = Path("figures")
OUT.mkdir(exist_ok=True)
FONT_PAIRS = (
    (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
    (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ),
)


def font(size, bold=False):
    for regular_path, bold_path in FONT_PAIRS:
        candidate = bold_path if bold else regular_path
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)


def save(image, stem):
    image.save(OUT / f"{stem}.png", dpi=(300, 300))
    image.convert("RGB").save(OUT / f"{stem}.pdf", resolution=300)


def axes(draw, box, title, xlabel, ylabel, xlim, ylim, log_y=False, categorical=False):
    x0, y0, x1, y1 = box
    left, top, right, bottom = x0 + 105, y0 + 65, x1 - 30, y1 - 85
    draw.line((left, top, left, bottom), fill="black", width=3)
    draw.line((left, bottom, right, bottom), fill="black", width=3)
    draw.text((x0 + 15, y0 + 10), title, font=font(30, True), fill="black")
    draw.text(((left + right) / 2, y1 - 50), xlabel, font=font(24), fill="black", anchor="mm")
    draw.text((left, top - 24), ylabel, font=font(19), fill="#333333", anchor="lm")

    def project(x, y):
        xx = left + (x - xlim[0]) / (xlim[1] - xlim[0]) * (right - left)
        value = np.log10(y) if log_y else y
        lo = np.log10(ylim[0]) if log_y else ylim[0]
        hi = np.log10(ylim[1]) if log_y else ylim[1]
        yy = bottom - (value - lo) / (hi - lo) * (bottom - top)
        return xx, yy

    if not categorical:
        for fraction in np.linspace(0, 1, 5):
            xv = xlim[0] + fraction * (xlim[1] - xlim[0])
            xx, _ = project(xv, ylim[0])
            draw.line((xx, bottom, xx, bottom + 8), fill="black", width=2)
            draw.text((xx, bottom + 25), f"{xv:g}", font=font(19), fill="black", anchor="mm")
    y_values = np.geomspace(ylim[0], ylim[1], 5) if log_y else np.linspace(ylim[0], ylim[1], 5)
    for yv in y_values:
        _, yy = project(xlim[0], yv)
        draw.line((left - 8, yy, right, yy), fill="#dddddd", width=1)
        draw.text((left - 14, yy), f"{yv:.3g}", font=font(18), fill="black", anchor="rm")
    return project


def main_figure():
    frontier = list(csv.DictReader(Path("symmetric_emission_frontier.csv").open(encoding="utf-8")))
    spectral = list(csv.DictReader(Path("symmetric_emission_spectral_audit.csv").open(encoding="utf-8")))
    robust = list(csv.DictReader(Path("detector_robustness.csv").open(encoding="utf-8")))[:4]
    sample = list(csv.DictReader(Path("finite_sample_power.csv").open(encoding="utf-8")))[:2]
    image = Image.new("RGB", (2400, 1700), "white")
    draw = ImageDraw.Draw(image)
    boxes = [(0, 0, 1200, 850), (1200, 0, 2400, 850), (0, 850, 1200, 1700), (1200, 850, 2400, 1700)]
    colors = {20.0: "#2166ac", 40.0: "#1a9850", 60.0: "#d73027"}

    project = axes(draw, boxes[0], "(a) Resource-constrained emulation", "Classical states", "KL rate", (10, 32), (0.02, 0.05))
    for affinity in (20.0, 40.0, 60.0):
        rows = [r for r in frontier if float(r["hidden_cycle_affinity"]) == affinity]
        points = [project(int(r["classical_states"]), float(r["asymptotic_kl_rate"])) for r in rows]
        draw.line(points, fill=colors[affinity], width=6)
        for point in points:
            draw.ellipse((point[0]-9, point[1]-9, point[0]+9, point[1]+9), fill=colors[affinity])
        draw.text((850, 105 + affinity * 3), f"A={affinity:g}", font=font(22), fill=colors[affinity])

    project = axes(draw, boxes[1], "(b) No-click pole separation", "Decay rate", "Frequency", (0.8, 2.3), (3.5, 4.1))
    rows = [r for r in spectral if int(r["phases_per_macrostate"]) == 15]
    for row in rows:
        affinity = float(row["hidden_cycle_affinity"])
        point = project(-float(row["killed_mode_real"]), float(row["killed_mode_imaginary"]))
        draw.ellipse((point[0]-12, point[1]-12, point[0]+12, point[1]+12), fill=colors[affinity])
        draw.text((point[0]+18, point[1]), f"A={affinity:g}", font=font(20), fill=colors[affinity], anchor="lm")
    point = project(0.9, 4.0)
    draw.regular_polygon((point[0], point[1], 18), 5, fill="black")
    draw.text((point[0]+24, point[1]), "quantum", font=font(20, True), fill="black", anchor="lm")

    project = axes(draw, boxes[2], "(c) Detector post-processing", "", "Information rate", (-0.5, 3.5), (0, 0.026), categorical=True)
    baseline_y = project(0, 0)[1]
    for index, row in enumerate(robust):
        for offset, key, color in ((-0.16, "information_rate_per_original_time", "#2166ac"), (0.16, "fourier_pinsker_rate_per_original_time", "#f4a582")):
            px0, py = project(index + offset, float(row[key])); px1, _ = project(index + offset + 0.25, 0)
            draw.rectangle((px0, py, px1, baseline_y), fill=color)
    for index, name in enumerate(("ideal", "mild", "moderate", "severe")):
        xx, yy = project(index, 0); draw.text((xx, yy + 50), name, font=font(18), fill="black", anchor="mm")
    draw.rectangle((750, 930, 775, 955), fill="#2166ac"); draw.text((785, 942), "cycle KL", font=font(20), fill="black", anchor="lm")
    draw.rectangle((750, 970, 775, 995), fill="#f4a582"); draw.text((785, 982), "Fourier bound", font=font(20), fill="black", anchor="lm")

    project = axes(draw, boxes[3], "(d) Finite-sample scales", "", "Complete cycles (log)", (-0.5, 1.5), (100, 20000), log_y=True, categorical=True)
    base = project(0, 100)[1]
    for index, row in enumerate(sample):
        for offset, key, color in ((-0.18, "gaussian_llr_cycles", "#2166ac"), (0.18, "hoeffding_fourier_cycles", "#f4a582")):
            xx, yy = project(index + offset, float(row[key])); xx2, _ = project(index + offset + 0.28, 100)
            draw.rectangle((xx, yy, xx2, base), fill=color)
        xx, yy = project(index, 100); draw.text((xx, yy + 50), f"{int(float(row['sigma_level']))} sigma", font=font(19), fill="black", anchor="mm")
    draw.rectangle((1940, 930, 1965, 955), fill="#2166ac"); draw.text((1975, 942), "LLR scale", font=font(20), fill="black", anchor="lm")
    draw.rectangle((1940, 970, 1965, 995), fill="#f4a582"); draw.text((1975, 982), "Fourier guarantee", font=font(20), fill="black", anchor="lm")
    save(image, "fig_main_results")


def resource_phase_diagram():
    states = np.arange(4, 61); affinities = np.linspace(2.0, 100.0, 99); extra = np.empty((len(affinities), len(states))); rows = []
    for i, affinity in enumerate(affinities):
        for j, count in enumerate(states):
            bound = 1 / np.tan(np.pi / count) * np.tanh(affinity / (2 * count)); extra[i, j] = 4.0 / bound
            rows.append({"hidden_ring_states": count, "cycle_affinity": affinity, "coherence_bound": bound, "minimum_extra_decay_at_omega_4": extra[i, j]})
    with Path("resource_phase_diagram.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    image = Image.new("RGB", (1700, 1150), "white"); draw = ImageDraw.Draw(image); left, top, right, bottom = 190, 100, 1500, 980
    for i in range(len(affinities)):
        for j in range(len(states)):
            value = np.clip((np.log10(extra[i,j]) + 0.6) / 1.5, 0, 1); color = (int(250-170*value), int(245-190*value), int(210-90*value))
            x0 = left + j/len(states)*(right-left); x1 = left + (j+1)/len(states)*(right-left); y1 = bottom-i/len(affinities)*(bottom-top); y0 = bottom-(i+1)/len(affinities)*(bottom-top)
            draw.rectangle((x0,y0,x1+1,y1+1), fill=color)
    draw.rectangle((left,top,right,bottom), outline="black", width=3); draw.text((850,35), "Minimum extra classical decay at frequency 4", font=font(38,True), fill="black", anchor="mm")
    draw.text((850,1070), "Hidden-ring states N", font=font(30), fill="black", anchor="mm"); draw.text((left,70), "Cycle affinity A", font=font(25), fill="black", anchor="lm")
    for count in (10,20,30,40,50,60):
        x = left+(count-states[0])/(states[-1]-states[0])*(right-left); draw.text((x,bottom+35),str(count),font=font(22),fill="black",anchor="mm")
    for affinity in (20,40,60,80,100):
        y = bottom-(affinity-affinities[0])/(affinities[-1]-affinities[0])*(bottom-top); draw.text((left-25,y),str(affinity),font=font(22),fill="black",anchor="rm")
    for k,label in enumerate(("0.25","0.5","1","2","4","8")):
        x0=1530; y0=180+k*90; value=k/5; color=(int(250-170*value),int(245-190*value),int(210-90*value)); draw.rectangle((x0,y0,x0+45,y0+55),fill=color); draw.text((1585,y0+27),label,font=font(22),fill="black",anchor="lm")
    save(image, "fig_resource_phase_diagram")


def channel_figure():
    dec = list(csv.DictReader(Path("symmetric_emission_decomposition_pairs.csv").open(encoding="utf-8"))); fou = list(csv.DictReader(Path("symmetric_emission_fourier_pairs.csv").open(encoding="utf-8")))
    matrices = [np.maximum(np.array([float(r["timing_kl_rate_contribution"]) for r in dec]).reshape(4,4), 0.0), np.array([float(r["optimal_phase_gap_contribution"]) for r in fou]).reshape(4,4)]
    image=Image.new("RGB",(1800,850),"white"); draw=ImageDraw.Draw(image); labels=("c down","c up","h down","h up")
    for panel,(matrix,title) in enumerate(zip(matrices,("(a) Timing KL rate","(b) Fourier-gap contribution"))):
        xbase=180+panel*880; ybase=150; cell=135; maximum=matrix.max(); draw.text((xbase+270,55),title,font=font(34,True),fill="black",anchor="mm")
        for i in range(4):
            draw.text((xbase-20,ybase+(i+.5)*cell),labels[i],font=font(21),fill="black",anchor="rm"); draw.text((xbase+(i+.5)*cell,ybase+4*cell+35),labels[i],font=font(21),fill="black",anchor="mm")
            for j in range(4):
                frac=matrix[i,j]/maximum if maximum else 0; color=(int(245-180*frac),int(245-105*frac),int(245-25*frac)); box=(xbase+j*cell,ybase+i*cell,xbase+(j+1)*cell,ybase+(i+1)*cell)
                draw.rectangle(box,fill=color,outline="white",width=2); draw.text((xbase+(j+.5)*cell,ybase+(i+.5)*cell),f"{matrix[i,j]:.3g}",font=font(19,True),fill="black",anchor="mm")
        draw.text((xbase+270,760),"Following label",font=font(25),fill="black",anchor="mm")
    save(image,"fig_channel_resolved_witness")


def main():
    main_figure(); resource_phase_diagram(); channel_figure(); print("wrote manuscript PNG/PDF figures and resource_phase_diagram.csv")


if __name__ == "__main__":
    main()
