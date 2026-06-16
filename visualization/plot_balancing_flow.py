"""
plot_balancing_flow.py
======================
Visualisasi alur penerapan balancing (class weights + Categorical Focal Loss)
pada pelatihan model BiLSTM untuk prediksi kekeringan.

Langkah-langkah yang divisualisasikan:
1. Menghitung distribusi jumlah sampel pada setiap kelas data pelatihan.
2. Menghitung bobot masing-masing kelas menggunakan Persamaan (3.1).
3. Mengintegrasikan bobot kelas ke dalam proses optimasi model melalui fungsi Categorical Focal Loss.
4. Melatih model BiLSTM menggunakan data pelatihan yang telah diberikan bobot kelas.
5. Mengevaluasi performa model menggunakan metrik Accuracy, Precision, Recall, F1-Score, dan Macro F1-Score.

Jalankan dari folder root project:
  python visualization/plot_balancing_flow.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flowchart():
    # Setup light mode theme
    fig, ax = plt.subplots(figsize=(8.8, 11.5), facecolor='#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    
    # Hide axes
    ax.axis('off')
    ax.set_xlim(0.6, 9.4)
    ax.set_ylim(0.1, 11.5)

    # Title
    ax.text(
        5, 11.0,
        "ALUR PENERAPAN CLASS BALANCING PADA MODEL BiLSTM",
        ha='center', va='center',
        fontsize=13.5, fontweight='bold', color='#0F172A',
        fontname='Segoe UI'
    )
    ax.text(
        5, 10.65,
        "Strategi Penanganan Imbalance Data menggunakan Bobot Kelas & Categorical Focal Loss",
        ha='center', va='center',
        fontsize=9.5, style='italic', color='#475569',
        fontname='Segoe UI'
    )

    # Define steps with light pastel backgrounds and dark readable text
    steps = [
        {
            "num": "1",
            "title": "DISTRIBUSI SAMPEL KELAS",
            "desc": "Menghitung distribusi jumlah sampel pada setiap kelas data pelatihan.",
            "formula": r"$N_c = \sum_{i=1}^{N_{train}} [y_i = c]$  untuk $c \in \{None, D0, D1, D2, D3, D4\}$",
            "color": "#0D9488",      # Teal 600
            "bg_color": "#F0FDFA",   # Teal 50
            "y": 9.5
        },
        {
            "num": "2",
            "title": "PERHITUNGAN BOBOT KELAS",
            "desc": "Menghitung bobot masing-masing kelas untuk menyeimbangkan kontribusi loss.",
            "formula": r"$\alpha_c = \frac{1 / N_c}{\sum_{j=0}^{C-1} (1 / N_j)}$   atau   $w_c = \frac{N_{train}}{C \times N_c}$",
            "color": "#0891B2",      # Cyan 600
            "bg_color": "#ECFEFF",   # Cyan 50
            "y": 7.4
        },
        {
            "num": "3",
            "title": "INTEGRASI KE CATEGORICAL FOCAL LOSS",
            "desc": "Mengintegrasikan bobot kelas (\u03b1_c) ke dalam fungsi loss untuk memfokuskan\noptimasi pada sampel yang sulit dideteksi (hard examples).",
            "formula": r"$L_{Focal} = - \sum_{c=0}^{C-1} \alpha_c (1 - p_c)^\gamma y_c \log(p_c)$   |   $\gamma = 2.0$, $p_c$ (pred), $y_c$ (aktual)",
            "color": "#4F46E5",      # Indigo 600
            "bg_color": "#EEF2FF",   # Indigo 50
            "y": 5.3
        },
        {
            "num": "4",
            "title": "PELATIHAN MODEL BiLSTM",
            "desc": "Melatih model Bidirectional LSTM menggunakan data pelatihan dengan fungsi\nCategorical Focal Loss terbobot untuk menangkap pola temporal runtun waktu.",
            "formula": r"$\theta^* = \arg\min_{\theta} \sum_{k} L_{Focal}(f_\theta(x_k), y_k)$",
            "color": "#7C3AED",      # Purple 600
            "bg_color": "#F5F3FF",   # Purple 50
            "y": 3.2
        },
        {
            "num": "5",
            "title": "EVALUASI PERFORMA MODEL",
            "desc": "Mengevaluasi performa model BiLSTM menggunakan data uji independen.",
            "formula": "Metrik: Accuracy, Precision, Recall, F1-Score, dan Macro F1-Score",
            "color": "#059669",      # Emerald 600
            "bg_color": "#ECFDF5",   # Emerald 50
            "y": 1.1
        }
    ]

    # Draw boxes and connectors
    box_width = 8.6
    box_height = 1.65

    for i, step in enumerate(steps):
        y_center = step["y"]
        x_left = 5.0 - box_width / 2
        y_bottom = y_center - box_height / 2

        # 1. Shadow effect (soft light gray offset box)
        shadow = patches.FancyBboxPatch(
            (x_left + 0.04, y_bottom - 0.04), box_width, box_height,
            boxstyle="round,pad=0.06,rounding_size=0.1",
            facecolor='#E2E8F0', edgecolor='none', alpha=0.4,
            zorder=1
        )
        ax.add_patch(shadow)

        # 2. Main Box with light pastel background and bold colored border
        box = patches.FancyBboxPatch(
            (x_left, y_bottom), box_width, box_height,
            boxstyle="round,pad=0.06,rounding_size=0.1",
            facecolor=step["bg_color"], edgecolor=step["color"], linewidth=1.5,
            zorder=2
        )
        ax.add_patch(box)

        # 3. Step Number Circle (Badge)
        badge_x = x_left + 0.45
        badge_y = y_center + 0.48
        circle = patches.Circle(
            (badge_x, badge_y), 0.28,
            facecolor=step["color"], edgecolor='none',
            zorder=3
        )
        ax.add_patch(circle)

        # Step Number Text
        ax.text(
            badge_x, badge_y, step["num"],
            ha='center', va='center',
            fontsize=11, fontweight='bold', color='#FFFFFF',
            fontname='Segoe UI', zorder=4
        )

        # 4. Box Title (High Contrast Dark Text - Aligned at x_left + 0.9)
        ax.text(
            x_left + 0.9, y_center + 0.48, step["title"],
            ha='left', va='center',
            fontsize=11, fontweight='bold', color='#0F172A',
            fontname='Segoe UI', zorder=3
        )

        # 5. Box Description (Aligned at x_left + 0.9 to avoid circle overlap)
        ax.text(
            x_left + 0.9, y_center - 0.02, step["desc"],
            ha='left', va='center',
            fontsize=9.5, color='#334155',
            fontname='Segoe UI', zorder=3
        )

        # 6. Formula / Details Block (Aligned at x_left + 0.9 to prevent overlaps)
        ax.text(
            x_left + 0.9, y_center - 0.56, step["formula"],
            ha='left', va='center',
            fontsize=9.5, fontweight='bold', color='#1E293B',
            fontname='DejaVu Sans', zorder=3
        )

        # 7. Connecting arrow to next step
        if i < len(steps) - 1:
            next_y_top = steps[i+1]["y"] + box_height / 2
            arrow_start_y = y_bottom
            arrow_end_y = next_y_top + 0.05
            
            ax.annotate(
                '',
                xy=(5, arrow_end_y),
                xytext=(5, arrow_start_y),
                arrowprops=dict(
                    arrowstyle="fancy,head_length=0.5,head_width=0.5,tail_width=0.12",
                    color=step["color"],
                    alpha=0.75,
                    shrinkA=1,
                    shrinkB=1,
                    connectionstyle="arc3,rad=0.0"
                ),
                zorder=2
            )

    # Save path
    out_dir = os.path.dirname(__file__)
    out_path = os.path.join(out_dir, 'plot_balancing_flow.png')
    
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"Flowchart berhasil disimpan di: {out_path}")

if __name__ == '__main__':
    draw_flowchart()
