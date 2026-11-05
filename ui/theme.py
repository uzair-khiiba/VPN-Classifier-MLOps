"""One light, high-contrast palette for the dashboard and its charts, tuned for projectors.

Projectors wash out pale colours and thin grey text, so the page is pure white with near-black ink,
borders are visible rather than faint, and text is a little larger than Streamlit's default.

Chart colours come from a validated colourblind-safe palette (checked against the white surface):
  * the two "pushes toward / away from VPN/proxy" colours: orange and blue, both >= 3:1 on white;
  * the four model roles, in fixed order: blue, orange, aqua, yellow. Aqua and yellow are below 3:1
    on white, so every bar in that chart carries a visible value label and the chart has a legend.
.streamlit/config.toml repeats these hex values, because Streamlit reads its theme only from there.
"""

import altair as alt

# Surfaces and ink
SURFACE = "#ffffff"
PANEL = "#f0efec"           # sidebar, cards and table headers
INK = "#0b0b0b"             # primary text, and value labels on charts
INK_SECONDARY = "#52514e"   # axis titles and captions
AXIS = "#c3c2b7"            # baseline and axis lines
GRID = "#e1e0d9"            # hairline gridlines

# Data colours (categorical, fixed order)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"

TOWARD_VPN = ORANGE         # a feature pushing the score toward VPN/proxy
AWAY_FROM_VPN = BLUE        # a feature pushing it away

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
LABEL_SIZE = 14             # axis labels, legend labels and value labels: readable from the back of a room
TITLE_SIZE = 15


def style(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """Apply the projector styling to an Altair chart. Pass the result to st.altair_chart(..., theme=None)."""
    return (chart
            .configure(background=SURFACE, font=FONT)
            .configure_view(stroke=None)
            .configure_axis(labelColor=INK, labelFontSize=LABEL_SIZE, titleColor=INK_SECONDARY,
                            titleFontSize=TITLE_SIZE, titleFontWeight=600, gridColor=GRID, domainColor=AXIS,
                            tickColor=AXIS, labelLimit=280)
            .configure_legend(labelColor=INK, labelFontSize=LABEL_SIZE, symbolSize=180, symbolType="square")
            .configure_text(fontSize=LABEL_SIZE, fontWeight=600, color=INK))
