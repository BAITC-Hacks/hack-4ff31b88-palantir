"""Decorative, self-contained wind farm backdrop for the repository agent."""

from html import escape


def _turbine(
    x: int,
    hub_y: int,
    ground_y: int,
    radius: int,
    duration: int,
    delay: int,
    opacity: float,
    *,
    distant: bool = False,
) -> str:
    """Build a tower and an independently rotating, balanced three-blade rotor."""
    tower_top = max(2.0, radius * 0.022)
    tower_base = radius * 0.042
    blade = (
        f"M {-radius * 0.018:.2f} {radius * 0.025:.2f} "
        f"C {-radius * 0.065:.2f} {-radius * 0.11:.2f}, "
        f"{-radius * 0.045:.2f} {-radius * 0.48:.2f}, "
        f"{radius * 0.014:.2f} {-radius:.2f} "
        f"Q {radius * 0.033:.2f} {-radius * 1.015:.2f}, "
        f"{radius * 0.035:.2f} {-radius * 0.96:.2f} "
        f"L {radius * 0.047:.2f} {-radius * 0.18:.2f} "
        f"Q {radius * 0.052:.2f} {-radius * 0.06:.2f}, "
        f"{radius * 0.018:.2f} {radius * 0.025:.2f} Z"
    )
    distance_class = " wind-scene__distant" if distant else ""
    return f"""
    <g class="wind-scene__turbine{distance_class}" opacity="{opacity}">
      <path d="M {x - tower_top:.2f} {hub_y}
               L {x + tower_top:.2f} {hub_y}
               L {x + tower_base:.2f} {ground_y}
               L {x - tower_base:.2f} {ground_y} Z"
            fill="url(#wind-scene-tower)"/>
      <path d="M {x + tower_top * 0.35:.2f} {hub_y + 5}
               L {x + tower_base * 0.5:.2f} {ground_y}"
            stroke="#e3f4e8" stroke-width="0.8" opacity="0.26"/>
      <g transform="translate({x} {hub_y})">
        <g class="wind-scene__rotor"
           style="--wind-duration:{duration}s;--wind-delay:-{delay}s">
          <circle r="{radius * 1.02:.2f}" fill="transparent"/>
          <path d="{blade}" fill="url(#wind-scene-blade)"/>
          <path d="{blade}" fill="url(#wind-scene-blade)" transform="rotate(120)"/>
          <path d="{blade}" fill="url(#wind-scene-blade)" transform="rotate(240)"/>
        </g>
        <circle r="{max(3, radius * 0.035):.2f}" fill="#b5d9cc"/>
        <circle r="{max(1.2, radius * 0.011):.2f}" fill="#e9c995" opacity="0.9"/>
      </g>
      <ellipse cx="{x}" cy="{ground_y}" rx="{radius * 0.22:.2f}"
               ry="{radius * 0.018:.2f}" fill="#071e25" opacity="0.75"/>
    </g>"""


def wind_scene_html(animate: bool = True) -> str:
    """Return a decorative HTML/SVG scene; no scripts or external assets required.

    Every moving object obeys the animation flag and the browser's reduced-motion
    preference. Transparent rotor circles keep the CSS transform bounds centred
    exactly on each hub, independently from the unequal three-blade bounds.
    """
    scene_class = "wind-scene" + ("" if animate else " wind-scene--paused")
    stars = "".join(
        f'<circle cx="{x}" cy="{y}" r="{r}" opacity="{opacity}"/>'
        for x, y, r, opacity in (
            (65, 78, 1, 0.24), (243, 125, 1.2, 0.35),
            (418, 58, 0.8, 0.26), (648, 177, 0.9, 0.23),
            (753, 76, 1.1, 0.30), (958, 116, 1, 0.30),
            (1023, 52, 1.4, 0.42), (1115, 198, 0.8, 0.28),
            (1225, 89, 1, 0.24), (1403, 61, 1.2, 0.40),
            (1508, 162, 0.8, 0.34), (1571, 47, 1, 0.23),
            (845, 261, 0.8, 0.23), (1345, 236, 0.7, 0.19),
        )
    )
    turbines = "".join(
        _turbine(*spec, distant=distant)
        for spec, distant in (
            ((130, 670, 875, 76, 34, 10, 0.19), True),
            ((446, 745, 900, 58, 31, 18, 0.15), True),
            ((1075, 555, 934, 142, 29, 7, 0.30), False),
            ((1510, 615, 992, 128, 32, 21, 0.24), True),
            ((1364, 306, 992, 230, 26, 3, 0.55), False),
        )
    )
    return f"""
<style>
.wind-scene {{
    position: fixed;
    inset: 0;
    z-index: 0;
    overflow: hidden;
    pointer-events: none;
    background: #081b23;
    contain: paint;
}}
.wind-scene__landscape {{
    display: block;
    width: 100%;
    height: 100%;
}}
.wind-scene__rotor {{
    transform-box: fill-box;
    transform-origin: center;
    animation: wind-scene-spin var(--wind-duration, 30s) linear infinite;
    animation-delay: var(--wind-delay, 0s);
}}
.wind-scene__haze {{
    animation: wind-scene-drift 42s ease-in-out infinite alternate;
}}
.wind-scene__haze--far {{
    animation-duration: 58s;
    animation-delay: -24s;
}}
.wind-scene__current {{
    stroke-dasharray: 35 480;
    stroke-linecap: round;
    animation: wind-scene-flow 22s linear infinite;
}}
.wind-scene__current--second {{
    animation-duration: 29s;
    animation-delay: -12s;
}}
.wind-scene--paused *, .wind-scene--paused *::before,
.wind-scene--paused *::after {{ animation-play-state: paused !important; }}
@keyframes wind-scene-spin {{ to {{ transform: rotate(360deg); }} }}
@keyframes wind-scene-drift {{
    from {{ transform: translateX(-18px); }}
    to {{ transform: translateX(30px); }}
}}
@keyframes wind-scene-flow {{ to {{ stroke-dashoffset: -1030; }} }}
@media (prefers-reduced-motion: reduce) {{
    .wind-scene *, .wind-scene *::before, .wind-scene *::after {{
        animation-play-state: paused !important;
    }}
}}
@media (max-width: 700px) {{
    .wind-scene__distant, .wind-scene__current {{ display: none; }}
    .wind-scene__landscape {{ opacity: 0.7; }}
}}
</style>
<div class="{escape(scene_class)}" aria-hidden="true">
  <svg class="wind-scene__landscape" xmlns="http://www.w3.org/2000/svg"
       viewBox="0 0 1600 1000" preserveAspectRatio="xMaxYMid slice"
       aria-hidden="true" focusable="false">
    <defs>
      <linearGradient id="wind-scene-sky" x1="0" y1="0" x2="0.65" y2="1">
        <stop offset="0" stop-color="#071820"/>
        <stop offset="0.53" stop-color="#0b2730"/>
        <stop offset="1" stop-color="#204740"/>
      </linearGradient>
      <radialGradient id="wind-scene-horizon">
        <stop offset="0" stop-color="#76b8a0" stop-opacity="0.23"/>
        <stop offset="1" stop-color="#76b8a0" stop-opacity="0"/>
      </radialGradient>
      <radialGradient id="wind-scene-moonlight">
        <stop offset="0" stop-color="#c5e7dc" stop-opacity="0.07"/>
        <stop offset="1" stop-color="#c5e7dc" stop-opacity="0"/>
      </radialGradient>
      <linearGradient id="wind-scene-tower" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0" stop-color="#355a5a"/>
        <stop offset="0.48" stop-color="#adcfc2"/>
        <stop offset="1" stop-color="#577970"/>
      </linearGradient>
      <linearGradient id="wind-scene-blade" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#d5e9db"/>
        <stop offset="0.55" stop-color="#adcfc2"/>
        <stop offset="1" stop-color="#6d998c"/>
      </linearGradient>
      <linearGradient id="wind-scene-foreground" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#0a242b" stop-opacity="0"/>
        <stop offset="1" stop-color="#061820" stop-opacity="0.95"/>
      </linearGradient>
      <linearGradient id="wind-scene-leftshade" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0" stop-color="#071820" stop-opacity="0.67"/>
        <stop offset="0.53" stop-color="#071820" stop-opacity="0.22"/>
        <stop offset="1" stop-color="#071820" stop-opacity="0"/>
      </linearGradient>
    </defs>
    <rect width="1600" height="1000" fill="url(#wind-scene-sky)"/>
    <ellipse cx="1230" cy="490" rx="680" ry="490" fill="url(#wind-scene-horizon)"/>
    <ellipse cx="1420" cy="130" rx="390" ry="390" fill="url(#wind-scene-moonlight)"/>
    <g fill="#c5e7dc">{stars}</g>
    <g class="wind-scene__haze wind-scene__haze--far" fill="none" stroke="#c5e7dc">
      <path d="M 540 370 C 840 260, 1180 365, 1670 202" stroke-width="68" opacity="0.015"/>
      <path d="M 800 180 C 1080 92, 1250 187, 1640 117" stroke-width="27" opacity="0.025"/>
    </g>
    <g fill="none" stroke="#91e5ca" stroke-width="1">
      <path class="wind-scene__current" opacity="0.21"
            d="M 740 405 C 948 340, 1020 466, 1220 410 S 1460 270, 1700 310"/>
      <path class="wind-scene__current wind-scene__current--second" opacity="0.12"
            d="M 750 585 C 1020 457, 1250 588, 1680 427"/>
    </g>
    <path d="M 0 815 C 180 758, 285 835, 450 790 S 740 774, 914 742
             S 1170 785, 1350 735 S 1520 760, 1600 730 L 1600 1000 L 0 1000 Z"
          fill="#21483f" opacity="0.45"/>
    <path d="M 0 865 C 175 840, 268 877, 472 840 S 705 885, 920 818
             S 1240 867, 1430 808 S 1530 840, 1600 812 L 1600 1000 L 0 1000 Z"
          fill="#102f31" opacity="0.85"/>
    {turbines}
    <g class="wind-scene__haze" fill="none">
      <path d="M -80 895 C 370 858, 640 913, 1050 863 S 1490 866, 1710 834"
            stroke="#afd2bc" stroke-width="32" opacity="0.025"/>
      <path d="M 450 807 C 820 790, 1300 820, 1670 750"
            stroke="#bcd4bd" stroke-width="12" opacity="0.035"/>
    </g>
    <path d="M 0 940 C 150 898, 345 972, 550 938 S 840 990, 1080 935
             S 1380 931, 1600 906 L 1600 1000 L 0 1000 Z" fill="#0a232b"/>
    <g fill="#e3be84" opacity="0.35">
      <circle cx="986" cy="875" r="1.4"/>
      <circle cx="1004" cy="871" r="1"/>
      <circle cx="1177" cy="908" r="1.2"/>
      <circle cx="1187" cy="908" r="0.9"/>
      <circle cx="1452" cy="877" r="1.2"/>
    </g>
    <rect y="740" width="1600" height="260" fill="url(#wind-scene-foreground)"/>
    <rect width="1600" height="1000" fill="url(#wind-scene-leftshade)"/>
  </svg>
</div>
"""
