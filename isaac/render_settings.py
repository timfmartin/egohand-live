"""Viewport render presets for the sim scripts (``--render``).

Isaac Lab cannot choose the Kit renderer, so Isaac Sim starts in RTX Real-Time 2.0 with DLSS,
which NVIDIA's performance handbook says can be slower than older modes on Ampere GPUs. When
hand tracking shares the GPU, every millisecond of rendering is taken from EgoHand, so the
default here is the cheapest mode that still shows the scene clearly:

* ``minimal`` (default): RTX Minimal rendering (diffuse/glossy/emission shading, no ray-traced
  lighting) at 960x540. Lets tracking and sim run in real time on one laptop GPU.
* ``fast``: RTX Real-Time 2.0 without reflections, translucency, indirect diffuse, ambient
  occlusion and auto-exposure, DLSS in performance mode, 960x540.
* ``quality``: Isaac Sim's defaults at the window size (for videos and screenshots).
"""

PRESETS = {
    "minimal": {
        "/rtx/rendermode": "MinimalRendering",
        "/rtx/minimal/mode": 1,  # 1 diffuse/glossy/emission, 2 textured diffuse, 3 constant diffuse
    },
    "fast": {
        "/rtx/rendermode": "RealTimePathTracing",
        "/rtx/post/dlss/execMode": 0,  # performance
        "/rtx/reflections/enabled": False,
        "/rtx/translucency/enabled": False,
        "/rtx/indirectDiffuse/enabled": False,
        "/rtx/ambientOcclusion/enabled": False,
        "/rtx/post/histogram/enabled": False,
    },
    "quality": {},
}
RESOLUTION = {"minimal": (960, 540), "fast": (960, 540), "quality": None}


def add_args(parser):
    parser.add_argument("--render", default="minimal", choices=list(PRESETS),
                        help="viewport rendering: minimal (lightest, default), fast, or quality (full RTX, for videos)")  # fmt: skip
    parser.add_argument("--render-size", help="viewport render resolution WxH (default: 960x540, quality: window)")


def configure_launch(args):
    """Before ``launch_simulation``: set the render resolution on the launcher namespace."""
    size = tuple(int(v) for v in args.render_size.lower().split("x")) if args.render_size else RESOLUTION[args.render]
    if size:
        args.width, args.height = size


def apply(mode: str):
    """After the app started (inside ``launch_simulation``), before the first render."""
    import carb

    settings = carb.settings.get_settings()
    for key, value in PRESETS[mode].items():
        settings.set(key, value)
    print(f"[egohand] viewport rendering: {mode}")
