def __getattr__(name):
    if name == "SplatRenderer":
        from .splat_renderer import SplatRenderer
        return SplatRenderer
    raise AttributeError(name)
