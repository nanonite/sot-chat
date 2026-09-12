"""Sketch-of-Thought package exports."""

__version__ = "1.0.0"
__all__ = ["SoT"]


def __getattr__(name):
    if name == "SoT":
        from .sketch_of_thought import SoT
        return SoT
    raise AttributeError(name)
