"""
Triton stubs for development on non-CUDA platforms (macOS/MPS, CPU).
"""
from types import ModuleType

__version__ = "3.5.0" #fake

class JITFunction:
    """Stub for Triton JIT-compiled functions."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "Triton kernels cannot run on this platform. "
            "This is a stub implementation for import compatibility only. "
            "Triton requires CUDA GPU support."
        )


def jit(fn=None, **kwargs):
    """Stub for @triton.jit decorator."""
    if fn is None:
        return lambda f: JITFunction(f)
    return JITFunction(fn)


def autotune(*args, **kwargs):
    """Stub for @triton.autotune decorator."""

    def decorator(fn):
        return fn

    return decorator


def heuristics(*args, **kwargs):
    """Stub for @triton.heuristics decorator."""

    def decorator(fn):
        return fn

    return decorator

# Common Triton types/constants
class Config:
    def __init__(self, kwargs, num_warps=4, num_stages=2):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_stages = num_stages


class LibDeviceStub:
    """Stub for triton.language.extra.libdevice module."""

    @staticmethod
    def __getattr__(name):
        """Return a stub function for any libdevice function."""

        def stub(*args, **kwargs):
            raise RuntimeError(f"Triton libdevice.{name} not available on this platform")

        return stub


class MathStub:
    """Stub for triton.language.math module."""

    @staticmethod
    def __getattr__(name):
        """Return a stub function for any math function."""

        def stub(*args, **kwargs):
            raise RuntimeError(f"Triton math.{name} not available on this platform")

        return stub

    # Explicitly define commonly used math functions
    @staticmethod
    def exp(*args, **kwargs):
        raise RuntimeError("Triton math.exp not available on this platform")

    @staticmethod
    def exp2(*args, **kwargs):
        raise RuntimeError("Triton math.exp2 not available on this platform")

    @staticmethod
    def log(*args, **kwargs):
        raise RuntimeError("Triton math.log not available on this platform")

    @staticmethod
    def log2(*args, **kwargs):
        raise RuntimeError("Triton math.log2 not available on this platform")

    @staticmethod
    def sqrt(*args, **kwargs):
        raise RuntimeError("Triton math.sqrt not available on this platform")

    @staticmethod
    def sin(*args, **kwargs):
        raise RuntimeError("Triton math.sin not available on this platform")

    @staticmethod
    def cos(*args, **kwargs):
        raise RuntimeError("Triton math.cos not available on this platform")

    @staticmethod
    def abs(*args, **kwargs):
        raise RuntimeError("Triton math.abs not available on this platform")

    @staticmethod
    def floor(*args, **kwargs):
        raise RuntimeError("Triton math.floor not available on this platform")

    @staticmethod
    def ceil(*args, **kwargs):
        raise RuntimeError("Triton math.ceil not available on this platform")


# Stub for triton.language (tl)
class LanguageStub:
    """Stub for triton.language module."""

    # Common functions
    @staticmethod
    def load(*args, **kwargs):
        raise RuntimeError("Triton operations not available on this platform")

    @staticmethod
    def store(*args, **kwargs):
        raise RuntimeError("Triton operations not available on this platform")

    @staticmethod
    def arange(*args, **kwargs):
        raise RuntimeError("Triton operations not available on this platform")

    @staticmethod
    def zeros(*args, **kwargs):
        raise RuntimeError("Triton operations not available on this platform")

    @staticmethod
    def program_id(*args, **kwargs):
        raise RuntimeError("Triton operations not available on this platform")

    @staticmethod
    def num_programs(*args, **kwargs):
        raise RuntimeError("Triton operations not available on this platform")

    @staticmethod
    def dot(*args, **kwargs):
        raise RuntimeError("Triton operations not available on this platform")

    @staticmethod
    def sum(*args, **kwargs):
        raise RuntimeError("Triton operations not available on this platform")

    @staticmethod
    def max(*args, **kwargs):
        raise RuntimeError("Triton operations not available on this platform")

    @staticmethod
    def exp(*args, **kwargs):
        raise RuntimeError("Triton operations not available on this platform")

    @staticmethod
    def sqrt(*args, **kwargs):
        raise RuntimeError("Triton operations not available on this platform")

    constexpr = staticmethod(lambda x: x)

class ExtraStub:
    """Stub for triton.language.extra module."""
    libdevice = LibDeviceStub()

# Create the language module
class _LanguageModule:
    def __init__(self):
        self.extra = ExtraStub()
        self.math = MathStub()

    def __getattr__(self, name):
        if hasattr(LanguageStub, name):
            return getattr(LanguageStub, name)

        # Return a dummy function for any other attribute
        def dummy(*args, **kwargs):
            raise RuntimeError(f"Triton language operation '{name}' not available on this platform")

        return dummy


class BackendsStub:
    """Stub for triton.backends module."""

    @staticmethod
    def __getattr__(name):
        """Return a stub for any backend attribute."""
        def stub(*args, **kwargs):
            raise RuntimeError(f"Triton backends.{name} not available on this platform")
        return stub


# Create a proper module-like object for triton that acts as a package
class TritonModule(ModuleType):
    """Main triton module stub that acts as a package."""

    def __init__(self, name):
        super().__init__(name)
        self.__package__ = name
        self.__path__ = []  # Makes this a package
        self.jit = jit
        self.autotune = autotune
        self.heuristics = heuristics
        self.Config = Config
        self.JITFunction = JITFunction
        self.__version__ = __version__
        self.language = _LanguageModule()
        self.backends = BackendsStub()


# Module-level exports for when imported as a module
Config = Config
language = _LanguageModule()
backends = BackendsStub()

# Common abbreviation
tl = language

__all__ = ['jit', 'autotune', 'heuristics', 'Config', 'language', 'backends', 'JITFunction', 'TritonModule']