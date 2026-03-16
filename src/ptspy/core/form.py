"""Skia-backed drawing surface and helpers."""
import functools
import numpy as _cpu_numpy  # Permanent reference to CPU numpy for Skia boundary operations
from .. import np_wrapper as np
from typing import (
    Union, Any, Callable, Sequence, Optional,
    cast, Concatenate, ParamSpec, TypeVar, Tuple
)
from ptspy.typing import (
    ColorLike,
    StrokeJoinCap,
    PointStyle,
    IntArray,
    PArray,
    NumberSequence1D,
    NumberSequence2D,
    NumberSequenceND,
    ColorOrShader,
)

from .elements import P, toPArray, partial
from .util import toARGBInt, getSkiaProp
from .compose import gridPoints
from .geom import Rectangle, cardinalToBezier, hermiteToBezier, bsplineToBezier
import skia

import importlib.util

try:
    import glfw  # type: ignore[import-untyped]
except ImportError:
    glfw = None  # type: ignore[assignment]

_has_opengl = importlib.util.find_spec("OpenGL") is not None
if _has_opengl:
    from OpenGL import GL  # type: ignore[import-untyped]
else:
    GL = None  # type: ignore[assignment]

S = ParamSpec("S")
T = TypeVar("T")


def _requireOpenGL() -> tuple[Any, Any]:
    """Return the optional OpenGL modules, or raise with a clear install hint."""
    missing = []
    if GL is None:
        missing.append("PyOpenGL")
    if glfw is None:
        missing.append("glfw")

    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"OpenGL support requires optional dependencies ({names}). "
            "Install them with `pip install 'ptspy[opengl]'`."
        )

    return cast(Any, GL), cast(Any, glfw)


def _coerceSize(size: Sequence[int]) -> Tuple[int, int]:
    """Ensure a (w, h) of positive ints."""
    if len(size) < 2:
        raise ValueError("size must have at least two elements")
    w, h = map(int, size[:2])
    if w <= 0 or h <= 0:
        raise ValueError("width and height must be positive")
    return w, h


def frame(
    bg: ColorLike = "e5e5e7",
    size: tuple[int, int] = (200, 200),
) -> Callable[[Callable[Concatenate["Form", S], object]], Callable[S, skia.Image]]:
    """
    Decorate a function to draw and return a Skia image.

    This decorator provides a drawing context with size, background color, and
    passed a `Form` instance to the first parameter of the decorated function.
    The decorated function is called as `func(form, *args, **kwargs)`, and returns a skia.Image as
    default, but your drawing function can return something else convertible to an image.

    Args:
        bg (ColorLike, optional): Background color of the frame. Defaults to `"e5e5e7"`.
        size (tuple[int, int], optional): Width and height of the frame in pixels. This can be accessed as `form.width`, `form.height` and `form.size`. Defaults to `(200, 200)`.

    Returns:
        Callable: A decorator that wraps a function, injecting a `Form` instance as its first argument.
        The wrapped function returns a `skia.Image`. If the decorated function returns a `skia.Image`
        or a snapshot-able object (e.g. `skia.Surface`), that is used directly; otherwise the
        Form's canvas is snapshot automatically.

    Examples:
        >>> @frame(bg=0xFFFFFF, size=(256, 256))
        ... def draw(form, r):
        ...     form.circle(128, 128, r)
        >>> img = draw(r=100)  # returns a skia.Image
    """
    width, height = _coerceSize(size)

    def decorator(userFn):  # type: ignore[override]
        @functools.wraps(userFn)
        def wrapper(*args, **kwargs):  # type: ignore[override]
            form = Form(width=width, height=height)
            form.clear(bg)

            try:
                result = userFn(form, *args, **kwargs)
            except Exception as err:
                raise RuntimeError(f"Exception in @frame decorator: {type(err).__name__}: {err}") from err

            # 1) already an image
            if isinstance(result, skia.Image):
                return result
            # 2) something snapshot-able (e.g. skia.Surface)
            if hasattr(result, "makeImageSnapshot"):
                snap = result.makeImageSnapshot()
                if isinstance(snap, skia.Image):
                    return snap
            # 3) fallback
            return form.toImage()

        return wrapper

    return decorator


def frames(
    bg: ColorLike = "F0F3F9",
    size: Sequence[int] = (200, 200),
    numFrames: int = 30,
    *,
    clearBg: bool = True,
    startAt: int = 0,
    keepAlpha: bool = False,
) -> Callable[[Callable[Concatenate["Form", int, S], object]], Callable[S, IntArray]]:
    """
    Decorate a function to render multiple frames as an array for animation or video.
    Like `frame`, this decorator provides a drawing context with size, background color,
    and passes two parameters: an instance of `Form` class, and the current frame index.

    The decorated function is called as `func(form, index, *args, **kwargs)` for each frame, and
    returns an array of shape `(numFrames, height, width, channels)`, with RGBA or RGB channels based on `keepAlpha`.

    Args:
        bg (ColorLike, optional): Background color for frames. Defaults to a light gray `"F0F3F9"`.
        size (Sequence[int], optional): Width and height of each frame. Defaults to `(200, 200)`.
        numFrames (int, optional): Total number of frames to render. Defaults to `30`.
        clearBg (bool, optional): If True, clears background before each frame. Defaults to `True`.
        startAt (int, optional): Initial frame index for rendering sequences. Defaults to `0`.
        keepAlpha (bool, optional): If True, retains alpha channel in output frames. Defaults to `False`.

    Returns:
        Callable: A decorated function that receives a `Form` instance and frame index (int) as its
        first two arguments. Returns a uint8 array of shape ``(numFrames, height, width, channels)``
        where channels is 4 (RGBA) if ``keepAlpha=True``, else 3 (RGB).

    Examples:`
        >>> @frames(numFrames=10, size=(256, 256))
        ... def animate(form, t):
        ...     form.circle(128, 128, t * 5)
        >>> video = animate()  # returns array shape (10, 256, 256, 3)

    """

    if numFrames < 1:
        raise ValueError("numFrames must be >= 1")

    if startAt < 0:
        raise ValueError("startAt must be ≥ 0")

    width, height = _coerceSize(size)
    channels = 4 if keepAlpha else 3

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> IntArray:
            video: IntArray = _cpu_numpy.empty((numFrames, height, width, channels), dtype=_cpu_numpy.uint8)
            form = Form(width=width, height=height)

            for i in range(numFrames):
                if clearBg:
                    form.clear(bg)

                try:
                    fn(form, startAt + i, *args, **kwargs)
                except Exception as err:
                    raise RuntimeError(f"Error in frame {i} (t={startAt + i}): {type(err).__name__}: {err}") from err

                frameArr = form.toArray()
                video[i] = frameArr if keepAlpha else frameArr[..., :3]

            return video

        return wrapper

    return decorator


def _makePaint(self, mode: str):
    """Return a skia.Paint for 'Fill' or 'Stroke', or None if not configured."""
    shader = self.shaderParams.get(mode)
    color = getattr(self, f"{mode.lower()}Color", None)

    if shader is None and color is None:
        return None

    paint = skia.Paint(Shader=shader, **self.paintParams) if shader else skia.Paint(**self.paintParams)
    paint.setStyle(getattr(skia.Paint, f"k{mode}_Style"))  # ensure correct style
    if color is not None:
        paint.setColor(color)
    return paint


def _applyPaints(self, modes, func, *args, **kwargs):
    """Run *func* once per paint; restore previous self.paint afterwards."""
    previousPaint = getattr(self, "paint", None)
    drewSomething = False

    for mode in modes:
        paint = _makePaint(self, mode)
        if paint is None:
            continue
        self.paint = paint
        func(self, *args, **kwargs)
        drewSomething = True

    if previousPaint is not None:
        self.paint = previousPaint

    return drewSomething


def stroke_or_fill(func):
    """
    Internal decorator used in ptspy's drawing functions. You typically won't need to use this.

    This decorator checks if a shader or color is set for fill and stroke, applies the appropriate paint settings,
    and then executes the decorated function.

    Args:
        func (Callable): The drawing function to be decorated.

    Returns:
        Callable: The wrapped function that sets the paint style before execution.

    Data:
        internal: true
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not _applyPaints(self, ("Fill", "Stroke"), func, *args, **kwargs):
            func(self, *args, **kwargs)
        return self

    return wrapper


def stroke_only(func):
    """
    Internal decorator used in ptspy's drawing functions. You typically won't need to use this.

    This decorator checks if a shader or color is set for stroke, applies the appropriate paint settings,
    and then executes the decorated function.

    Args:
        func (Callable): The drawing function to be decorated.

    Returns:
        Callable: The wrapped function that sets the stroke style before execution.

    Data:
        internal: true
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not _applyPaints(self, ("Stroke",), func, *args, **kwargs):
            func(self, *args, **kwargs)
        return self

    return wrapper


def fill_only(func):
    """
    Internal decorator used in ptspy's drawing functions. You typically won't need to use this.

    This decorator checks if a shader or color is set for fill, applies the appropriate paint settings,
    and then executes the decorated function.

    Args:
        func (Callable): The drawing function to be decorated.

    Returns:
        Callable: The wrapped function that sets the fill style before execution.

    Data:
        internal: true
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not _applyPaints(self, ("Fill",), func, *args, **kwargs):
            func(self, *args, **kwargs)
        return self

    return wrapper


class Form:
    """
    Skia-backed drawing surface for 2D graphics.

    Form wraps a Skia canvas and provides methods to draw shapes, text, and images.
    Configure stroke and fill colors, then chain drawing calls. Most methods return
    `self` for fluent chaining.

    Typically created via the `@frame` decorator or `Form.setup()`:

        @frame(size=(400, 400))
        def draw(form):
            form.fill("#ff0000").circle([[50, 50], [150, 150]])

    Or manually:

        form = Form(width=400, height=400)
        form.fill("#ff0000").circle([[50, 50], [150, 150]])
        img = form.toImage()
    """

    @staticmethod
    def createRasterCanvas(width: int, height: int) -> skia.Canvas:
        """
        Create a skia.Canvas that uses raster as its backend. For convenience, you can also use [@frame](`pts/frame`) decorator instead.

        This method creates a `skia.Surface` with the specified width and height,
        and returns the canvas associated with that surface.

        Args:
            width (int): The width of the canvas.
            height (int): The height of the canvas.

        Returns:
            skia.Canvas: A canvas object for raster drawing.

        Examples:
            >>> canvas = Form.createRasterCanvas(800, 600)
        """
        surface = skia.Surface(width, height)
        return surface.getCanvas()

    @staticmethod
    def createGLCanvas(width: int, height: int) -> skia.Canvas:
        """
        Create a skia.Canvas that uses OpenGL as its backend. glfw.init() and a glfw window must be created before calling this function.

        Returns:
            skia.Canvas: the canvas

        Data:
            internal: true
        """
        gl, _ = _requireOpenGL()
        context = skia.GrDirectContext.MakeGL()
        backend_render_target = skia.GrBackendRenderTarget(
            width,
            height,
            0,
            0,
            skia.GrGLFramebufferInfo(0, gl.GL_RGBA8),  # sample count  # stencil bits
        )
        surface = skia.Surface.MakeFromBackendRenderTarget(
            context,
            backend_render_target,
            skia.kBottomLeft_GrSurfaceOrigin,
            skia.kRGBA_8888_ColorType,
            skia.ColorSpace.MakeSRGB(),
        )
        assert surface, "Failed to create a surface"
        return surface.getCanvas()

    @staticmethod
    def createGLWindow(width: int, height: int, title: str = "Demo") -> Any:
        """
        Create an OpenGL window with the specified width, height, and title.

        Args:
            width (int): The width of the window in pixels.
            height (int): The height of the window in pixels.
            title (str, optional): The title of the window. Default is "Demo".

        Returns:
            Any: The created OpenGL window context.

        Raises:
            RuntimeError: If OpenGL is not ready or if the window creation fails.

        Data:
            internal: true
        """
        _, glfwMod = _requireOpenGL()

        if not glfwMod.init():
            raise RuntimeError("glfw.init() failed")

        # glfw.window_hint(glfw.VISIBLE, glfw.FALSE) # hide the window
        glfwMod.window_hint(glfwMod.STENCIL_BITS, 8)
        window = glfwMod.create_window(width, height, title, None, None)
        if not window:
            glfwMod.terminate()
            raise RuntimeError("glfw.create_window() failed")
        glfwMod.make_context_current(window)
        glfwMod.swap_interval(1)
        return window

    @classmethod
    def setup(cls, width: int, height: int) -> "Form":
        """
        Create a Form object with a raster canvas backend.

        Args:
            width (int): The width of the canvas in pixels.
            height (int): The height of the canvas in pixels.

        Returns:
            Form: A Form object initialized with a raster canvas backend.

        Examples:
            >>> form = Form.setup(800, 600)
            >>> # Use the form object for drawing or other operations.
        """
        return cls(cls.createRasterCanvas(width, height))

    @classmethod
    def setupGL(cls, width: int, height: int, window: Any = None) -> "Form":
        """
        Create a Form object with an OpenGL canvas backend.

        Args:
            width (int): The width of the canvas in pixels.
            height (int): The height of the canvas in pixels.
            window (Any, optional): An existing OpenGL window context. If None, a new window will be created. Default is None.

        Returns:
            Form: A Form object initialized with an OpenGL canvas backend.

        Raises:
            RuntimeError: If OpenGL is not ready.

        Examples:
            >>> form = Form.setupGL(800, 600)
            >>> # Use the form object for OpenGL rendering.
        """
        _requireOpenGL()

        if window is None:
            window = cls.createGLWindow(width, height)
        form = cls(cls.createGLCanvas(width, height))
        form.window = window
        return form

    @staticmethod
    def renderGL(window: Any, renderFn: Any) -> None:
        """
        Render using an OpenGL context in a loop until the user closes the window.

        Args:
            window (Any): The OpenGL window context.
            renderFn (Any): A function that performs rendering. It should take the window context as an argument.

        Raises:
            RuntimeError: If OpenGL is not ready.

        Examples:
            >>> def render(window):
            >>>     # Your rendering code here
            >>>     pass
            >>> window = Form.createGLWindow(800, 600)
            >>> Form.renderGL(window, render)

        Data:
            internal: true
        """
        _, glfwMod = _requireOpenGL()

        # TODO: Provide exit hooks or time limits for long loops.

        # Loop until the user closes the window
        while not glfwMod.window_should_close(window):
            glfwMod.wait_events()

            # Render here
            renderFn(window)

            # Swap front and back buffers
            glfwMod.swap_buffers(window)

            # Poll for and process events
            glfwMod.poll_events()

        glfwMod.terminate()

    def __init__(self, canvas: skia.Canvas = None, width: int = 512, height: int = 512):
        """
        Initialize a Form object with a specified canvas, width, and height.

        Args:
            canvas (skia.Canvas, optional): An existing Skia canvas. If None, a new raster canvas is created with the specified width and height. Default is None.
            width (int, optional): The width of the canvas in pixels. Default is 512.
            height (int, optional): The height of the canvas in pixels. Default is 512.

        Examples:
            >>> form = Form(width=800, height=600)
            >>> # Use the form object for drawing.
        """
        self.canvas = canvas if canvas else Form.createRasterCanvas(width, height)
        self.window = None
        self.paintParams = {
            "Alpha": 255,
            "AntiAlias": True,
            "Color": 0xFFFFFFFF,
            "StrokeCap": skia.Paint.kRound_Cap,
            "StrokeJoin": skia.Paint.kRound_Join,
            "StrokeMiter": 0,
            "StrokeWidth": 1,
            "Style": skia.Paint.kStrokeAndFill_Style,
        }

        self.shaderParams = {
            "Fill": None,
            "Stroke": None,
        }

        self.paint = skia.Paint(**self.paintParams)
        self.fontStyle = skia.Font(skia.Typeface("Arial"), 12)
        self.fillColor: Optional[int] = 0x99AABBFF
        self.strokeColor: Optional[int] = 0xFF3366FF

    @property
    def width(self) -> int:
        """
        Get the width of the canvas.

        Returns:
            int: The width of the canvas in pixels.
        """
        return self.surface().width()

    @property
    def height(self) -> int:
        """
        Get the height of the canvas.

        Returns:
            int: The height of the canvas in pixels.
        """
        return self.surface().height()

    @property
    def size(self) -> PArray:
        """
        Get the size of the canvas as [width, height].

        Returns:
            PArray: A PArray of [width, height]
        """
        return P(self.surface().width(), self.surface().height())

    def stroke(
        self,
        color: Optional[ColorOrShader] = None,
        width: Optional[float] = None,
        cap: Optional[StrokeJoinCap] = None,
        join: Optional[StrokeJoinCap] = None,
        miter: Optional[float] = None,
        hasAlpha: Optional[bool] = None,
    ):
        """
        Configure stroke properties for subsequent drawing operations.

        Sets attributes like stroke color, line width, line cap style, line join style, and miter limit.
        If no argument is provided, the stroke color and shader are cleared (disables stroke).

        Args:
            color (ColorOrShader, optional): The stroke color. Can be an RGB/RGBA array-like object ([255,0,0,255]), an RGB/RGBA string ("#ff0000ff"), or an RGB/RGBA int (0xff0000ff) or a skia Shader. Default is None.
            width (float, optional): The stroke width. Default is None.
            cap (StrokeJoinCap, optional): The stroke cap style ("bevel", "round", "miter", or "last"). Default is None.
            join (StrokeJoinCap, optional): The stroke join style ("bevel", "round", "miter", or "last"). Default is None.
            miter (float, optional): The stroke miter limit. Default is None.
            hasAlpha (bool, optional): Optionally set whether the color value has alpha channel. This is only necessary for RGBA color integer less than 0xffffff, where heuristics can become ambiguous.

        Returns:
            Form: self.

        Examples:
            >>> form = Form()
            >>> form.stroke("#ff0000", width=2.0, cap="round", join="miter", miter=10.0)
            >>> # Use the form object for drawing with the specified stroke properties.
        """
        if color is not None:
            if isinstance(color, skia.Shader):
                self.shaderParams["Stroke"] = cast(skia.Shader, color)
                self.strokeColor = None
            else:
                self.shaderParams["Stroke"] = None
                self.strokeColor = toARGBInt(color, hasAlpha=hasAlpha)
        else:
            self.strokeColor = None
            self.shaderParams["Stroke"] = None

        if width is not None:
            if width < 0:
                raise ValueError("width must be >= 0")
            self.paintParams["StrokeWidth"] = width

        if cap is not None:
            self.paintParams["StrokeCap"] = getSkiaProp(f"stroke-cap-{cap}")
        if join is not None:
            self.paintParams["StrokeJoin"] = getSkiaProp(f"stroke-join-{join}")
        if miter is not None:
            self.paintParams["StrokeMiter"] = miter

        self.paint = skia.Paint(**self.paintParams)
        return self

    def fill(self, color: Optional[ColorOrShader] = None, hasAlpha: Optional[bool] = None):
        """
        Configure fill properties for subsequent drawing operations. Accepts a color value or a ``skia.Shader``.
        If no argument is provided, the fill color and shader are cleared (disables fill).

        Args:
            color (ColorOrShader, optional): The fill color or a [skia.Shader](https://kyamagu.github.io/skia-python/reference/skia.Shader.html) object. Color can be an RGB/RGBA array-like object ([255,0,0,255]), an RGB/RGBA string ("#ff0000ff"), or an RGB/RGBA int (0xff0000ff). Default is None.
            hasAlpha (bool, optional): Optionally set whether the color value has alpha channel. This is only necessary for RGBA color integer less than 0xffffff, where heuristics can become ambiguous.

        Returns:
            Form: self.

        Examples:
            >>> form.fill(0xFF00FFFF)  # Set fill color to pink
            >>> form.fill(skia.Shader.MakeLinearGradient(...))  # Set fill shader
        """
        if color is not None:
            if isinstance(color, skia.Shader):
                self.shaderParams["Fill"] = cast(skia.Shader, color)
                self.fillColor = None
            else:
                self.shaderParams["Fill"] = None
                self.fillColor = toARGBInt(color, hasAlpha=hasAlpha)
        else:
            self.fillColor = None
            self.shaderParams["Fill"] = None
        return self

    def fillOnly(self, color: Optional[ColorOrShader] = None, hasAlpha: Optional[bool] = None):
        """
        Set the fill color and disable the stroke for the form instance. See `fill` and `stroke` for details.

        Args:
            color (ColorOrShader, optional): The fill color or a [skia.Shader](https://kyamagu.github.io/skia-python/reference/skia.Shader.html) object. Color can be an RGB/RGBA array-like object ([255,0,0,255]), an RGB/RGBA string ("#ff0000ff"), or an RGB/RGBA int (0xff0000ff). Default is None.
            hasAlpha (bool, optional): Optionally set whether the color value has alpha channel. This is only necessary for RGBA color integer less than 0xffffff, where heuristics can become ambiguous.

        Returns:
            Form: self.

        Examples:
            >>> form = Form()
            >>> form.fillOnly("#ff0000")  # Set fill color to red and disable stroke
        """
        return self.fill(color, hasAlpha).stroke(None)

    def strokeOnly(
        self,
        color: Optional[ColorOrShader] = None,
        width: Optional[float] = None,
        cap: Optional[StrokeJoinCap] = None,
        join: Optional[StrokeJoinCap] = None,
        miter: Optional[float] = None,
        hasAlpha: Optional[bool] = None,
    ):
        """
        Set the stroke color and disable the fill for the form instance. See `fill` and `stroke` for details.

        Args:
            color (ColorOrShader, optional): The stroke color or a skia.Shader object. Color can be an RGB/RGBA array-like object ([255,0,0,255]), an RGB/RGBA string ("#ff0000ff"), or an RGB/RGBA int (0xff0000ff). Default is None.
            width (float, optional): The stroke width. Default is None.
            cap (StrokeJoinCap, optional): The stroke cap style. Can be "bevel", "round", "miter", or "last". Default is None.
            join (StrokeJoinCap, optional): The stroke join style. Can be "bevel", "round", "miter", or "last". Default is None.
            miter (float, optional): The stroke miter limit. Default is None.
            hasAlpha (bool, optional): Optionally set whether the color value has alpha channel. This is only necessary for RGBA color integer less than 0xffffff, where heuristics can become ambiguous.

        Returns:
            Form: self.

        Examples:
            >>> form = Form()
            >>> form.strokeOnly("#00ff00", width=2.0, cap="round", join="miter", miter=10.0)  # Set stroke properties and disable fill
        """
        return self.stroke(color, width, cap=cap, join=join, miter=miter, hasAlpha=hasAlpha).fill(None)

    def clear(self, color: ColorLike, alpha: float = 1.0) -> "Form":
        """
        Clear the canvas with the specified color.

        When alpha is 1.0 (default), fully replaces all pixels. When alpha < 1.0, draws the color
        at that opacity over the existing content, creating a fade/trail effect.

        Args:
            color (ColorLike): The color to clear the canvas with. Can be an RGB/RGBA array-like object ([255, 0, 0, 255]), an RGB/RGBA string ("#ff0000ff"), or an RGB/RGBA int (0xff0000ff).
            alpha (float): Opacity of the clear operation, from 0.0 (no-op) to 1.0 (full clear). Default is 1.0.

        Returns:
            Form: self.

        Examples:
            >>> form = Form()
            >>> form.clear("#ffffff")  # Clear the canvas with white color
            >>> form.clear("000", alpha=0.1)  # Fade toward black (motion trail effect)
        """
        argb = toARGBInt(color)
        if alpha >= 1.0:
            self.canvas.clear(argb)
        else:
            a = int(np.clip(alpha, 0.0, 1.0) * 255)
            faded = (a << 24) | (argb & 0x00FFFFFF)
            self.canvas.drawPaint(skia.Paint(Color=faded))
        return self

    def surface(self) -> skia.Surface:
        """
        Get the Skia surface associated with the canvas.

        Returns:
            skia.Surface: The surface of the canvas.
        """
        return self.canvas.getSurface()

    @stroke_only
    def dots(self, pts: NumberSequence1D):
        """
        Draw dots at the specified points on the canvas. For simple visualization of points, this method is faster than using `point()`.

        Args:
            pts (NumberSequence1D): An array of points. Can be of shape (n, 2) or (n, 2, 2).

        Returns:
            Form: self.

        Raises:
            ValueError: If `pts` is not of shape (n, 2) or (n, 2, 2).

        Examples:
            >>> form = Form()
            >>> points = P([10, 10], [20, 20], [30, 30])
            >>> form.dots(points)  # Draw dots at the specified points
        """
        pts = toPArray(pts)

        if pts.ndim == 3:
            if pts.shape[1:] != (2, 2):
                raise ValueError("pts must be of shape (n, 2) or (n, 2, 2)")
            pts = pts.reshape(-1, 2)

        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("pts must be of shape (n, 2) or (n, 2, 2)")

        self.canvas.drawPoints(
            skia.Canvas.PointMode.kPoints_PointMode, [skia.Point(p[0], p[1]) for p in pts], self.paint
        )
        return self

    @stroke_or_fill
    def point(self, pt: NumberSequence1D, radius: float = 5.0, style: PointStyle = "circle"):
        """
        Draw a point on the canvas with the specified style.

        Args:
            pt (NumberSequence1D): The coordinates of the point. Should be a 2-element array or list [x, y].
            radius (float, optional): The radius of the point. Default is 5.0.
            style (PointStyle, optional): The style of the point. Can be "circle" or "rect". Default is "circle".

        Returns:
            Form: self.

        Examples:
            >>> form = Form()
            >>> form.point([50, 50], radius=10.0, style="circle")  # Draw a circle point
            >>> form.point([100, 100], radius=5.0, style="rect")  # Draw a rectangle point
        """
        pt = toPArray(pt)
        if style == "circle":
            self.canvas.drawCircle(pt[0], pt[1], radius, self.paint)
        elif style == "rect":
            self.canvas.drawRect(skia.Rect.MakeXYWH(pt[0] - radius, pt[1] - radius, radius * 2, radius * 2), self.paint)
        return self

    def points(self, pts: NumberSequence1D, radius: float = 5.0, style: PointStyle = "circle"):
        """
        Draw multiple points on the canvas with the specified style.

        Args:
            pts (NumberSequence1D): Array of shape (n, 2) for [x, y] positions, or (n, 3) where the third column is a per-point radius override.
            radius (float, optional): The radius of each point. Default is 5.0.
            style (PointStyle, optional): The style of each point. Can be "circle" or "rect". Default is "circle".

        Returns:
            Form: self.

        Raises:
            ValueError: If `pts` is not of shape (n, 2) or (n, 3).

        Examples:
            >>> form = Form()
            >>> points = P([50, 50], [100, 100], [150, 150])
            >>> form.points(points, radius=10.0, style="circle")  # Draw multiple circle points
        """

        pts = toPArray(pts)

        if pts.ndim != 2 or pts.shape[1] not in (2, 3):
            raise ValueError("pts must be of shape (n, 2) or (n, 3)")

        # TODO: investigate if this can be vectorized instead of for-loop
        if pts.shape[1] > 2:
            for pt in pts:
                self.point(pt, pt[2], style)
        else:
            for pt in pts:
                self.point(pt, radius, style)

        return self

    @stroke_only
    def line(self, pts: NumberSequence1D):
        """
        Draw lines on the canvas.

        This method draws a continuous polyline if the input is of shape (n, 2), or multiple line segments if the input is of shape (n, 2, 2).

        Args:
            pts (NumberSequence1D): An array of points. Can be of shape (n, 2) to draw a continuous polyline, or shape (n, 2, 2) to draw multiple line segments.

        Returns:
            Form: self.

        Raises:
            ValueError: If `pts` is not of shape (n, 2) or (n, 2, 2).

        Examples:
            >>> form = Form()
            >>> continuous_line = P([10, 10], [20, 20], [30, 30])
            >>> form.line(continuous_line)  # Draw a continuous polyline
            >>> segments = P([[[10, 10], [20, 20]], [[30, 30], [40, 40]]])
            >>> form.line(segments)  # Draw multiple line segments
        """
        pts = toPArray(pts)

        if pts.ndim == 3:
            if pts.shape[1:] != (2, 2):
                raise ValueError(
                    "pts must be of shape (n, 2) for a continuous line. Or (n, 2, 2) for multiple line segments"
                )
            pts = pts.reshape(-1, 2)
            self.canvas.drawPoints(
                skia.Canvas.PointMode.kLines_PointMode, [skia.Point(p[0], p[1]) for p in pts], self.paint
            )
        elif pts.ndim == 2:
            if pts.shape[1] != 2:
                raise ValueError(
                    "pts must be of shape (n, 2) for a continuous line. Or (n, 2, 2) for multiple line segments"
                )
            self.canvas.drawPoints(
                skia.Canvas.PointMode.kPolygon_PointMode, [skia.Point(p[0], p[1]) for p in pts], self.paint
            )
        else:
            raise ValueError(
                "pts must be of shape (n, 2) for a continuous line. Or (n, 2, 2) for multiple line segments"
            )
        return self

    def lines(self, pts: NumberSequence1D):
        """
        Alias for `line()`. Reads better when drawing multiple independent segments via shape (n, 2, 2).
        """
        self.line(pts)
        return self

    @stroke_or_fill
    def rect(self, pts: NumberSequence2D):
        """
        Draw a rectangle on the canvas.

        This method draws a rectangle using the coordinates of two diagonal corners.

        Args:
            pts (NumberSequence2D): An array of points specifying the diagonal corners of the rectangle. Should be of shape (2, 2).

        Returns:
            Form: self.

        Examples:
            >>> form = Form()
            >>> corners = P([10, 10], [50, 50])
            >>> form.rect(corners)  # Draw a rectangle from (10, 10) to (50, 50)
        """
        if np.isarray(pts) and cast(PArray, pts).ndim != 2:
            raise ValueError("pts must be 2D")

        pts = toPArray(pts)
        self.canvas.drawRect(skia.Rect.MakeLTRB(pts[0][0], pts[0][1], pts[1][0], pts[1][1]), self.paint)
        return self

    def rects(self, pts: Union[PArray, Sequence[NumberSequence2D]]):
        """
        Draw multiple rectangles on the canvas.

        This method draws multiple rectangles using the coordinates of their diagonal corners.

        Args:
            pts (NumberSequence2D): An array of points specifying the diagonal corners of each rectangle. Should be of shape (n, 2, 2).

        Returns:
            Form: self.

        Raises:
            ValueError: If `pts` is not of shape (n, 2, 2).

        Examples:
            >>> form = Form()
            >>> rectangles = P([[[10, 10], [50, 50]], [[60, 60], [100, 100]]])
            >>> form.rects(rectangles)  # Draw two rectangles
        """
        pts = toPArray(pts)
        if pts.ndim != 3 or pts.shape[1:] != (2, 2):
            raise ValueError("pts must be of shape (n, 2, 2)")

        # TODO: investigate if this can be vectorized instead of for-loop
        for rect in pts:
            self.rect(rect)

        return self

    @stroke_or_fill
    def circle(self, pts: NumberSequence2D):
        """
        Draw a circle or ellipse inscribed in a bounding rectangle.

        The shape is defined by two diagonal corners. A square bounding box produces
        a circle; a rectangular one produces an ellipse.

        Args:
            pts (NumberSequence2D): Diagonal corners of the bounding rectangle as an array of shape (2, 2).

        Returns:
            Form: self.

        Examples:
            >>> form = Form()
            >>> form.circle([[10, 10], [50, 50]])
        """
        if np.isarray(pts) and cast(PArray, pts).ndim != 2:
            raise ValueError("pts must be 2D")

        pts = toPArray(pts)
        self.canvas.drawOval(skia.Rect.MakeLTRB(pts[0][0], pts[0][1], pts[1][0], pts[1][1]), self.paint)
        return self

    def circles(self, pts: Union[PArray, Sequence[NumberSequence2D]]):
        """
        Draw multiple circles or ellipses inscribed in bounding rectangles. Each shape is defined by two diagonal corners.

        Args:
            pts (NumberSequence2D): Bounding rectangles in an array of shape (n, 2, 2).

        Returns:
            Form: self.

        Examples:
            >>> form = Form()
            >>> bounds = np.array([[[10, 10], [50, 50]], [[60, 60], [100, 100]]])
            >>> form.circles(bounds)  # Draw two circles
        """
        pts = toPArray(pts)
        if pts.ndim != 3 or pts.shape[1:] != (2, 2):
            raise ValueError("pts must be of shape (n, 2, 2)")

        # TODO: investigate if this can be vectorized instead of for-loop
        for circle in pts:
            self.circle(circle)
        return self

    @stroke_or_fill
    def ellipse(
        self,
        pts: NumberSequence2D,
        rotation: Optional[float] = None,
        startAngle: Optional[float] = None,
        endAngle: Optional[float] = None,
        counterClockwise: Optional[bool] = False,
        wedge: Optional[bool] = False,
    ):
        """
        Draw an ellipse on the canvas.

        This method draws an ellipse using the coordinates of two diagonal corners of the bounding rectangle. Optionally, it can draw a rotated ellipse or an elliptical arc.

        Args:
            pts (NumberSequence2D): An array of points specifying the diagonal corners of the bounding rectangle. Should be of shape (2, 2).
            rotation (float, optional): The rotation angle in degrees. Default is None.
            startAngle (float, optional): The starting angle in degrees for the arc. Default is None.
            endAngle (float, optional): The ending angle in degrees for the arc. Default is None.
            counterClockwise (bool, optional): Whether the arc is drawn counterclockwise. Default is False.
            wedge (bool, optional): Whether to draw a wedge instead of an arc. Default is False.

        Returns:
            Form: self.

        Examples:
            >>> form = Form()
            >>> corners = P([10, 10], [50, 50])
            >>> form.ellipse(corners)  # Draw an ellipse
            >>> form.ellipse(corners, rotation=45)  # Draw a rotated ellipse
            >>> form.ellipse(corners, startAngle=0, endAngle=180)  # Draw an arc
        """
        pts = toPArray(pts)
        rect = skia.Rect.MakeLTRB(pts[0][0], pts[0][1], pts[1][0], pts[1][1])
        if rotation is not None:
            self.canvas.save()
            self.canvas.rotate(rotation, rect.centerX(), rect.centerY())

        if startAngle is None or endAngle is None:
            self.canvas.drawOval(rect, self.paint)
        else:
            sweep = endAngle - startAngle
            if counterClockwise:
                sweep = -sweep
            self.canvas.drawArc(rect, startAngle, sweep, wedge, self.paint)

        if rotation is not None:
            self.canvas.restore()

        return self

    def ellipses(
        self,
        pts: Union[PArray, Sequence[NumberSequence2D]],
        rotation: Optional[float] = None,
        startAngle: Optional[float] = None,
        endAngle: Optional[float] = None,
        counterClockwise: Optional[bool] = False,
        wedge: Optional[bool] = False,
    ):
        """
        Draw multiple ellipses on the canvas.

        This method draws multiple ellipses using the coordinates of their diagonal corners of the bounding rectangles. Optionally, it can draw rotated ellipses or elliptical arcs.

        Args:
            pts (NumberSequence2D): An array of points specifying the diagonal corners of each bounding rectangle. Should be of shape (n, 2, 2).
            rotation (float, optional): The rotation angle in degrees for each ellipse. Default is None.
            startAngle (float, optional): The starting angle in degrees for each arc. Default is None.
            endAngle (float, optional): The ending angle in degrees for each arc. Default is None.
            counterClockwise (bool, optional): Whether each arc is drawn counterclockwise. Default is False.
            wedge (bool, optional): Whether to draw wedges instead of arcs. Default is False.

        Returns:
            Form: self.

        Raises:
            ValueError: If `pts` is not of shape (n, 2, 2).

        Examples:
            >>> form = Form()
            >>> ellipses = P([[[10, 10], [50, 50]], [[60, 60], [100, 100]]])
            >>> form.ellipses(ellipses)  # Draw multiple ellipses
            >>> form.ellipses(ellipses, rotation=45)  # Draw multiple rotated ellipses
            >>> form.ellipses(ellipses, startAngle=0, endAngle=180)  # Draw multiple arcs
        """
        pts = toPArray(pts)
        if pts.ndim != 3 or pts.shape[1:] != (2, 2):
            raise ValueError("pts must be of shape (n, 2, 2)")

        # TODO: investigate if this can be vectorized instead of for-loop
        for el in pts:
            self.ellipse(el, rotation, startAngle, endAngle, counterClockwise, wedge)

        return self

    @stroke_or_fill
    def polygon(self, pts: NumberSequence2D, closed: bool = True):
        """
        Draw a polygon on the canvas.

        This method draws a polygon using the provided points. Optionally, the polygon can be closed to form a complete shape.

        Args:
            pts (NumberSequence2D): An array of points specifying the vertices of the polygon. Should be of shape (n, 2).
            closed (bool, optional): Whether to close the polygon by connecting the last point to the first. Default is True.

        Returns:
            Form: self.

        Examples:
            >>> form = Form()
            >>> vertices = P([10, 10], [50, 50], [50, 10])
            >>> form.polygon(vertices)  # Draw a closed triangle
            >>> form.polygon(vertices, closed=False)  # Draw an open polyline
        """
        if np.isarray(pts) and cast(PArray, pts).ndim != 2:
            raise ValueError("pts must be 2D")

        cpuPts = toPArray(pts)
        path = skia.Path()
        path.moveTo(cpuPts[0][0], cpuPts[0][1])
        for pt in cpuPts[1:]:
            path.lineTo(pt[0], pt[1])

        if closed:
            path.close()

        self.canvas.drawPath(path, self.paint)
        return self

    @stroke_or_fill
    def polygons(self, pts: Union[PArray, Sequence[NumberSequenceND]], closed: bool = True):
        """
        Draw multiple polygons on the canvas in a single draw call.

        Args:
            pts (NumberSequenceND): an array of polygons, of shape `(n, m, 2)`
            closed (bool): Whether to close each polygon. Default is True.

        Returns:
            Form: self.

        Examples:
            >>> form.polygons(P([[[10, 10], [50, 50], [50, 10]], [[60, 60], [100, 100], [100, 60]]]))
            >>> form.polygons([[[0,0], [10,0], [5,10]], [[20,0], [30,0], [30,10], [20,10]]])
        """
        # Convert to CPU numpy in one shot when pts is a single array (e.g. shape (n, m, 2)).
        # Skips ragged Python lists (different vertex counts per polygon) which can't be bulk-converted.
        if np.isarray(pts):
            pts = np.asnumpy(pts)
        path = skia.Path()
        for p in pts:
            poly = toPArray(p)
            if poly.shape[0] < 2:
                continue
            path.moveTo(float(poly[0, 0]), float(poly[0, 1]))
            for pt in poly[1:]:
                path.lineTo(float(pt[0]), float(pt[1]))
            if closed:
                path.close()

        self.canvas.drawPath(path, self.paint)
        return self

    def _drawCubicPath(self, pts, closed: bool = False):
        """Build and draw a cubic Bézier path. Converts to CPU numpy internally."""
        pts = toPArray(pts)
        path = skia.Path()
        path.moveTo(pts[0][0], pts[0][1])

        rest = pts[1:]
        keep = len(rest) - (len(rest) % 3)
        segments = rest[:keep].reshape(-1, 3, 2) if keep else _cpu_numpy.empty((0, 3, 2))

        for c1, c2, end in segments:
            path.cubicTo(*c1, *c2, *end)

        if closed:
            path.close()

        self.canvas.drawPath(path, self.paint)

    @stroke_or_fill
    def bezier(self, pts: NumberSequence2D, closed: bool = False):
        """
        Draw a 2D Bézier curve on the canvas.

        This method draws a Bézier curve using the provided control points. Optionally, the curve can be closed to form a complete shape.

        Expects points in flat format: ``[start, cp1, cp2, end, cp1, cp2, end, ...]``.
        If ``(len(pts) - 1)`` is not divisible by 3, trailing points are silently dropped.

        Args:
            pts (NumberSequence2D): An array of points specifying the control points of the Bézier curve. Should be of shape (n, 2).
            closed (bool, optional): Whether to close the curve by connecting the last point to the first. Default is False.

        Returns:
            Form: self.

        Examples:
            >>> form = Form()
            >>> control_points = P([10, 10], [20, 30], [30, 10], [40, 30])
            >>> form.bezier(control_points)  # Draw an open Bézier curve
            >>> form.bezier(control_points, closed=True)  # Draw a closed Bézier curve
        """
        pts = toPArray(pts)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("pts must be of shape (n, 2)")

        self._drawCubicPath(pts, closed)
        return self

    @stroke_or_fill
    def cardinal(self, pts: NumberSequence2D, tension: float = 0.5, closed: bool = False):
        """
        Draw a cardinal spline through the given control points.

        Converts cardinal spline points to cubic Bezier control points via
        ``cardinalToBezier``, then delegates to ``bezier()``.

        Args:
            pts (NumberSequence2D): Shape (n, 2) — the points the curve passes through.
            tension (float): Tension parameter (0–1). 0.5 is Catmull-Rom, 1.0 is straight lines.
            closed (bool): If True, the curve loops back to the first point.

        Returns:
            Form: self.

        Examples:
            >>> form = Form()
            >>> form.cardinal([[10, 10], [50, 80], [90, 10], [130, 80]])
        """
        bezPts = cardinalToBezier(pts, tension, closed)
        self._drawCubicPath(bezPts, closed)
        return self

    @stroke_or_fill
    def hermite(self, pts: NumberSequence2D, tangents: NumberSequence2D, closed: bool = False):
        """
        Draw a Hermite spline through the given points with explicit tangent vectors.

        Converts positions and tangents to cubic Bezier control points via
        ``hermiteToBezier``, then draws the resulting path.

        Args:
            pts (NumberSequence2D): Shape (n, 2) — positions the curve passes through.
            tangents (NumberSequence2D): Shape (n, 2) — tangent vectors at each point.
            closed (bool): If True, the curve loops back to the first point.

        Returns:
            Form: self.

        Examples:
            >>> form = Form()
            >>> form.hermite([[10, 50], [50, 10], [90, 50]], [[20, 0], [0, -20], [-20, 0]])
        """
        bezPts = hermiteToBezier(pts, tangents, closed)
        self._drawCubicPath(bezPts, closed)
        return self

    @stroke_or_fill
    def bspline(self, pts: NumberSequence2D, closed: bool = False):
        """
        Draw a uniform cubic B-spline through the given control points.

        Unlike cardinal/hermite splines, B-splines are *approximating* — the
        curve does **not** pass through the control points.

        Converts B-spline control points to cubic Bezier control points via
        ``bsplineToBezier``, then draws the resulting path.

        Args:
            pts (NumberSequence2D): Shape (n, 2) — B-spline control points.
                Minimum 4 for open, 3 for closed.
            closed (bool): If True, the curve loops back to the start.

        Returns:
            Form: self.

        Examples:
            >>> form = Form()
            >>> form.bspline([[10, 50], [40, 10], [75, 90], [110, 30], [140, 60]])
        """
        bezPts = bsplineToBezier(pts, closed)
        self._drawCubicPath(bezPts, closed)
        return self

    def font(
        self,
        size: int = 12,
        name: str = "Arial",
        bold: bool = False,
        italic: bool = False,
        weight: Optional[int] = None,
        width: Optional[int] = None,
    ):
        """
        Set the font style for text drawing.

        Args:
            size (int, optional): The size of the font. Default is 12.
            name (str, optional): The name of the font. Default is "Arial".
            bold (bool, optional): Whether the font is bold. Ignored if `weight` is specified. Default is False.
            italic (bool, optional): Whether the font is italic. Default is False.
            weight (int, optional): The font weight (100-900). Common values: 100 (Thin), 200 (ExtraLight), 300 (Light), 400 (Normal), 500 (Medium), 600 (SemiBold), 700 (Bold), 800 (ExtraBold), 900 (Black). If specified, overrides the `bold` parameter. Default is None.
            width (int, optional): The font width (1-9). Values: 1 (UltraCondensed), 2 (ExtraCondensed), 3 (Condensed), 4 (SemiCondensed), 5 (Normal), 6 (SemiExpanded), 7 (Expanded), 8 (ExtraExpanded), 9 (UltraExpanded). Default is None (Normal).

        Returns:
            Form: self.

        Examples:
            >>> form = Form()
            >>> form.font(size=16, name="Times New Roman", bold=True)  # Set font to 16pt Times New Roman, bold
            >>> form.font(size=20, name="Roboto", weight=500)  # Set font to 20pt Roboto Medium
            >>> form.font(size=18, name="Inter", weight=600, italic=True)  # Set font to 18pt Inter SemiBold Italic
        """
        # Determine font style
        if weight is not None or width is not None:
            # Use explicit weight and width values
            font_weight = weight if weight is not None else (700 if bold else 400)
            font_width = width if width is not None else 5  # 5 = Normal
            slant = skia.FontStyle.kItalic_Slant if italic else skia.FontStyle.kUpright_Slant
            style = skia.FontStyle(font_weight, font_width, slant)
        else:
            # Use predefined styles for backward compatibility
            if bold and italic:
                style = skia.FontStyle.BoldItalic()
            elif bold:
                style = skia.FontStyle.Bold()
            elif italic:
                style = skia.FontStyle.Italic()
            else:
                style = skia.FontStyle.Normal()

        self.fontStyle = skia.Font(skia.Typeface(name, style), size)
        self.fontStyle.setEdging(skia.Font.Edging.kSubpixelAntiAlias)
        return self

    @stroke_or_fill
    def text(self, pt: NumberSequence1D, txt: str):
        """
        Draw text on the canvas at the specified position.

        Args:
            pt (NumberSequence1D): The position to draw the text. Should be a 2-element array or list [x, y].
            txt (str): The text string to draw.

        Returns:
            Form: self.

        Examples:
            >>> form = Form()
            >>> form.font(size=16, name="Arial", bold=True)
            >>> form.text([50, 50], "Hello, World!")  # Draw text at position (50, 50)
        """
        pt = toPArray(pt)
        textblob = skia.TextBlob(txt, self.fontStyle)
        self.canvas.drawTextBlob(textblob, pt[0], pt[1], self.paint)
        return self

    @stroke_or_fill
    def colorMap(self, table: IntArray, rect: NumberSequence2D, style: PointStyle = "rect", useCeiling: bool = False):
        """
        Draw a color map that visualizes a 2D array of colors within a specified rectangle, using either rectangles or circles to represent each cell in the array.

        This is currently non-vectorized but can draw different styles (rectangles or circles). For large arrays, use `image` for better performance.

        Args:
            table (IntArray): A 2D array of colors of shape (n, m).
            rect (NumberSequence2D): The rectangle in which to draw the color map. Should be of shape (2, 2).
            style (PointStyle, optional): The style of each cell. Can be "rect" or "circle". Default is "rect".
            useCeiling (bool, optional): Whether to use the ceiling value for cell sizes. Default is False.

        Returns:
            Form: self.

        Raises:
            ValueError: If `table` is not of shape (n, m) or `rect` is not of shape (2, 2).

        Examples:
            >>> form = Form()
            >>> table = P([0xFF0000, 0x00FF00], [0x0000FF, 0xFFFF00]).astype("uint32")
            >>> rect = P([10, 10], [110, 110])
            >>> form.colorMap(table, rect, style="rect")  # Draw a color map with rectangles
        """
        if len(table.shape) == 1:
            table = table.reshape(1, -1)
        if len(table.shape) != 2:
            raise ValueError("table must be of shape (nRows, nCols)")

        rect_arr = toPArray(rect)
        if rect_arr.shape != (2, 2):
            raise ValueError("rect must be of shape (2, 2)")

        nRows, nCols = table.shape

        # compute each cell’s size
        cellSize = Rectangle.size(rect_arr) / [nRows, nCols]
        if useCeiling:
            cellSize = np.ceil(cellSize)

        # Build a (nRows, nCols, 2) array of top-left origins.
        # gridPoints / Rectangle use P() internally which creates arrays on the active backend (GPU when cupy).
        # Precompute all cell boxes as a single vectorized op, then transfer to CPU once before the Skia loop.
        origins = gridPoints(repeat=[nRows, nCols], spacing=cellSize).reshape(nRows, nCols, 2) + rect_arr[0]
        originsCpu = np.asnumpy(origins)
        cellSizeCpu = np.asnumpy(cellSize)
        tableCpu = np.asnumpy(table)
        boxes = _cpu_numpy.stack([originsCpu, originsCpu + cellSizeCpu], axis=-2)

        for i in range(nRows):
            for j in range(nCols):
                if style == "circle":
                    self.fillOnly(tableCpu[i, j]).circle(boxes[i, j])
                else:
                    self.fillOnly(tableCpu[i, j]).rect(boxes[i, j])

        return self


    def image(self, table: IntArray, rect: NumberSequence2D, filterMode: str | None = "nearest") -> "Form":
        """
        Draw a color table into a rectangle as a bitmap image.

        This is the vectorized version of `colorMap`, optimized for large arrays.

        Args:
            table (IntArray): A 2D array of colors of shape (n, m).
            rect (NumberSequence2D): The rectangle in which to draw the color map. Should be of shape (2, 2).
            filterMode (str, optional): The filtering mode for the color map. Can be "nearest" or "linear". Default is "nearest".

        Returns:
            Form: self.

        Raises:
            ValueError: If `table` is not of shape (n, m) or `rect` is not of shape (2, 2).

        Examples:
            >>> form = Form()
            >>> table = P([0xFF0000, 0x00FF00], [0x0000FF, 0xFFFF00]).astype("uint32")
            >>> rect = P([10, 10], [110, 110])
            >>> form.image(table, rect)
        """
        rect_arr = toPArray(rect)
        if rect_arr.shape != (2, 2):
            raise ValueError("rect must be of shape (2, 2)")

        tableCpu = np.asnumpy(table, dtype=_cpu_numpy.uint32)
        arr_xy = _cpu_numpy.atleast_2d(tableCpu)

        # swap axes
        arr = arr_xy.T
        nRows, nCols = arr.shape

        # pack to contiguous RGBA buffer
        rgba = _cpu_numpy.empty((nRows, nCols, 4), dtype=_cpu_numpy.uint8)
        rgba[..., 0] = (arr >> 16) & 0xFF  # R
        rgba[..., 1] = (arr >> 8) & 0xFF  # G
        rgba[..., 2] = arr & 0xFF  # B

        # Optimized alpha channel handling
        alpha_channel = arr >> 24
        if _cpu_numpy.any(alpha_channel != 0):
            # Some pixels have alpha values, use conditional assignment
            rgba[..., 3] = _cpu_numpy.where(alpha_channel == 0, 255, alpha_channel & 0xFF)
        else:
            # No alpha values present, use fast fill
            rgba[..., 3] = 255

        # wrap buffer in a Skia bitmap (must be on CPU)
        info = skia.ImageInfo.Make(
            int(nCols),
            int(nRows),
            skia.ColorType.kRGBA_8888_ColorType,
            skia.AlphaType.kUnpremul_AlphaType,
        )
        bmp = skia.Bitmap()
        bmp.installPixels(info, rgba, rgba.strides[0])
        self._pinnedColorMap = rgba  # keep CPU memory alive for Skia
        image = skia.Image.MakeFromBitmap(bmp)

        (x0, y0), (x1, y1) = rect_arr
        dst = skia.Rect.MakeLTRB(float(x0), float(y0), float(x1), float(y1))

        # sampling / filtering
        sampling = None
        if filterMode:
            try:
                fm = {"nearest": skia.FilterMode.kNearest, "linear": skia.FilterMode.kLinear}[filterMode]
                sampling = skia.SamplingOptions(fm, skia.MipmapMode.kNone)
            except AttributeError:
                paint = skia.Paint()
                paint.setFilterQuality(
                    skia.FilterQuality.kNone_FilterQuality
                    if filterMode == "nearest"
                    else skia.FilterQuality.kLow_FilterQuality
                )
                sampling = paint

        if isinstance(sampling, skia.SamplingOptions):
            self.canvas.drawImageRect(image, dst, sampling, paint=None)
        else:
            self.canvas.drawImageRect(image, dst, sampling)

        return self

    def toImage(self) -> skia.Image:
        """
        Create an image snapshot of the current canvas.

        Returns:
            skia.Image: An image snapshot of the current canvas, or None if the surface is not available.

        Examples:
            >>> form = Form()
            >>> image = form.toImage()
            >>> # Use the image for saving to a file or other operations.
        """
        img = self.canvas.getSurface().makeImageSnapshot()

        if img is None:
            raise RuntimeError("Failed to make a snapshot image.")
        return img

    def toArray(self) -> IntArray:
        """
        Convert the current canvas to a NumPy array.

        Returns:
            np.ndarray: A uint8 array of shape (height, width, 4) in RGBA channel order.

        Examples:
            >>> form = Form()
            >>> arr = form.toArray()
            >>> # Use the array for image processing or other operations.
        """
        arr = self.canvas.getSurface().toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        if arr is None:
            raise RuntimeError("Surface toarray() returned None.")

        return cast(IntArray, arr)

    def save(self) -> "Form":
        """
        Save the current canvas state (transformation matrix, clip region, etc.) onto a stack.
        Use `restore()` to pop the state back. This is useful when you want to temporarily
        modify the canvas state and then return to the previous state.

        Returns:
            Form: self.

        Examples:
            >>> form = Form()
            >>> form.save()
            >>> form.clip(path)
            >>> # ... draw something with clipping
            >>> form.restore()
        """
        self.canvas.save()
        return self

    def restore(self) -> "Form":
        """
        Restore the canvas state to the most recently saved state from the stack.
        This undoes any transformations or clipping applied after the corresponding `save()`.

        Returns:
            Form: self.

        Examples:
            >>> form = Form()
            >>> form.save()
            >>> form.clip(path)
            >>> form.restore()  # Remove the clip
        """
        self.canvas.restore()
        return self

    def translate(self, dx: float, dy: float) -> "Form":
        """
        Translate (move) the canvas origin by (dx, dy).

        Args:
            dx (float): Horizontal translation offset.
            dy (float): Vertical translation offset.

        Returns:
            Form: self.

        Examples:
            >>> form.save()
            >>> form.translate(50, 50)
            >>> form.point([0, 0], 3)  # Drawn at (50, 50)
            >>> form.restore()
        """
        self.canvas.translate(dx, dy)
        return self

    def rotate(self, degrees: float, pivot: Optional[NumberSequence1D] = None) -> "Form":
        """
        Rotate the canvas by degrees, optionally around a pivot point.

        Args:
            degrees (float): Rotation angle in degrees (clockwise).
            pivot (Optional[NumberSequence1D]): Center of rotation [x, y]. If None, rotates around origin.

        Returns:
            Form: self.

        Examples:
            >>> form.save()
            >>> form.rotate(45, pivot=[100, 100])
            >>> form.rect([[75, 75], [125, 125]])
            >>> form.restore()
        """
        if pivot is not None:
            pivot = toPArray(pivot)
            self.canvas.rotate(degrees, pivot[0], pivot[1])
        else:
            self.canvas.rotate(degrees)
        return self

    def scale(self, sx: float, sy: Optional[float] = None) -> "Form":
        """
        Scale the canvas by (sx, sy). If sy is None, scales uniformly by sx.

        Args:
            sx (float): Horizontal scale factor.
            sy (Optional[float]): Vertical scale factor. If None, uses sx for uniform scaling.

        Returns:
            Form: self.

        Examples:
            >>> form.save()
            >>> form.translate(100, 100).scale(2)
            >>> form.circle([[0, 0]])  # Drawn at 2x size
            >>> form.restore()
        """
        self.canvas.scale(sx, sy if sy is not None else sx)
        return self

    def skew(self, sx: float, sy: float) -> "Form":
        """
        Skew the canvas. Values are tangent factors, not degrees (e.g. ``sx=1`` shears x by 45 degrees).

        Args:
            sx (float): Horizontal skew factor (tan of the angle).
            sy (float): Vertical skew factor (tan of the angle).

        Returns:
            Form: self.

        Examples:
            >>> form.save()
            >>> form.translate(100, 100).skew(15, 0)
            >>> form.rect([[-20, -20], [20, 20]])
            >>> form.restore()
        """
        self.canvas.skew(sx, sy)
        return self

    def resetMatrix(self) -> "Form":
        """
        Reset the transformation matrix to identity. Does not affect the clip stack.

        Returns:
            Form: self.

        Examples:
            >>> form.translate(50, 50).rotate(45)
            >>> form.resetMatrix()  # Back to identity
        """
        self.canvas.resetMatrix()
        return self

    def concat(self, matrix: skia.Matrix) -> "Form":
        """
        Post-multiply the current transformation matrix by the given matrix (``current = current * matrix``).

        Args:
            matrix (skia.Matrix): The transformation matrix to apply.

        Returns:
            Form: self.

        Examples:
            >>> m = skia.Matrix()
            >>> m.setRotate(45, 100, 100)
            >>> form.concat(m)
        """
        self.canvas.concat(matrix)
        return self

    def clip(self, path: skia.Path, mode: str = "intersect", antialias: bool = True) -> "Form":
        """
        Set a clipping region using a path. All subsequent drawing operations will be
        confined to this region until `restore()` is called.

        Args:
            path (skia.Path): The path defining the clipping region.
            mode (str, optional): The clipping mode. Can be "intersect" or "difference". Default is "intersect".
            antialias (bool, optional): Whether to apply antialiasing to the clip edge. Default is True.

        Returns:
            Form: self.

        Examples:
            >>> circle = skia.Path()
            >>> circle.addCircle(100, 100, 50)
            >>> form.save().clip(circle)
            >>> form.fillOnly("blue").rect([[0, 0], [200, 200]])
            >>> form.restore()
        """
        if mode not in ("intersect", "difference"):
            raise ValueError(f"mode must be 'intersect' or 'difference', got {mode!r}")
        clip_op = skia.ClipOp.kIntersect if mode == "intersect" else skia.ClipOp.kDifference
        self.canvas.clipPath(path, clip_op, antialias)
        return self

    def erase(self, region: Optional[skia.Path] = None) -> "Form":
        """
        Erase pixels to fully transparent (both color and alpha) within the current clip region
        or a specified region. Useful for punching transparent holes in existing drawings.

        Args:
            region (skia.Path, optional): The region to erase. If None, erases within the
                current clip region. Default is None.

        Returns:
            Form: self.

        Examples:
            >>> form = Form()
            >>> form.fillOnly("blue").rect([[0, 0], [200, 200]])
            >>> circle = skia.Path()
            >>> circle.addCircle(100, 100, 50)
            >>> form.save().clip(circle).erase().restore()  # Punch a hole
        """
        clear_paint = skia.Paint(BlendMode=skia.BlendMode.kClear)

        if region is not None:
            self.canvas.save()
            self.canvas.clipPath(region, skia.ClipOp.kIntersect, True)
            self.canvas.drawPaint(clear_paint)
            self.canvas.restore()
        else:
            self.canvas.drawPaint(clear_paint)

        return self



@partial(optional=True)
def perlinNoise(
    freqX: float, freqY: float, octaves: int, seed: float, tileSize: skia.ISize = None, type: str = "fractal"
) -> skia.Shader:
    """
    Generates a Perlin noise shader.

    This function creates a Perlin noise shader using either fractal noise or turbulence noise. The generated shader can be used
    for various visual effects in graphics and design. You can then use [form.fill](`core/Form.fill`) to apply the shader.

    Args:
        freqX (float): The frequency of the noise in the X direction.
        freqY (float): The frequency of the noise in the Y direction.
        octaves (int): The number of octaves to use for the noise. Higher values result in more detail.
        seed (float): The seed value for randomization. Changing this value will result in different noise patterns.
        tileSize (skia.ISize, optional): The tile size for the noise. If provided, the noise will repeat seamlessly at this size.
        type (str, optional): The type of Perlin noise to generate. Can be "fractal" for fractal noise or "turbulence" for turbulence noise. Default is "fractal".

    Returns:
        skia.Shader: A Skia shader object that generates Perlin noise.

    Examples:
        >>> shader = perlinNoise(0.1, 0.1, 4, 42.0)
        >>> isinstance(shader, skia.Shader)
        True

        >>> shader_turbulence = perlinNoise(0.1, 0.1, 4, 42.0, type="turbulence")
        >>> isinstance(shader_turbulence, skia.Shader)
        True
    """

    if type not in ("fractal", "turbulence"):
        raise ValueError(f"type must be 'fractal' or 'turbulence', got {type!r}")
    if type == "turbulence":
        return skia.PerlinNoiseShader.MakeTurbulence(freqX, freqY, octaves, seed, tileSize)
    return skia.PerlinNoiseShader.MakeFractalNoise(freqX, freqY, octaves, seed, tileSize)
