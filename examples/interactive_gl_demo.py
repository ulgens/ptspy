"""Interactive OpenGL window with mouse-driven drawing using skia + ptspy.

Click to place points. 10 cardinal splines at different tensions (0..1) are drawn,
each at 0.2 opacity. Right-click to clear. Hover to see a cursor circle.

Requires OpenGL deps:  pip install 'ptspy[opengl]' (or: pip install glfw PyOpenGL)

If you're running from ptspy repo. You can do:
```bash
uv sync --extra dev --extra opengl
source .venv/bin/activate
```

Run:
    python examples/interactive_gl_demo.py
"""

import glfw
from OpenGL import GL
import skia
import numpy as np
from ptspy.core.form import Form
from ptspy import P

WIDTH, HEIGHT = 800, 600

# State
mouseX, mouseY = 0.0, 0.0
anchors: list[list[float]] = []


def onMouseMove(_win, x, y):
    global mouseX, mouseY
    mouseX, mouseY = x, y


def onMouseButton(win, button, action, _mods):
    if action != glfw.PRESS:
        return
    if button == glfw.MOUSE_BUTTON_LEFT:
        anchors.append([mouseX, mouseY])
    elif button == glfw.MOUSE_BUTTON_RIGHT:
        anchors.clear()


def draw(form: Form):
    form.clear("000")

    # Cursor ring
    r = 10
    form.strokeOnly("fff", width=1.5)
    form.circle([[mouseX - r, mouseY - r], [mouseX + r, mouseY + r]])

    if len(anchors) == 0:
        form.fillOnly("0ca").font(16).text([40, HEIGHT - 40], "Click to add points. Right-click to clear.")
        return

    pts = P(anchors)

    # Draw 10 cardinal curves at different tensions, each at 0.2 opacity
    if len(anchors) >= 3:
        for i in range(10):
            tension = i / 9
            form.strokeOnly("ffffff33", width=2, hasAlpha=True)
            form.cardinal(pts, tension=tension)
    elif len(anchors) >= 2:
        form.strokeOnly("ffffff33", width=2, hasAlpha=True).line(pts)

    # Draw anchor dots
    form.fillOnly("60f").points(pts, radius=7)


def main():
    if not glfw.init():
        raise RuntimeError("glfw.init() failed")

    glfw.window_hint(glfw.STENCIL_BITS, 8)
    window = glfw.create_window(WIDTH, HEIGHT, "Pts.py interactive demo", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("glfw.create_window() failed")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    glfw.set_cursor_pos_callback(window, onMouseMove)
    glfw.set_mouse_button_callback(window, onMouseButton)

    context = skia.GrDirectContext.MakeGL()
    fbInfo = skia.GrGLFramebufferInfo(0, GL.GL_RGBA8)
    target = skia.GrBackendRenderTarget(WIDTH, HEIGHT, 0, 0, fbInfo)
    surface = skia.Surface.MakeFromBackendRenderTarget(
        context,
        target,
        skia.kBottomLeft_GrSurfaceOrigin,
        skia.kRGBA_8888_ColorType,
        skia.ColorSpace.MakeSRGB(),
    )
    assert surface, "Failed to create GL surface"
    canvas = surface.getCanvas()
    form = Form(canvas)

    while not glfw.window_should_close(window):
        draw(form)
        surface.flushAndSubmit()
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()
