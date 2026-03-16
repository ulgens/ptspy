import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import python from "react-syntax-highlighter/dist/esm/languages/hljs/python";
import hljs from "../components/syntaxStyle";
import React from "react";
import asset_01 from "../assets/quickstart/qs_01_point.png";
import asset_02 from "../assets/quickstart/qs_02_points.png";
import asset_03 from "../assets/quickstart/qs_03_point_anim.mp4";
import asset_04 from "../assets/quickstart/qs_04_shuffle_anim.mp4";
import asset_05 from "../assets/quickstart/qs_05_color_cube.png";
import asset_06 from "../assets/quickstart/qs_06_color_rect.png";
import asset_07 from "../assets/quickstart/qs_07_color_palette.png";
import asset_08 from "../assets/quickstart/qs_08_color_interpolate.png";
import asset_09 from "../assets/quickstart/qs_09_color_field.png";
import asset_10 from "../assets/quickstart/qs_10_color_gradient.png";
import asset_11 from "../assets/quickstart/qs_11_steps.png";
import asset_12 from "../assets/quickstart/qs_12_steps_anim.mp4";
import asset_13 from "../assets/quickstart/qs_13_partial_anim.mp4";
import asset_14 from "../assets/quickstart/qs_14_diamonds.png";
import asset_15 from "../assets/quickstart/qs_15_hexagon_anim.mp4";
import asset_16 from "../assets/quickstart/qs_16_stripe_anim.mp4";
import { CTAButton } from "./Home";

SyntaxHighlighter.registerLanguage("python", python);

export function ImagePreview({
  src,
  isVideo,
  alt,
}: {
  src: string;
  isVideo?: boolean;
  alt: string;
}) {
  if (isVideo) {
    return (
      <div className="object-contain xl:w-[300px] w-full aspect-square bg-gray-200 overflow-hidden rounded-lg">
        <video src={src} aria-label={alt} autoPlay muted loop playsInline />
      </div>
    );
  }
  return (
    <div className="object-contain xl:w-[300px] w-full aspect-square overflow-hidden rounded-lg">
      <img src={src} alt={alt} />
    </div>
  );
}

export function ConsolePreview({ children }: { children: React.ReactNode }) {
  return (
    <div className="xl:w-[300px] w-full p-4 min-h-[120px] rounded-lg bg-white">
      <pre>{children}</pre>
    </div>
  );
}

export function CodeExample({
  children,
  code,
  reverse = false,
}: {
  children: React.ReactNode;
  code: string;
  reverse?: boolean;
}) {
  return (
    <div
      className={`flex ${
        reverse ? "xl:flex-row-reverse" : "xl:flex-row"
      } flex-col-reverse gap-4 my-12 overflow-visible relative`}
    >
      <div
        className={`flex-1 xl:absolute xl:bottom-0 ${
          reverse ? "xl:left-[-320px]" : "xl:right-[-320px]"
        }`}
      >
        {children}
      </div>
      <SyntaxHighlighter
        language="python"
        className="hljs rounded-lg lg:p-6 p-2 lg:text-sm text-xs flex-1 lg:w-[500px] w-[90dvw] overflow-x-auto"
        style={hljs}
        showLineNumbers
        lineNumberStyle={{ color: "#404449" }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}

export default function Start() {
  return (
    <div className="prose mx-4 lg:mx-2 mt-8 max-w-prose ptspy-mdcontent">
      <h1 className="mb-10 xl:mb-16 mt-[3rem] xl:mt-[6rem] font-medium text-[5rem] leading-[4.8rem] xl:text-[14rem] xl:leading-[13.5rem] ">
        Start
      </h1>
      <p>
        Welcome to Pts.py! The easiest way to install ptspy is via{" "}
        <code>pip install ptspy</code>. If you want to run from source, check
        out the{" "}
        <a
          href="https://github.com/williamngan/ptspy"
          target="_blank"
          rel="noopener noreferrer"
        >
          github repo
        </a>
        . You can also play with the examples in this guide interactively with
        the{" "}
        <a
          href="https://github.com/williamngan/ptspy/tree/main/notebooks"
          target="_blank"
          rel="noopener noreferrer"
        >
          QuickStart notebook
        </a>
        .
      </p>
      <p>
        If you're not familiar with python and numpy, Pts.py is a good way to
        learn to code visually. Just download or clone the{" "}
        <a
          href="https://github.com/williamngan/ptspy"
          target="_blank"
          rel="noopener noreferrer"
        >
          github project
        </a>
        , and ask an AI agent to guide you through. A basic AGENTS.md is
        included to get you started, and the repo has beginner-friendly
        instructions.
      </p>
      {/* ONE */}
      <h2 className="mt-12 mb-4">Make a point</h2>
      <p>Begin from the beginning, let's begin with a point.</p>
      <p>
        Ptspy makes things simple. To create an image, you start with a function
        that has <code>form</code> as the first parameter, which is an instance
        of the <code>Form</code> class for drawing. Then, decorate the function
        with <code>@frame</code> and call it. That's it!
      </p>
      <CodeExample
        code={`
@frame()
def draw(form):
    form.fillOnly("000").point([100, 100], 10)

draw()`.trimStart()}
        reverse={true}
      >
        <ImagePreview src={asset_01} alt="point image" />
      </CodeExample>
      {/* TWO */}
      <p>
        The default size is (200, 200) and so the center is at (100,100). Easy
        to remember. You can customize this by passing <code>size=(w, h)</code>{" "}
        parameter in <code>@frame</code>.
      </p>
      <p>
        Here we set the frame's dimensions and color by passing parameters to
        <code>@frame</code>, and introduce a <code>radius</code> parameter to
        draw circles at each corner of a hexagon. 5 lines of code and that's it.
      </p>
      <p>
        This means you can also use UI widgets such as <code>ipywidgets</code>{" "}
        to connect the parameters and make it interactive.
      </p>
      <CodeExample
        code={`@frame(bg="000", size=(900, 900))
def draw(form, radius):
    points = regularPolygon(form.size/2, 6, radius)
    form.fillOnly("fff").points(points, radius/2)

draw(200)`.trimStart()}
        reverse={true}
      >
        <ImagePreview src={asset_02} alt="circles in a hexagonal arrangement" />
      </CodeExample>
      {/* THREE */}
      <p>
        Making an animation is just as easy and concise. Instead of{" "}
        <code>@frame</code>, we'll use
        <code>@frames</code> and render a number of frames. In our draw
        function, we'll add a second parameter <code>index</code> to receive the
        current frame index. Here we use `mediapy` package to render the video
        frames into a video.
      </p>
      <CodeExample
        code={`@frames(numFrames=60)
def draw(form, index, radius):
    form.fillOnly("000").point(form.size/2, radius + index)

video = draw(radius=10)
mediapy.show_video(video, codec="h264", fps=30)`.trimStart()}
        reverse={true}
      >
        <ImagePreview src={asset_03} alt="point animation" isVideo />
      </CodeExample>
      {/* FOUR */}
      <h2 className="mt-12 mb-4">Connect the dots</h2>
      <p>
        A point is the simplest representation of a concept. The meaning of a
        point depends on the definition of a space. To create points in
        n-dimensional space, use <code>P</code> as a shorthand for{" "}
        <code>np.array</code> function.
      </p>
      <CodeExample code={`P(1, 2, 3)`.trimStart()}>
        <ConsolePreview>array([1., 2., 3.])</ConsolePreview>
      </CodeExample>
      <p>
        <code>P</code> is short and flexible. It can take lists, tuples, or
        floats and returns an array of float values to represent points in
        space.
      </p>
      <CodeExample code={`P((1, 2), (3, 4))`.trimStart()}>
        <ConsolePreview>
          {`array([
  [1., 2.],
  [3., 4.]
])`.trimStart()}
        </ConsolePreview>
      </CodeExample>
      {/* FIVE */}
      <p>
        <code>P2</code> is a shortcut for creating a grid of 2D points, and
        <code>P3</code> will create a cube of 3D points.
      </p>
      <p>
        Numpy is used as default, so you have access to all the numpy features.
        In future versions, you can also swap it to another numpy-like library,
        like cupy.
      </p>
      <CodeExample
        code={`grid = P2([1, 2], [4, 5])
points = grid.reshape(-1, 2).T
print(pretty(points))`.trimStart()}
      >
        <ConsolePreview>
          {`[
  [1.0, 2.0, 1.0, 2.0],
  [4.0, 4.0, 5.0, 5.0]
]`.trimStart()}
        </ConsolePreview>
      </CodeExample>

      {/* SIX */}
      <p>
        Ptspy comes with many convenient functions. For example, quickly make a
        series for testing with <code>populate</code>. Take a look at the
        <code>compose</code> module for other ways to create structures.
      </p>
      <CodeExample code={`populate(4, 0.1, 3)`.trimStart()}>
        <ConsolePreview>
          {`array([0.1, 3.1, 6.1, 9.1])`.trimStart()}
        </ConsolePreview>
      </CodeExample>

      {/* SEVEN */}
      <p>
        `random` is a handy function to generate a single random number, or a
        range of uniformly distributed numbers.
      </p>
      <CodeExample code={`print(pretty(random((3, 1))))`.trimStart()}>
        <ConsolePreview>
          {`[
  [0.644405773855445],
  [0.594948845737958],
  [0.33845925210247585]
]`.trimStart()}
        </ConsolePreview>
      </CodeExample>

      {/* EIGHT */}
      <p>
        <code>randomizer</code> returns a random generator. Use it to access the
        random methods and distribution options in numpy.{" "}
      </p>
      <p>Let's visualize a simple shuffling of numbers in a quick video.</p>
      <CodeExample
        code={`@frames(bg="000", size=(300, 150), numFrames=30)
def draw(form, index, rand):
    row = P( [50, 50], [100, 50], [150, 50], [200, 50], [250, 50] )
    row += P(-8, 5) # adjust position
    values = populate(row.shape[0], 1, 1)

    row2 = row + P(0, 50)
    shuffled = np.copy(values)
    rand.shuffle(shuffled)

    form.font(24).fillOnly("fff")
    for pos, val, pos2, val2 in zip(row, values, row2, shuffled):
        form.text(pos, str(int(val))).text(pos2, str(int(val2)))

video = draw(rand=randomizer())
mediapy.show_video(video, codec="h264", fps=10)`.trimStart()}
        reverse={true}
      >
        <ImagePreview
          src={asset_04}
          alt="shuffling numbers animation"
          isVideo={true}
        />
      </CodeExample>

      {/* FIFTEEN */}
      <h2 className="mt-12 mb-4">Form follows function()</h2>
      <p>
        How do we make a visual structure in code? We use functions to compose
        it, and we use functions to change it. Instead of thinking inputs and
        outputs, we can think of it as the relationship of possibilities.
      </p>
      <p>
        Here's a simple example of distributing points on a path, using the
        <code>steps</code> function.
      </p>
      <CodeExample
        code={`@frame(bg="000", size=(300, 300))
def draw(form):
    points = steps([40, 150], [form.width-40, 150],
                   count=20, shaping="cubicOut")
    form.fillOnly("fff").points(points, 2)

draw()`.trimStart()}
        reverse={true}
      >
        <ImagePreview
          src={asset_11}
          alt="stepping points with cubicOut shaping"
        />
      </CodeExample>

      {/* SIXTEEN */}
      <p>
        What if we want to distribute points, but don't know where it'll go and
        how many there will be? An incomplete function returns a new function,
        rescoping the range of possibilities.
      </p>
      <p>
        Here the <code>step</code> function starts out incomplete and commits
        later on in the animation loop.
      </p>
      <CodeExample
        code={`# you can skip positional argument or
# use TBD token to set a parameter placeholder (eg, count=TBD)
pointsFn = steps([40, 150], shaping="cubicOut")

@frames(bg="000", size=(300, 300), numFrames=60)
def draw(form, index):
    count = int(index/2.0)
    end = [form.width-40, 40 + index/60.0 * (form.height - 80)]
    form.fillOnly("fff").points(pointsFn(end, count), 2)

video = draw()
mediapy.show_video(video, codec="h264", fps=30)`.trimStart()}
        reverse={true}
      >
        <ImagePreview
          src={asset_12}
          alt="an animation of stepping points"
          isVideo={true}
        />
      </CodeExample>

      {/* SEVENTEEN */}
      <h2 className="mt-12 mb-4">Partial functions</h2>
      <p>
        Most of the functions in ptspy can be applied partially. There's also a
        convenient
        <code>@partial</code> decorator to define your own partials.
      </p>
      <p>
        In this example, we first set up polygons with specific sides and
        normalized positions, then fill in the size and rotation parameters when
        they are available.
      </p>
      <CodeExample
        code={`@partial(placeholder=TBD)
def createPolygon(form, center, sides, rotation):
    poly = regularPolygon(
        center=center*form.size, radius=form.width/6,
        sides=sides, closed=True
    )
    form.save()
    form.rotate(rotation, pivot=center*form.size)
    form.fillOnly("fff", 2).polygon( poly )
    form.restore()

# create 4 polygons in a grid but set size and rotation later
centers = P([0.33, 0.33], [0.66, 0.33], [0.33, 0.66], [0.66, 0.66])
polygons = [createPolygon(form=TBD, center=c, sides=s)
            for c, s in zip(centers, [4,5,6,7])]

@frames(bg="000", size=(300, 300), numFrames=90)
def draw(form, index):
    for poly in polygons:
        poly(form=form, rotation=index*4)

video = draw()
mediapy.show_video(video, codec="h264", fps=30)`.trimStart()}
        reverse={true}
      >
        <ImagePreview src={asset_13} alt="two color discs" isVideo={true} />
      </CodeExample>

      {/* EIGHTEEN */}
      <p>
        Partial function may feel like a simple intuition, but it holds a deeper
        insight about creative work. Outcomes converge and diverge with each
        parameter you set, each function you apply. We wander in an infinite
        field of possibilities, seeking hidden relationships between disparate
        ideas.{" "}
      </p>

      <h2 className="mt-12 mb-4">Sequence of functions</h2>

      <p>
        Another common functional practice is that of a pipeline, often
        confusingly called `pipe` and `compose`, but the idea is simple enough:
        Compose a series of functions so that the output of one function becomes
        the input into another function.
      </p>
      <CodeExample
        code={`def a(x,n):
    return P(x, x*n)

def b(p):
    # return a tuple to pass multiple args to next function
    return (P(p, p+100), 5)

def c(path, repeat):
    return steps(path[0], path[1], repeat)

# You can also write this using syntax: \`F(a) >> b >> c\`
# or reversePipeline as: \`F(a) << b << c\`
fn = pipeline(a, b, c)
fn(10, 2)`.trimStart()}
      >
        <ConsolePreview>
          {`array([[ 10.,  20.],
       [ 35.,  45.],
       [ 60.,  70.],
       [ 85.,  95.],
       [110., 120.]])`.trimStart()}
        </ConsolePreview>
      </CodeExample>

      {/* NINETEEN */}
      <p>
        This means we can deconstruct a complex shape into simpler components,
        and reconstruct them in various ways. We are the puzzle maker and the
        puzzle player.
      </p>
      <CodeExample
        code={`makePolygon = regularPolygon(TBD, sides=4, radius=40, closed=True)

def getNext(form, pts):
    results = []
    for pt in pts:
        next = makePolygon(pt)
        form.strokeOnly("fff", 1).lines(next)
        results.append(next)
    # auto-unpack tuples as parameters for next function
    return (form, np.vstack(results))

def makeDot(form, pts):
    form.fillOnly("fff").points(pts, 3)
    return (form, pts)

# same as pipeline(getNext, makeDot, ...)
makeAll = F(getNext) >> makeDot >> getNext

@frame(bg="000", size=(300, 300))
def draw(form):
    poly = makePolygon([150,150])
    form.strokeOnly("fff", 1).lines(poly)
    makeAll(form, poly)

draw()`.trimStart()}
        reverse={true}
      >
        <ImagePreview src={asset_14} alt="an arrangement of diamond shapes" />
      </CodeExample>

      {/* TWENTY */}
      <p>
        Now for a more complex example, you can use <code>lazy=True</code> to
        get an array of generator functions of each intermediate step. This can
        be useful for debugging and animation. Pipeline also supports additional
        features such as <code>async</code> and <code>trace</code>.
      </p>
      <CodeExample
        code={`makePolygon = regularPolygon(TBD, TBD, radius=30, closed=True)

def getNext(pts, sides):
    if len(pts.shape) > 2:
        pts = pts.reshape(-1, 2)
    return (P([makePolygon(center, sides) for center in pts]), sides)

def makeDot(pts, sides):
    return (pts.reshape(-1, 2), sides)

# get intermediate steps in the pipeline as a generator
makeAll = pipeline(getNext, getNext, getNext, makeDot, lazy=True)
progress = list(makeAll(P(150,150), 6))

@frames(bg="000", size=(300, 300), numFrames=20)
def draw(form, index):
    i = index % len(progress)
    if len(progress[i][0].shape) > 2:
        form.strokeOnly("fff")
        form.lines(progress[i][0].reshape(-1, 2))
    else:
        form.fillOnly("fff").points(progress[i][0], 2)

video = draw()
mediapy.show_video(video, codec="h264", fps=2)`.trimStart()}
        reverse={true}
      >
        <ImagePreview
          src={asset_15}
          alt="an animation of expanding hexagon shapes"
          isVideo={true}
        />
      </CodeExample>

      {/* CONCLUSION */}
      <p>
        Also look for other convenient functions in <code>process</code> and{" "}
        <code>compose</code> modules. Let your imagination connect the dots!
      </p>
      <CodeExample
        code={`@frames(bg="000", size=(300, 300), numFrames=60)
def draw(form, index):
    top = steps(0, form.width, 20,
                shaping=lambda t, a, b: a+(b-a) * (t*t*index/30.0))
    # reshape with new axis and append 0. shape is now (n,2)
    top =  expand(top[:, np.newaxis], 0)
    lines = np.stack([top, top+[0, form.height]], axis=1)

    form.strokeOnly("fff", 10-index/10.0).lines(lines)

video = draw()
mediapy.show_video(video, codec="h264", fps=30)`.trimStart()}
        reverse={true}
      >
        <ImagePreview src={asset_16} alt="a stripe animation" isVideo={true} />
      </CodeExample>

      {/* NINE */}
      <h2 className="mt-12 mb-4">Find the right colors</h2>
      <p>
        Color spaces are unintuitive and color choices are subjective. How do we
        make it easier?
      </p>
      <p>
        Letting go of technical jargons and preconceptions, an intuitive way to
        think about color can be: what the hue is, how vivid it is, and how
        light it is. This maps intuitively to a 3D cube with 3 axes: hue,
        chroma, lightness.
      </p>
      <p>
        Here we aim for intuition over perfection - Decide how many steps you
        need in each dimension, make a cube with <code>colorCube()</code>, and
        slice it to get a palette.
      </p>
      <CodeExample
        code={`@frame(size=(300, 300))
def draw(form, steps):
    cube = colorCube(steps)
    slice = cube[:, 14, :] # middle chroma slice
    form.image(slice, P([0, 0], form.size))

draw(steps=25)`.trimStart()}
        reverse={true}
      >
        <ImagePreview src={asset_05} alt="a slice of the color cube" />
      </CodeExample>

      {/* TEN */}
      <p>
        <code>colorPicker()</code> can retrieve a color in a color cube. Using
        values between 0 to 1, you can easily get to nuanced color variations.
      </p>
      <CodeExample
        code={`@frame(size=(300, 300))
def draw(form, steps):
    cube = colorCube(steps)
    form.fillOnly( colorPicker(cube, 0.20, 0.9, 0.7) )
    form.rect(P([[0,0], form.size]))
    form.fillOnly( colorPicker(cube, 0.18, 0.9, 0.8) )
    form.rect(P([form.size * 0.25, form.size * 0.75]))

draw(30)`.trimStart()}
        reverse={true}
      >
        <ImagePreview src={asset_06} alt="point image" />
      </CodeExample>

      {/* ELEVEN */}
      <p>
        For an even simpler starting point, apply <code>colorScheme</code> to
        your
        <code>colorCube</code>. This gives you a practical color palette with
        consistent steps from dark to light.
      </p>
      <p>
        The <code>intensity</code> parameter (from 0 to 1) adjusts both
        lightness and vividity as a whole. If you passed a random generator to{" "}
        <code>variant</code>, the colors will vary by random each time.
      </p>
      <CodeExample
        code={`@frame(bg="000", size=(300, 300))
def draw(form, intensity=0.5, hues=9, swatches=9, variantSeed=0):
    cube = colorCube(steps=24)
    variant = randomizer(variantSeed)
    palette = colorScheme(cube, hues, swatches, intensity, variant)
    form.colorMap(palette[:, :, 0], P([0, 0], form.size),
                  style="circle")

draw(variantSeed=8)  # random variant with seed 8`.trimStart()}
        reverse={true}
      >
        <ImagePreview src={asset_07} alt="color palette generation" />
      </CodeExample>

      {/* TWELVE */}
      <p>
        Another use of the <code>colorCube</code> is to interpolate between two
        colors. We apply <code>colorInterpolation</code> to get the indices
        between two color swatches.
      </p>
      <p>You can interpolate them as linear paths:</p>
      <CodeExample
        code={`@frame(bg="000", size=(45, 205))
def draw(form):
    cc = colorCube(steps=100)
    # batch 2 interpolations in one call
    startColors = [P(80, 28, 99), P(50, 90, 50)] # hue, chroma, lightness
    endColors = [P(10, 99, 70), P(80, 30, 10)]
    indices = colorInterpolation(cc, start=startColors,
                                 end=endColors,
                                 steps=8, asPath=True)
    form.colorMap(indices, P([0, 0], form.size), style="circle")

draw()`.trimStart()}
        reverse={true}
      >
        <ImagePreview src={asset_08} alt="color interpoation" />
      </CodeExample>

      {/* THIRTEEN */}
      <p>Or a 2D slice from the color cube:</p>
      <CodeExample
        code={`@frame(size=(300, 300))
def draw(form, lightness):
    cc = colorCube(steps=100)

    indices = colorInterpolation(
      cc, P(20, 28, 99), P(50, 99, 10), steps=20, asPath=False
    )

    first = np.squeeze(indices, axis=0)
    form.image(first[:, :, lightness].T, P([0, 0], form.size))

draw(lightness=5)`.trimStart()}
        reverse={true}
      >
        <ImagePreview src={asset_09} alt="color field" />
      </CodeExample>

      {/* FOURTEEN */}
      <p>
        <code>gradient()</code> is a convenient function to create gradients
        directly. It supports color stops and 4 different styles: linear,
        radial, sweep, and conical.
      </p>
      <CodeExample
        code={`@frame(size=(300, 300))
def draw(form):
    colors = [0xFF2099FF, 0xFFFF66FF, 0x20CCE0FF] # ARGB
    shader = gradient(P([[100,100]]), colors,
                      "sweep", stops=[0.2, 0.5, 0.8])
    form.fill(shader).rect(P([[0, 0], form.size]))

draw()`.trimStart()}
        reverse={true}
      >
        <ImagePreview src={asset_10} alt="color gradient" />
      </CodeExample>

      {/* ADVANCED TOPICS */}
      <h2 className="mt-12 mb-4">Advanced Topics</h2>
      <p>
        Pts.py is built with{" "}
        <a href="https://numpy.org/" target="_blank" rel="noopener noreferrer">
          numpy
        </a>{" "}
        for computations and{" "}
        <a href="https://skia.org/" target="_blank" rel="noopener noreferrer">
          skia-python
        </a>{" "}
        for rendering. This means you can freely mix numpy and skia functions if
        you need something extra. For example, with numpy vectorization, you
        don't need to write for-loops for many operations.
      </p>
      <h3 className="mt-12 mb-4">numpy / cupy swap</h3>
      <p>
        Switch from numpy to cupy if you need to access GPU for faster
        computation, with just a single line of code. This is an experimental
        feature and only tested with Nvidia CUDA 12 on Linux. Please{" "}
        <a
          href="https://github.com/williamngan/ptspy/issues"
          target="_blank"
          rel="noopener noreferrer"
        >
          file an issue
        </a>{" "}
        if you run into bugs.
      </p>
      <CodeExample
        code={`from ptspy import np_wrapper as np
np.reset("cupy") # switch from numpy (default) to cupy
pts = P(10, 20, 30) # this is now a cupy array on GPU
`}
        reverse={true}
      >
        <></>
      </CodeExample>
      <h3 className="mt-12 mb-4">Interactive window</h3>
      <p>
        Skia supports OpenGL rendering, so you can render in an interactive
        window instead of running Jupyter notebooks. This is also an
        experimental feature, so expect some bugs and rough edges.
      </p>
      <CodeExample
        code={`# See examples folder in the repo for more
...
canvas = surface.getCanvas()
form = Form(canvas)
while not glfw.window_should_close(window):
  draw(form)
  ...`}
        reverse={true}
      >
        <></>
      </CodeExample>
      <h3 className="mt-12 mb-4">More in development</h3>
      <p>
        The <a href="/demos">Demos section</a> has a collection of things you
        can do with Pts.py. Head over there to see some advanced examples: numpy
        vectorization, splines, sampling and more.
      </p>
      <p>
        Last but not least, there is a basic AGENTS.md file in the repo to help
        you set up an AI agent for assistance, or you can simply ask your AI to
        review{" "}
        <a href="https://ptspy.org" target="_blank" rel="noopener noreferrer">
          ptspy.org
        </a>
        . You decide how you you want to explore Pts.py. Have fun!
      </p>
      <div className="mt-12 mb-28 flex items-center justify-center ">
        <CTAButton src="/demos">See some demos</CTAButton>
      </div>
    </div>
  );
}
