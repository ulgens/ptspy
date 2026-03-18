import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import python from "react-syntax-highlighter/dist/esm/languages/hljs/python";
import hljs from "../components/syntaxStyle";
import { useRef, useState } from "react";
import s0 from "../assets/demo/s0.png";
import s1 from "../assets/demo/s1.png";
import s2 from "../assets/demo/s2.png";
import s3 from "../assets/demo/s3.png";
import s4 from "../assets/demo/s4.png";
import s5 from "../assets/demo/s5.png";
import s6 from "../assets/demo/s6.png";
import s7 from "../assets/demo/s7.png";
import s8 from "../assets/demo/s8.png";

import { ArrowUpRight, Close } from "@carbon/icons-react";

const asset0 = [
  s0,
  "@frame",
  `@frame(bg="fff", size=(600, 600))
def draw(form):
    form.fillOnly("fff").rect(bounds)
    form.strokeOnly("000", 2).circle(bounds)

    # make a stamp
    form.erase(perforate(form.size))
    stampId(form, 0, color="000")

draw()
# ptspy v0.1.0`,
  "Use `@frame` decorator to create an image, or `@frames` to create a video. Include a `form` parameter in the decorated function for drawing.",
];

const asset1 = [
  s1,
  "Shaping functions",
  `@frame(bg="fff", size=(600, 600))
def draw(form):
    one = steps([100, padding*2.5, 20], [100, boundSize, 2], count=20, shaping="linear")
    two = steps([200, padding*2, 2], [200, boundSize-padding*2, 60], count=10, shaping="quadraticOut")
    three = steps([300, padding*2.5, 20], [300, boundSize, 2], count=20, shaping="cubicInOut")
    # custom shaping function
    four = steps([400, padding*2, 2], [400, boundSize-padding*2, 60], count=10,
                 shaping=lambda t, a, b: a + (b - a) * (t * t * t))

    form.fillOnly("000").rect( bounds )
    form.fillOnly("fff").points(one).points(two).points(three).points(four)

    # make a stamp
    form.erase(perforate(form.size))
    stampId(form, 1, color="fff")

draw()
# ptspy v0.1.0`,
  "Divide a range linearly or non-linearly. Use shaping functions to distribute time, space, and more.",
];

const asset2 = [
  s2,
  "Vectorized operations",
  `@frame(bg="fff", size=(600, 600))
def draw(form):
    top = steps(25, 575, 30, shaping="cubicIn")
    # reshape with new axis and append 0. shape is now (n,2)
    top = expand(top[:, np.newaxis], 0)
    lines = np.stack([top+[0, bounds[0][1]], top+[padding*2, bounds[1][1]]], axis=1)

    # or just step through 2D directly
    another = np.stack(
        [steps([25,100], [575, 500], 30),
         steps([500, 25], [100, 575], 30)], axis=1)

    form.fillOnly("000").rect( bounds )
    form.strokeOnly("fff", 3).lines(np.concatenate([lines, another]))

    # make a stamp
    form.erase(perforate(form.size))
    stampId(form, 2, color="fff", adjust=5)

draw()
# ptspy v0.1.0`,
  "No loops needed. Transform points by arranging and rearranging arrays and dimensions.",
];

const asset3 = [
  s3,
  "Color cubes",
  `@frame(bg="ffffff", size=(600, 600))
def draw(form):
    # create a color cube of 100 steps
    cc = colorCube(steps=100)
    start = P(90, 99, 0)
    end = P(60, 19, 30)
    # get a sub-cube
    indices = colorInterpolation(cc, start, end, steps=30, asPath=False)
    first = np.squeeze(indices, axis=0)
    form.image(first[25, :, :], bounds) # draw a slice

    # get a sub-path
    indices = colorInterpolation(cc, start, end, steps=12, asPath=True)
    form.colorMap(indices, [bounds[0]+[50, 1], bounds[1]-[125,1]])

    # make a stamp
    form.erase(perforate(form.size))
    stampId(form, 3, adjust=5)

draw()
# ptspy v0.1.0`,
  "A simple cube of (hue, chroma, lightness) is the easiest for human to reason. Precalculate the cube at specific granularity, and then slice, dice, and interpolate.",
];

const asset4 = [
  s4,
  "Partial functions",
  `@frame(bg="ffffff", size=(600, 600))
def draw(form):
    # partial function
    poly = regularPolygon( center=TBD, radius=form.width/6, sides=TBD, closed=True )

    centers = P([0.33, 0.33], [0.33, 0.66], [0.66, 0.33], [0.66, 0.66]) * form.size - [15, -15]
    shapes = [poly(center=c, sides=s) for c, s in zip(centers, [5,7,9,11])]

    form.fillOnly("000", 3).polygons( shapes )

    # make a stamp
    form.erase(perforate(form.size))
    stampId(form, 4, color="000", adjust=-10)

draw()
# ptspy v0.1.0`,
  "Use partial functions to create variations of a base form and explore new compositions. Turn your own functions into partials with `@partial`.",
];

const asset5 = [
  s5,
  "Grids",
  `@frame(bg="fff", size=(600, 600))
def draw(form):
    points = gridPoints([10,11], [53, 32]) + [50, 200]
    regions = pointsToRegions(points[::4], [30, 30], [0, 0.5])
    form.fillOnly("000").rect( bounds )

    form.strokeOnly("06f", 3).rects(regions)
    form.fillOnly("fff").points(points[1::2], 3).points(points[1::3], 10)
    form.strokeOnly("fff", 2).points(points[::3], 7, style="rect")

    # make a stamp
    form.erase(perforate(form.size))
    stampId(form, 5, color="fff", adjust=5)

draw()
# ptspy v0.1.0`,
  "A grid is a substrate for structures and rhythms, depending on how you count and how you weave.",
];

const asset6 = [
  s6,
  "Curves",
  `@frame(bg="fff", size=(600, 600))
def draw(form):
    original = regularPolygon( center=form.size/2, radius=form.width/3, sides=24, closed=True )
    poly = rotate2D( original, PI, origin=TBD )
    spiralA = P([scale2D(p, i*1/12, form.size/2) for i, p in enumerate(poly(origin=form.size/2))])
    spiralB = P([scale2D(p, i*1/8, form.size/2) for i, p in enumerate(poly(origin=form.size/3))])

    form.strokeOnly("E02", 80).cardinal(spiralB, tension=0.5)
    form.strokeOnly("000", 80).bspline(spiralA)
    form.fillOnly("FFF").points(sampleBspline(spiralA, numSamples=2), 3)

    # make a stamp
    form.erase(perforate(form.size))
    stampId(form, 6, color="000", adjust=5)

draw()
# ptspy v0.1.0`,
  "Transform points gradually into control points of splines, and find points that travel along the curves.",
];

const asset7 = [
  s7,
  "Sampling points",
  `@frame(bg="ffffff", size=(600, 600))
def draw(form):
    bands = Rectangle.strips(bounds, 0.25, axis=1)
    form.fillOnly("000").points(poissonDisc(bands[0], 30), 3)
    form.points(poissonDisc(bands[1], 10), 3)
    form.points(poissonDisc(bands[2], 6), 3)
    form.points(poissonDisc(bands[3], 4), 3)

    # make a stamp
    form.erase(perforate(form.size))
    stampId(form, 7, color="000", adjust=-10)

draw()
# ptspy v0.1.0`,
  "Making marks unevenly yet consistently across a surface. At different densities, we may perceive different tactile qualities.",
];

const asset8 = [
  s8,
  "Noise functions",
  `@frame(bg="fff", size=(600, 600))
def draw(form):

    # create a color cube of 50 steps
    cc = colorCube(steps=50)
    colors = colorInterpolation(cc,  P(0, 49, 40), P(30, 15, 25), steps=10)

    perlin = createPerlinNoise2D([101, 1], seed=123)
    corners = P([boundSize, boundSize], [0, boundSize]) + padding

    form.fillOnly("000").rect( bounds )

    # show different frequencies
    for i in range(1, 10):
        values = perlin(freq=1-i/10, octaves=2).reshape(-1)
        wave = attachPositionIndex(values) * P(boundSize/100, 200) + P(padding, 250+i*15)
        polygon = rotate2D( np.append(wave, corners, axis=0), PI/2, boundRect.center)
        form.fillOnly(colors[i]).polygon(polygon)

    # make a stamp
    form.erase(perforate(form.size))
    stampId(form, 8, color="fff", adjust=-2)

draw()
# ptspy v0.1.0`,
  "Continuous values from a noise function express changes, shifting from soothing to violent.",
];

SyntaxHighlighter.registerLanguage("python", python);

export function ImagePreview({
  src,
  isVideo,
  alt,
  hasHover = false,
}: {
  src: string;
  isVideo?: boolean;
  alt: string;
  hasHover?: boolean;
}) {
  if (isVideo) {
    return (
      <div className="object-contain w-full aspect-square bg-gray-200 overflow-hidden rounded-lg">
        <video src={src} title={alt} autoPlay muted loop playsInline />
      </div>
    );
  }
  return (
    <div className="overflow-visible object-contain w-full aspect-square rounded-lg group relative">
      {hasHover && (
        <div className="absolute top-0 bottom-0 left-0 right-0 blur-3xl transition-opacity duration-500 bg-black opacity-0 group-hover:opacity-30 z-0" />
      )}
      <img
        src={src}
        alt={alt}
        className={
          hasHover
            ? `hover:scale-100 hover:rotate-2 transition-transform duration-300 relative z-10`
            : undefined
        }
      />
    </div>
  );
}

export function Stamp({
  asset,
  onClick,
  hasHover = true,
}: {
  asset: string[];
  onClick?: () => void;
  hasHover?: boolean;
}) {
  return (
    <div
      className="m-[1px] relative inline-block cursor-pointer group"
      onClick={onClick}
    >
      <div className="flex flex-col gap-3 py-8 xl:px-8 px-4">
        <h3 className="text-lg font-medium text-gray-600 group-hover:text-gray-900">
          {asset[1]}
        </h3>
        <ImagePreview src={asset[0]} alt={asset[1]} hasHover={hasHover} />
      </div>
    </div>
  );
}

export default function Stamps() {
  const modalRef = useRef<HTMLDialogElement>(null);
  const [activeAsset, setActiveAsset] = useState<string[] | null>(null);

  const openModal = (asset: string[]) => {
    setActiveAsset(asset);
    modalRef.current?.showModal();
  };

  return (
    <>
      <div className="prose mx-4 xl:mx-2 mt-8 max-w-prose">
        <h1 className="lg:text-center lg:text-left lg:ml-[-1rem] tracking-tighter mb-10 lg:mb-16 mt-[3rem] lg:mt-[6rem] font-medium text-[6.5rem] leading-[4.8rem] lg:text-[14rem] lg:leading-[13.5rem] ">
          Demos
        </h1>
        <h3 className="lg:text-center leading-[1.35] mx-2 lg:mx-0">
          Here we set out on a long journey full of adventures. Along the way, I
          made some discoveries and some stamps for the postcards to send home.
        </h3>
      </div>

      <div className="lg:my-24 my-16 leading-[0px] grid grid-cols-1 lg:grid-cols-3 max-w-[1220px] mx-auto px-2">
        <Stamp asset={asset0} onClick={() => openModal(asset0)} />
        <Stamp asset={asset1} onClick={() => openModal(asset1)} />
        <Stamp asset={asset2} onClick={() => openModal(asset2)} />
        <Stamp asset={asset3} onClick={() => openModal(asset3)} />
        <Stamp asset={asset4} onClick={() => openModal(asset4)} />
        <Stamp asset={asset5} onClick={() => openModal(asset5)} />
        <Stamp asset={asset6} onClick={() => openModal(asset6)} />
        <Stamp asset={asset7} onClick={() => openModal(asset7)} />
        <Stamp asset={asset8} onClick={() => openModal(asset8)} />
      </div>

      <dialog ref={modalRef} className="modal">
        <div className="modal-box max-w-full lg:max-w-[75rem] bg-gray-200 rounded-3xl lg:p-16 p-8 relative">
          <form method="dialog">
            <button className="btn btn-circle border-none btn-ghost absolute lg:right-4 right-2 lg:top-4 top-2 border-none">
              <Close size={32} />
            </button>
          </form>
          {activeAsset && (
            <div className="flex flex-col gap-4">
              <div className="flex flex-col lg:flex-row-reverse lg:gap-12 gap-6 lg:mb-4 mb-0">
                <div className="flex flex-0 flex-col gap-4">
                  <div className="flex-0 max-h-[300px] max-w-[300px] mt-6 lg:mt-0">
                    <ImagePreview
                      src={activeAsset[0]}
                      alt={activeAsset[1]}
                      hasHover={false}
                    />
                  </div>
                </div>
                {activeAsset[2] && (
                  <div className="flex flex-col gap-2 flex-1">
                    <h3 className="lg:text-2xl text-xl leading-7">
                      {activeAsset[3] || ""}
                    </h3>
                    <div className="lg:flex lg:flex-col lg:gap-3 block my-2 lg:mb-0 lg:mt-4">
                      <div className="inline-block lg:block pr-4 pt-2 lg:pt-0">
                        <a
                          href="https://github.com/williamngan/ptspy/blob/main/notebooks/Stamps.ipynb"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="whitespace-nowrap text-gray-700  no-underline hover:no-underline btn btn-outline rounded-full btn-sm"
                        >
                          View source in notebook{" "}
                          <ArrowUpRight className="inline-block w-3 h-3" />
                        </a>
                      </div>
                    </div>
                  </div>
                )}
              </div>
              {activeAsset[2] && (
                <SyntaxHighlighter
                  language="python"
                  className="hljs rounded-lg lg:p-4 lg:text-sm text-xs overflow-x-auto"
                  style={hljs}
                  showLineNumbers
                  lineNumberStyle={{ color: "#404449" }}
                >
                  {activeAsset[2]}
                </SyntaxHighlighter>
              )}
            </div>
          )}
        </div>
        <form method="dialog" className="modal-backdrop">
          <button>close</button>
        </form>
      </dialog>
    </>
  );
}
