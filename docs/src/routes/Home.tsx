import asset_01 from "../assets/home/home_01.mp4";
import asset_02 from "../assets/home/home_02.png";
import { Link, useNavigate } from "react-router-dom";

export default function Home() {
  return (
    <>
      <div className="prose-sm lg:prose mt-8 max-w-[48rem] leading-snug mx-6">
        <h1 className="font-semibold text-center xl:mb-12 mb-8">
          Pts.py is an experimental library for visual thinking
        </h1>
        <div className="mb-20 flex items-center justify-center ">
          <CTAButton src="/start">Start</CTAButton>
        </div>
        <h3 className="mt-12">
          A vision appears. It twinkles in the vast space of imagination. We
          squint to connect the dots, looking for hidden relationships in a sky
          full of stars.
        </h3>
      </div>
      <div className="w-full max-w-[56rem] mx-auto mt-12 aspect-square bg-gray-200 overflow-hidden lg:rounded-xl">
        <video
          src={asset_01}
          title="cloud-like waves move downward to reveal twinkling dots on a dark background"
          autoPlay
          muted
          loop
          playsInline
        />
      </div>
      <div className="prose-sm lg:prose max-w-[48rem] leading-snug mx-6">
        <h3 className="mt-12">
          Can we translate abstract thoughts into visual primitives, and arrange
          and rearrange them through code? Can form follow <em>function()</em>?
        </h3>
        <h3 className="mt-12">
          Here we ponder the multiplicity of forms with a new mindset.
        </h3>
        <h3 className="mt-12">
          I started this series of explorations, first with an idea about a
          simple point (
          <a href="https://williamngan.github.io/pt/" target="_blank">
            Pt
          </a>
          ,{" "}
          <a
            href="https://medium.com/@williamngan/pt-93382bf5943e"
            target="_blank"
          >
            essay
          </a>
          , 2015), then an idea about connecting them, (
          <a href="https://ptsjs.org" target="_blank">
            Pts.js
          </a>
          , 2017). Starting anew in Pts.py (2026), this is again both a software
          library and a hypothesis on visual composability. Software libraries
          are my favourite conceptual substrates.
        </h3>
        <h3 className="mt-12 mb-12">
          Pts.py is AI-ready, but at its core I hope it remains human-centric.
          We take time to sip tea, play puzzles, plant flowers, create for the
          sheer joy of creating. While the world hurls itself into an
          exponential unknown, our slow journey starts from a single point, in 4
          lines of code.
        </h3>
        <div className="flex xl:flex-row flex-col mt-6 w-full gap-8">
          <div className="object-contain w-full aspect-square bg-black overflow-hidden rounded-lg">
            <img
              src={asset_02}
              alt="A single black dot in the center of a white square"
            />
          </div>
          <div className="flex-1">
            <pre>{`@frame()
def draw(form):
    form.fillOnly("000")
        .point(form.size/2, 10)

draw()`}</pre>
          </div>
        </div>
        <div className="my-24 flex items-center justify-center ">
          <CTAButton src="/start" large={true}>
            Start
          </CTAButton>
        </div>
        <div className="my-24 text-sm flex items-center justify-center ">
          <p>
            Pts.py is an{" "}
            <a
              href="https://github.com/williamngan/ptspy"
              target="_blank"
              rel="noopener noreferrer"
            >
              open source project
            </a>
            . Copyright &copy; 2026-present{" "}
            <a
              href="https://williamngan.com"
              target="_blank"
              rel="noopener noreferrer"
            >
              William Ngan
            </a>
            . All rights reserved.
          </p>
        </div>
      </div>
    </>
  );
}

export function CTAButton({
  children,
  large = false,
  src,
}: {
  children: React.ReactNode;
  src: string;
  large?: boolean;
}) {
  const navigate = useNavigate();

  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault();
    window.scrollTo(0, 0);
    navigate(src);
  };

  return (
    <Link to={src} className="!no-underline" onClick={handleClick}>
      <button
        className={`${
          large ? "btn-lg" : "btn-md"
        } btn btn-primary border-none ${large ? "h-[6rem]" : "h-[4rem]"} ${
          large ? "px-16" : "px-8"
        } ${
          large ? "text-[1.75rem]" : "text-[1.2rem]"
        } rounded-full relative overflow-hidden group`}
      >
        <span className="relative z-10">{children}</span>
        <span className="absolute inset-0 bg-gradient-to-br from-pink-400 via-violet-700 to-orange-400 opacity-0 group-hover:opacity-100 transition-opacity animate-[gradient_5s_ease_infinite] bg-[length:200%_200%]"></span>
      </button>
    </Link>
  );
}
