import { Outlet, Link, useNavigate } from "react-router-dom";
import { Menu, ArrowUpRight } from "@carbon/icons-react";

function TopMenu() {
  const navigate = useNavigate();

  const handleLinkClick = (
    e: React.MouseEvent<HTMLAnchorElement>,
    to: string,
  ) => {
    e.preventDefault();
    window.scrollTo(0, 0);
    navigate(to);
  };

  return (
    <>
      <li>
        <Link to="/start" onClick={(e) => handleLinkClick(e, "/start")}>
          Start
        </Link>
      </li>
      <li>
        <Link to="/demos" onClick={(e) => handleLinkClick(e, "/demos")}>
          Demos
        </Link>
      </li>
      <li>
        <Link to="/docs" onClick={(e) => handleLinkClick(e, "/docs")}>
          Docs
        </Link>
      </li>
      <li>
        <a
          href="https://github.com/williamngan/ptspy"
          target="_blank"
          rel="noopener noreferrer"
        >
          Github
          <ArrowUpRight className="inline-block w-3 h-3" />
        </a>
      </li>
    </>
  );
}

function TopNav() {
  const navigate = useNavigate();

  const handleLogoClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
    e.preventDefault();
    window.scrollTo(0, 0);
    navigate("/");
  };

  return (
    <div className="fixed w-full h-[80px] pb-5 navbar font-sans z-50 bg-gradient-to-b from-gray-100 from-70% to-transparent">
      <div className="navbar-start">
        <Link
          to="/"
          className="btn btn-ghost text-xl heading"
          onClick={handleLogoClick}
        >
          <span className="font-bold">
            Pts.py<span className="md:inline hidden">:</span>
          </span>
          <span className="md:inline hidden">visual thinking in code</span>
        </Link>
      </div>

      <div className="navbar-end">
        <div className="dropdown dropdown-end lg:hidden">
          <button
            tabIndex={0}
            className="btn btn-ghost"
            aria-label="Open navigation menu"
          >
            <Menu className="h-5 w-5" />
          </button>
          <ul
            tabIndex={0}
            className="menu menu-lg dropdown-content bg-base-100 font-semibold rounded-lg z-[1] mt-3 w-52 p-2 shadow"
          >
            <TopMenu />
          </ul>
        </div>
        <div className="navbar-center hidden lg:flex">
          <ul className="menu menu-lg menu-horizontal px-1">
            <TopMenu />
          </ul>
        </div>
      </div>
    </div>
  );
}

export default function Root() {
  return (
    <>
      <TopNav />
      <div className="relative flex flex-col pt-[60px] items-center justify-center">
        <Outlet />
      </div>
    </>
  );
}
