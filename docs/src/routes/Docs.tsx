import { useParams } from "react-router-dom";
import { MainMenu, SubpackageMenu } from "../components/Menu";
import { Book, ChevronLeft, Search, Close } from "@carbon/icons-react";
import {
  getAllMembers,
  getClass,
  getClassMethod,
  getFunction,
  search,
} from "../data";
import type { SearchResult } from "../data";
import {
  ClassContent,
  FunctionContent,
  TypeContent,
} from "../components/Content";
import { useEffect, useRef, useState } from "react";
import { useScrollSpy } from "../hooks/useScrollSpy";

type Params = {
  packageId: string;
  memberType: string;
  memberId?: string;
  subType?: string;
  subId?: string;
  typeId?: string;
};

export default function Docs() {
  const { packageId, memberId } = useParams<Params>();

  const functions =
    packageId && packageId !== "typing"
      ? getAllMembers(packageId, "functions")
      : [];
  const classes =
    packageId && packageId !== "typing"
      ? getAllMembers(packageId, "classes")
      : [];

  const [anchorID, setAnchorID] = useState<string | null>(null);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const subMenuRef = useRef<HTMLDivElement>(null);
  const [searchQuery, setSearchQuery] = useState<SearchResult[] | undefined>();
  const [searchTerm, setSearchTerm] = useState<string>("");

  // Use scroll spy to update URL based on scroll position
  // Pass initialMemberId so scroll spy knows to wait for initial scroll
  const { setUserNavigating, isUserNavigating } = useScrollSpy(
    packageId,
    memberId
  );

  function handleInputChange(event: React.ChangeEvent<HTMLInputElement>) {
    setSearchTerm(event.target.value);
    const data = search(event.target.value);
    setSearchQuery(data);
  }

  function handleClearSearch() {
    setSearchTerm("");
    setSearchQuery(undefined);
  }

  // Handle menu clicks - mark as user navigation so we scroll to the target
  function handleMenuClick() {
    setUserNavigating();
  }

  // Track if this is the initial mount with a memberId
  const [isInitialLoad, setIsInitialLoad] = useState(!!memberId);

  useEffect(() => {
    if (memberId) {
      setAnchorID(memberId);
    }
  }, [memberId]);

  // Scroll active menu item into view when it changes
  useEffect(() => {
    if (!memberId || !subMenuRef.current) return;
    const activeElement = subMenuRef.current.querySelector(".menu-active");
    if (activeElement) {
      activeElement.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [memberId]);

  useEffect(() => {
    // Scroll to element on:
    // 1. Initial page load with memberId in URL
    // 2. User explicitly clicked a menu item
    if (!isInitialLoad && !isUserNavigating()) return;

    const element = document.getElementById(anchorID || "");
    if (element) {
      // Small delay to ensure DOM is ready
      setTimeout(() => {
        element.scrollIntoView({
          behavior: "auto",
          block: "start",
          inline: "start",
        });
        // Mark initial load as complete
        if (isInitialLoad) {
          setIsInitialLoad(false);
        }
      }, 50);
    }
  }, [anchorID, isUserNavigating, isInitialLoad]);

  // When searching, show submenu results from all matching packages; otherwise just the current one
  const subMenuPackages = searchQuery && searchQuery.length > 0
    ? [...new Set(searchQuery.map(r => r.package))]
    : packageId ? [packageId] : [];

  return (
    <div className="flex flex-row xl:w-[1200px] w-full relative">
      {/* Mobile Menu Toggle Button */}
      <button
        className="lg:hidden fixed bottom-4 left-4 z-50 btn btn-circle btn-primary shadow-lg"
        onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
        aria-label={isMobileMenuOpen ? "Close documentation menu" : "Open documentation menu"}
        aria-expanded={isMobileMenuOpen}
      >
        {isMobileMenuOpen ? (
          <ChevronLeft className="h-6 w-6" />
        ) : (
          <Book className="h-6 w-6" />
        )}
      </button>

      {/* Mobile Menu Drawer */}
      <div
        className={`lg:hidden fixed inset-0 z-40 transition-opacity duration-300 ${
          isMobileMenuOpen ? "opacity-100" : "opacity-0 pointer-events-none"
        }`}
      >
        {/* Backdrop */}
        <div
          className="absolute inset-0 bg-black bg-opacity-50"
          onClick={() => setIsMobileMenuOpen(false)}
        ></div>

        {/* Drawer */}
        <div
          className={`absolute top-0 left-0 h-full w-full bg-base-100 shadow-xl transform transition-transform duration-300 ease-in-out ${
            isMobileMenuOpen ? "translate-x-0" : "-translate-x-full"
          }`}
        >
          <div className="flex flex-col h-full">
            {/* Header with close button */}
            <div className="flex justify-between items-center p-4 border-b">
              <h3 className="text-md lg:text-lg font-semibold">
                Documentation
              </h3>
              <button
                className="btn btn-sm btn-circle btn-ghost"
                onClick={() => setIsMobileMenuOpen(false)}
                aria-label="Close menu"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-6 w-6"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>

            {/* Search */}
            <div className="p-4">
              <label className="input input-lg flex items-center gap-2 pr-3 shadow-md rounded-md outline-none hover:border hover:border-gray-200 transition-all has-[:focus]:shadow-xl has-[:focus]:outline-none has-[:focus]:border-gray-100">
                <Search className="h-5 w-5 text-gray-400" />
                <input
                  type="text"
                  className="grow"
                  placeholder="Search documentation..."
                  value={searchTerm}
                  onChange={handleInputChange}
                />
                {searchTerm && (
                  <button
                    className="btn btn-sm btn-ghost btn-circle"
                    onClick={handleClearSearch}
                    aria-label="Clear search"
                  >
                    <Close className="h-6 w-6 text-gray-400" aria-hidden="true" />
                  </button>
                )}
              </label>
            </div>

            {/* Menu Content */}
            <div className="flex-1 overflow-y-auto p-4">
              <div className="flex flex-col gap-4">
                <div className="ptspy-main-menu">
                  <MainMenu
                    packageId={packageId}
                    onClick={() => {
                      handleMenuClick();
                      setIsMobileMenuOpen(false);
                    }}
                    query={searchQuery}
                  />
                </div>

                {subMenuPackages.map(pkg => (
                  <div key={pkg} className="border-t pt-4">
                    <SubpackageMenu
                      name={pkg}
                      member={memberId}
                      onClick={() => {
                        handleMenuClick();
                        setIsMobileMenuOpen(false);
                      }}
                      query={searchQuery}
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Desktop Menu */}
      <div className="relative xl:min-w-[500px] h-[90vh] hidden lg:flex">
        <div className="hidden lg:flex fixed w-[480px] top-[70px] bottom-0 flex-col rounded-xl p-4">
          <div className="h-[40px] pl-2 mt-2">
            <label className="input input-lg flex items-center gap-2 pr-3 shadow-md rounded-md outline-none hover:border hover:border-gray-200 transition-all has-[:focus]:shadow-xl has-[:focus]:outline-none has-[:focus]:border-gray-100">
              <Search className="h-5 w-5 text-gray-400" />
              <input
                type="text"
                className={`grow ${
                  searchQuery?.length === 0 && "text-red-600"
                }`}
                placeholder="Search documentation..."
                value={searchTerm}
                onChange={handleInputChange}
              />
              {searchTerm && (
                <button
                  className="btn btn-sm btn-ghost btn-circle"
                  onClick={handleClearSearch}
                  aria-label="Clear search"
                >
                  <Close className="h-6 w-6 text-gray-400" aria-hidden="true" />
                </button>
              )}
            </label>
          </div>
          <div className="mt-10 flex flex-row h-[calc(100%-80px)]">
            <div className="ptspy-main-menu flex-none pr-4 border-r border-r-gray-200">
              <MainMenu
                packageId={packageId}
                query={searchQuery}
                onClick={handleMenuClick}
              />
            </div>

            <div ref={subMenuRef} className="ptspy-sub-menu flex-1 overflow-y-auto h-full">
              {/* Sub menu */}
              {subMenuPackages.map(pkg => (
                <div key={pkg} className="w-full">
                  <SubpackageMenu
                    name={pkg}
                    member={memberId}
                    query={searchQuery}
                    onClick={handleMenuClick}
                  />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 flex lg:mx-0 mx-4 my-8 overflow-x-hidden">
        {/* Contents */}
        {packageId && packageId === "typing" && <TypeContent />}

        {packageId && packageId !== "typing" && (
          <div
            className="max-w-full pb-8 bg-transparent prose flex flex-col gap-8"
            key={packageId}
          >
            {(() => {
              // Group functions by category
              const functionsByCategory: { [category: string]: string[] } = {};
              functions.forEach((f) => {
                const fn = getFunction(packageId, f);
                if (fn?.category) {
                  if (!functionsByCategory[fn.category]) {
                    functionsByCategory[fn.category] = [];
                  }
                  functionsByCategory[fn.category].push(f);
                }
              });

              // Group classes by category
              const classesByCategory: { [category: string]: string[] } = {};
              classes.forEach((c) => {
                const cls = getClass(packageId, c);
                if (cls?.category) {
                  if (!classesByCategory[cls.category]) {
                    classesByCategory[cls.category] = [];
                  }
                  classesByCategory[cls.category].push(c);
                }
              });

              // Get all categories sorted alphabetically
              const allCategories = Array.from(
                new Set([
                  ...Object.keys(functionsByCategory),
                  ...Object.keys(classesByCategory),
                ])
              ).sort();

              return allCategories.map((category) => {
                const categoryFunctions = functionsByCategory[category] || [];
                const categoryClasses = classesByCategory[category] || [];

                return (
                  <div key={category} className="flex flex-col gap-8">
                    {categoryFunctions.map((name) => (
                      <FunctionContent
                        key={`${packageId}-function-${name}`}
                        lib={packageId}
                        name={name}
                        data={getFunction(packageId, name)}
                      />
                    ))}

                    {categoryClasses.map((classId) => {
                      const cls = getClass(packageId, classId);
                      const methods = Object.keys(cls?.methods || {});
                      return (
                        <div key={`${packageId}-class-${classId}`} className="flex flex-col gap-8">
                          <ClassContent
                            lib={packageId}
                            name={classId}
                            data={getClass(packageId, classId)}
                          />
                          <div className="flex flex-col gap-8">
                            {methods.map((subId) => (
                              <FunctionContent
                                key={`${packageId}-class-${classId}-method-${subId}`}
                                lib={`${packageId}.${classId}`}
                                name={subId}
                                data={getClassMethod(packageId, classId, subId)}
                                namePrefix={`${classId}.`}
                              />
                            ))}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                );
              });
            })()}
          </div>
        )}
      </div>
    </div>
  );
}
