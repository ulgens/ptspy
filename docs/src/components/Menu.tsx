import { Link } from "react-router-dom";
import { getDocURL } from "./Content";
import { DOCS_TYPES, getAllMembers, getClass, getFunction } from "../data";
import type { SearchResult } from "../data";

type MainMenuProps = {
  packageId?: string;
  onClick?: () => void;
  query?: SearchResult[];
};

export function MainMenu({ packageId, onClick, query }: MainMenuProps) {
  const activeClass = "menu-active";

  // Get unique packages from search results
  const searchPackages = query
    ? [...new Set(query.map((result) => result.package))]
    : [];

  // List of all available packages
  const allPackages = ["core", "more", "typing"];

  // If there's a search query, only show packages that have results
  const packagesToShow =
    query && query.length > 0 ? searchPackages : allPackages;

  return (
    <ul className="menu menu-lg" onClick={onClick}>
      {packagesToShow.map((pkg) => (
        <li key={pkg}>
          <Link
            to={getDocURL(pkg)}
            className={packageId === pkg ? activeClass : "font-semibold"}
          >
            {pkg}
          </Link>
        </li>
      ))}
    </ul>
  );
}

type SubpackageMenuProps = {
  name: string;
  member?: string;
  onClick?: () => void;
  query?: SearchResult[];
};

export function SubpackageMenu({
  name,
  member,
  onClick,
  query,
}: SubpackageMenuProps) {
  const activeClass = "menu-active";

  // Filter search results for this specific package
  const packageResults = query
    ? query.filter((result) => result.package === name)
    : [];

  // Special case for types
  if (name === "typing") {
    // If there are search results, only show matching types
    const typesToShow =
      packageResults.length > 0
        ? packageResults.map((result) => result.name)
        : DOCS_TYPES;

    return (
      <ul className="menu lg:menu-lg menu-md">
        {typesToShow.map((t) => {
          return (
            <li key={t}>
              <Link
                to={getDocURL(`typing/${t}`)}
                className={member === t ? activeClass : ""}
              >
                {t}
              </Link>
            </li>
          );
        })}
      </ul>
    );
  }

  const functions = getAllMembers(name, "functions");
  const classes = getAllMembers(name, "classes");

  // Filter functions and classes based on search results
  const functionsToShow =
    packageResults.length > 0
      ? functions.filter((fn) =>
          packageResults.some(
            (result) => result.type === "functions" && result.name === fn
          )
        )
      : functions;

  const classesToShow =
    packageResults.length > 0
      ? classes.filter((cls) =>
          packageResults.some(
            (result) => result.type === "classes" && result.name === cls
          )
        )
      : classes;

  // Group functions by category
  const functionsByCategory: { [category: string]: string[] } = {};
  functionsToShow.forEach((f) => {
    const fn = getFunction(name, f);
    if (fn?.category) {
      if (!functionsByCategory[fn.category]) {
        functionsByCategory[fn.category] = [];
      }
      functionsByCategory[fn.category].push(f);
    }
  });

  // Group classes by category
  const classesByCategory: { [category: string]: string[] } = {};
  classesToShow.forEach((c) => {
    const cls = getClass(name, c);
    if (cls?.category) {
      if (!classesByCategory[cls.category]) {
        classesByCategory[cls.category] = [];
      }
      classesByCategory[cls.category].push(c);
    }
  });

  // Create combined sections by category
  const allCategories = new Set([
    ...Object.keys(functionsByCategory),
    ...Object.keys(classesByCategory),
  ]);

  const categorySections = Array.from(allCategories)
    .sort() // Sort categories alphabetically
    .map((category) => {
      const categoryFunctions = functionsByCategory[category] || [];
      const categoryClasses = classesByCategory[category] || [];

      const functionLinks = categoryFunctions.map((f) => {
        const fn = getFunction(name, f);
        const isInternal: boolean = fn?.docstring?.Data?.internal != null;
        return (
          <li key={f} className={isInternal ? "text-gray-400" : ""}>
            <Link
              to={getDocURL(`${name}/${f}`)}
              className={member === f ? activeClass : ""}
            >
              {f}
            </Link>
          </li>
        );
      });

      const classLinks = categoryClasses.map((f) => {
        const cls = getClass(name, f);
        const methods = Object.keys(cls?.methods || {});
        const isInternal: boolean = cls?.docstring?.Data?.internal != null;

        // Filter methods based on search results
        const methodsToShow =
          packageResults.length > 0
            ? methods.filter((method) =>
                packageResults.some(
                  (result) =>
                    result.type === "classes" &&
                    result.name === f &&
                    result.classMethod === method
                )
              )
            : methods;

        return (
          <li key={f} className={isInternal ? "text-gray-400" : ""}>
            <Link
              to={getDocURL(`${name}/${f}`)}
              className={member === f ? activeClass : ""}
            >
              {f}
            </Link>
            <ul>
              {methodsToShow &&
                methodsToShow.map((m) => {
                  return (
                    <li key={`${name}-${m}`}>
                      <Link
                        to={getDocURL(`${name}/${f}.${m}`)}
                        className={member === `${f}.${m}` ? activeClass : ""}
                      >
                        {m}
                      </Link>
                    </li>
                  );
                })}
            </ul>
          </li>
        );
      });

      return (
        <ul
          key={category}
          className="menu w-56 lg:text-lg text-md"
          onClick={onClick}
        >
          <li className="menu-title text-xl text-gray-400 font-bold flex flex-row items-center">
            {category}
          </li>
          {functionLinks}
          {classLinks}
        </ul>
      );
    });

  return <>{categorySections}</>;
}
