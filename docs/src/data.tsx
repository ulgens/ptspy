import data from "./assets/docs.json";
import {
  ClassesDoc,
  Documentation,
  Func,
  Class,
  FunctionsDoc,
  MemberTypes,
} from "./types";

export const DOCS: Documentation = Object.freeze({ ...data });

export const DOCS_TYPES = Object.freeze(getAllTypes());

export type SearchResult = {
  package: string;
  type: MemberTypes;
  name: string;
  classMethod?: string;
};

export function getAllPackages() {
  return Object.keys(DOCS["packages"]);
}

export function search(term: string): SearchResult[] | undefined {
  const query = term.trim().toLowerCase();
  if (query.length <= 1 && query !== "p") {
    return undefined;
  }

  const results: SearchResult[] = [];

  const packages = getAllPackages();

  for (const pkg of packages) {
    const packageData = DOCS["packages"][pkg];

    // Search functions
    const functionNames = Object.keys(packageData.functions);
    for (const fn of functionNames) {
      if (fn.toLowerCase().includes(query)) {
        results.push({ package: pkg, type: "functions", name: fn });
      }
    }

    // Search classes and their methods
    const classNames = Object.keys(packageData.classes);
    for (const cls of classNames) {
      // Check class name
      if (cls.toLowerCase().includes(query)) {
        results.push({ package: pkg, type: "classes", name: cls });
      }

      // Search class methods
      const classData = (packageData.classes as ClassesDoc)[cls];
      if (classData?.methods) {
        const methodNames = Object.keys(classData.methods);
        for (const methodName of methodNames) {
          if (methodName.toLowerCase().includes(query)) {
            results.push({
              package: pkg,
              type: "classes",
              name: cls,
              classMethod: methodName,
            });
          }
        }
      }
    }
  }

  // Search typing documentation
  const typingDocs = DOCS["typing"];
  for (const typeDoc of typingDocs) {
    if (typeDoc.name.toLowerCase().includes(query)) {
      results.push({
        package: "typing",
        type: "functions", // Using "functions" as the type for consistency
        name: typeDoc.name,
      });
    }
  }

  return results;
}

export function getAllMembers(
  packageName: string,
  memberType: MemberTypes = "functions"
) {
  return Object.keys(DOCS["packages"][packageName][memberType]);
}

export function getAllTypes() {
  return DOCS["typing"].map((t) => t.name);
}

export function getFunction(
  packageName: string,
  memberName: string
): Func | undefined {
  const f = (DOCS["packages"][packageName]["functions"] as FunctionsDoc)[
    memberName
  ];
  return f || undefined;
}

export function getClass(
  packageName: string,
  className: string
): Class | undefined {
  return (DOCS["packages"][packageName]["classes"] as ClassesDoc)[className];
}

export function getClassMethod(
  packageName: string,
  className: string,
  methodName: string
): Func | undefined {
  const cls = getClass(packageName, className);
  return cls?.methods[methodName];
}

export function getType(name: string) {
  return DOCS["typing"].find((t) => t.name === name);
}
