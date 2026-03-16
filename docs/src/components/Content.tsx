import { DOCS_TYPES, getType } from "../data";
import { Argument, Func, MarkdownElement, Class } from "../types";

type FunctionContentProps = {
  data: Func | undefined;
  name: string;
  lib?: string;
  namePrefix?: string;
};

type ClassContetntProps = {
  data: Class | undefined;
  name: string;
  lib?: string;
};

export function getDocURL(path: string) {
  const base = import.meta.env.BASE_URL;
  return `${base}docs/${path}`;
}

export function getTypeURL(name: string) {
  const first = name.indexOf(",") > 0 ? name.split(",")[0].trim() : name;
  return DOCS_TYPES.indexOf(first) >= 0 ? (
    <a
      key={name}
      href={getDocURL(`typing/${first}`)}
      className="no-underline hover:underline text-blue-600"
    >
      {name}
    </a>
  ) : (
    name
  );
}

function expandMarkdown(
  contents: (MarkdownElement | string)[] | undefined,
  asParagraph = false
) {
  function _render(s: MarkdownElement) {
    if ("link" in s) {
      return (
        <a className="mx-2" href={s.link[0]}>
          <code className="bg-blue-200">{s.link[1]}</code>
        </a>
      );
    } else if ("code" in s) {
      return <code>{s.code}</code>;
    } else if ("ref" in s) {
      let path = `${s.ref[1][0]}`;
      if (s.ref[1].length > 2) {
        path += `classes/${s.ref[1][1]}/members/${s.ref[1][2]}`;
      } else if (s.ref[1].length > 1) {
        path += `/${s.ref[1][1]}`;
      }
      return (
        <a className="mx-2" href={getDocURL(path)}>
          <code className="bg-blue-200">{s.ref[0]}</code>
        </a>
      );
    } else {
      return s.toString();
    }
  }

  function _renderText(s: string) {
    return s === `\n` ? "" : s;
  }
  if (!contents) return null;
  return (
    <div className="ptspy-mdcontent text-md lg:text-lg text-gray-600">
      {contents.map((s, i) => {
        const content = typeof s === "string" ? _renderText(s) : _render(s);
        return asParagraph ? (
          <p key={`c${i}`}>{content}</p>
        ) : (
          <span key={`c${i}`} className={!content ? "block py-2" : ""}>
            {content}
          </span>
        );
      })}
    </div>
  );
}

function expandTypes(types: string[] | undefined) {
  if (!types) return null;
  return types.map((t, i) =>
    t.toLowerCase() !== "optional" ? (
      getTypeURL(t)
    ) : (
      <span
        className="font-medium inline-block opacity-60 ml-2"
        key={`type${i}`}
      >
        {t}
      </span>
    )
  );
}

function expandSignature(
  name: string,
  args: Argument[] | string | undefined,
  isStatic?: boolean | undefined
) {
  return (
    <h3 className="mt-0 mb-4">
      {isStatic && (
        <span className="text-xs text-gray-500 inline-block pr-1">STATIC </span>
      )}
      <span className="font-semibold">{name} </span>
      <span className="font-light text-gray-600">
        {args &&
          (typeof args === "string" ? (
            <>{args}</>
          ) : (
            <>
              (
              {(args as Argument[]).reduce(
                (a, b) => `${a ? a + "," : a} ${b.name}`,
                ""
              )}{" "}
              )
            </>
          ))}
      </span>
    </h3>
  );
}

function expandExample(contents: string | undefined) {
  if (!contents) return null;
  return <pre>{contents.replace(/\t/g, "  ")}</pre>;
}

export function FunctionContent({
  lib,
  name,
  data,
  namePrefix = "",
}: FunctionContentProps) {
  return (
    <div id={`${namePrefix}${name}`} className="p-8 rounded-xl bg-white">
      {lib && (
        <div className="text-gray-600 flex flex-row gap-2 mb-2">
          {lib !== "pts" && lib} {/* Arbitrary data */}
          {data?.docstring?.Data &&
            Object.keys(data?.docstring?.Data).map((k, i) => (
              <span key={`data${i}`}>
                [ {k} {data?.docstring?.Data![k]}]
              </span>
            ))}
        </div>
      )}

      {/* Signature */}
      {expandSignature(name, data?.docstring?.Args, data?.static)}

      {/* Summary */}
      {expandMarkdown(data?.docstring?.Summary, false)}

      {/* Args */}
      {data?.docstring?.Args && (
        <h4 className="text-xl font-medium mt-6">Parameters</h4>
      )}
      {data?.docstring?.Args?.map((s, i) => (
        <div className="flex flex-col items-start gap-2 my-2" key={`s${i}`}>
          <div>
            <code className="flex flex-row gap-1 font-sans rounded-md">
              <span className="font-semibold">{s.name}</span>
              {" : "}
              {expandTypes(s.type)}
            </code>
          </div>
          <div className="flex-1 mb-4">{expandMarkdown(s.info)}</div>
        </div>
      ))}

      {/* Returns */}
      {data?.docstring?.Returns && (
        <>
          <h4 className="text-xl font-medium">Returns</h4>
          <div className="flex flex-col items-start gap-2 my-2">
            <div>
              <code className="flex flex-row gap-2 font-sans rounded-md">
                {expandTypes(data?.docstring?.Returns?.type)}
              </code>
            </div>
            <div className="flex-1">
              {expandMarkdown(data?.docstring?.Returns?.info)}
            </div>
          </div>
        </>
      )}

      {/* Raise */}
      {data?.docstring?.Raises && (
        <h4 className="text-xl font-medium mt-6">Raises</h4>
      )}
      {data?.docstring?.Raises?.map((s, i) => (
        <div key={`s${i}`} className="flex flex-col items-start gap-2 my-2">
          <div>
            <code className="inline-block font-sans rounded-md">
              {s.exception}
            </code>
          </div>
          <div className="flex-1">{expandMarkdown(s.info)}</div>
        </div>
      ))}

      {/* Example */}
      {data?.docstring?.Examples && (
        <>
          <h4 className="text-xl font-medium mt-6">Examples</h4>
          <div className="my-2 text-gray-600 text-sm">
            {expandExample(data?.docstring?.Examples)}
          </div>
        </>
      )}
    </div>
  );
}

export function ClassContent({ lib, name, data }: ClassContetntProps) {
  return (
    <div id={name} className="p-8 rounded-xl bg-white">
      {lib && <div className="text-gray-600">{lib !== "pts" && lib}</div>}
      {/* Signature */}
      {expandSignature(name, "class")}
      {/* Summary */}
      {expandMarkdown(data?.docstring?.Summary, true)}

      {/* Arbitrary data */}
      {data?.docstring?.Data &&
        Object.keys(data?.docstring?.Data).map((k, i) => (
          <div key={`data${i}`}>
            <code>{k}</code>
            <span className="ml-2">{data?.docstring?.Data![k]}</span>
          </div>
        ))}

      <div>{expandExample(data?.docstring?.Examples)}</div>
    </div>
  );
}

export function TypeContent() {
  return (
    <div className="prose">
      {DOCS_TYPES.map((t) => {
        const data = getType(t);
        return (
          <div key={t} id={data?.name} className="pt-16">
            <h3 className="mb-2">{data?.name}</h3>
            <p className="mb-2">
              <code className="bg-blue-100 rounded-md">{data?.value}</code>
            </p>
            <p>{data?.summary}</p>
          </div>
        );
      })}
    </div>
  );
}
