// types.ts

// Type for the Markdown elements
export type MarkdownElement =
  | { text: string }
  | { code: string }
  | { image: [string, string] }
  | { link: [string, string] }
  | { ref: [string, string[]] };

// Type for function/method arguments
export type Argument = {
  name: string;
  type: string[];
  info: MarkdownElement[];
};

// Type for the returns section
export type Return = {
  type: string[];
  info: MarkdownElement[];
};

// Type for the raises section
export type Raise = {
  exception: string;
  info: MarkdownElement[];
};

// Type for the docstring sections
export type Docstring = {
  Summary: MarkdownElement[];
  Args?: Argument[];
  Returns?: Return;
  Raises?: Raise[];
  Examples?: string;
  Data?: Record<string, string | boolean | number>;
};

// Type for class methods
export type Method = {
  docstring: Docstring;
  source: [string, number];
  category: string;
  static?: boolean;
};

// Type for classes
export type Class = {
  docstring: Docstring;
  source: [string, number];
  category: string;
  methods: { [methodName: string]: Method };
};

// Type for functions
export type Func = {
  docstring: Docstring;
  source: [string, number];
  category: string;
  static?: boolean;
};

export type Typing = {
  name: string;
  value: string;
  summary: string;
};

export type MemberTypes = "functions" | "classes" | "module";

export type FunctionsDoc = { [functionName: string]: Func };
export type ClassesDoc = { [className: string]: Class };

// Type for the entire documentation structure
export type Documentation = {
  typing: Typing[];
  packages: {
    [packageName: string]: {
      module: { docstring?: Docstring; source?: [string, number] };
      classes: ClassesDoc | {};
      functions: FunctionsDoc | {};
    };
  };
};
