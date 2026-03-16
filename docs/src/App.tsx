import { getAllPackages, getAllMembers } from "./data";

function App() {
  return (
    <div className="prose">
      <div>{getAllPackages().join(", ")}</div>
      <div>{getAllMembers("process", "functions").join(",")}</div>
    </div>
  );
}

export default App;
