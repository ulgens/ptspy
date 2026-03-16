import { useRouteError } from "react-router-dom";

export default function ErrorPage() {
  const error = useRouteError();
  console.error(error);

  return (
    <div className="prose p-8 bg-gray-50">
      <h1>Oops! An error has occured</h1>
      <p>Did you get the correct url? Want to file an issue?</p>
    </div>
  );
}
