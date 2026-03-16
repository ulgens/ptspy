import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import Root from "./routes/Root";
import ErrorPage from "./routes/ErrorPage";
import Docs from "./routes/Docs";
import {
  createBrowserRouter,
  RouterProvider,
  Navigate,
} from "react-router-dom";
import Start from "./routes/Start";
import Demo from "./routes/Demo";
import Home from "./routes/Home";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Root />,
    errorElement: <ErrorPage />,
    children: [
      {
        path: "start",
        element: <Start />,
      },
      {
        path: "demos",
        element: <Demo />,
      },
      {
        path: "docs",
        element: <Navigate to="/docs/core" replace />,
      },
      {
        path: "docs/:packageId",
        element: <Docs />,
      },
      {
        path: "docs/:packageId/:memberId",
        element: <Docs />,
      },
      { path: "/", element: <Home /> },
      // {
      //   path: "docs/:packageId/:memberType/:memberId/:subType/:subId",
      //   element: <Docs />,
      // },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
