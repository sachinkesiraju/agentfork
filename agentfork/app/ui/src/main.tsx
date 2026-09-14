import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider, Navigate } from "react-router-dom";
import "./index.css";
import ProjectsPage from "./routes/ProjectsPage";
import NewProjectPage from "./routes/NewProjectPage";
import ProjectPage from "./routes/ProjectPage";

const router = createBrowserRouter([
  { path: "/", element: <ProjectsPage /> },
  { path: "/new", element: <NewProjectPage /> },
  { path: "/p/:projectId", element: <ProjectPage /> },
  { path: "*", element: <Navigate to="/" replace /> },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
);
