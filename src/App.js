import React from "react";
import { createHashRouter, RouterProvider, Navigate } from "react-router-dom";
import Layout from "./Layout";
import ErrorBoundary from "./components/ErrorBoundary";

const MainPage        = React.lazy(() => import("./pages/main_page"));
const QueryPage       = React.lazy(() => import("./pages/query_page"));
const TasksPage       = React.lazy(() => import("./pages/tasks_page"));
const DocumentationPage = React.lazy(() => import("./pages/documentation_page"));
const NotFoundPage    = React.lazy(() => import("./pages/notfound_page"));

const router = createHashRouter([
  {
    element: <Layout />,
    children: [
      { path: "/",             element: <Navigate to="/main" /> },
      { path: "/main",         element: <MainPage />,          errorElement: <NotFoundPage /> },
      { path: "/query",        element: <QueryPage />,         errorElement: <NotFoundPage /> },
      { path: "/tasks",        element: <TasksPage />,         errorElement: <NotFoundPage /> },
      { path: "/documentation",element: <DocumentationPage />, errorElement: <NotFoundPage /> },
    ],
  },
]);

function App() {
  return (
    <ErrorBoundary>
      <RouterProvider router={router} />
    </ErrorBoundary>
  );
}

export default App;
