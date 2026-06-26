import { createHashRouter, RouterProvider, Navigate } from "react-router-dom";
import Layout from "./Layout";
import QueryPage from "./pages/query_page";
import MainPage from "./pages/main_page";
import DocumentationPage from "./pages/documentation_page";
import TasksPage from "./pages/tasks_page";
import NotFoundPage from "./pages/notfound_page";
import ErrorBoundary from "./components/ErrorBoundary";

const router = createHashRouter([
  {
    element: <Layout />,
    children: [
      {
        path: "/",
        element: <Navigate to="/main" />,
      },
      {
        path: "/main",
        element: <MainPage />,
        errorElement: <NotFoundPage />,
      },
      {
        path: "/query",
        element: <QueryPage />,
        errorElement: <NotFoundPage />,
      },
      {
        path: "/tasks",
        element: <TasksPage />,
        errorElement: <NotFoundPage />,
      },
      {
        path: "/documentation",
        element: <DocumentationPage />,
        errorElement: <NotFoundPage />,
      },
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
