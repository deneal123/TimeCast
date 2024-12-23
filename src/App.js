import { createHashRouter, RouterProvider, Navigate } from "react-router-dom";
import Layout from "./Layout";
import QueryPage from "./pages/query_page";
import MainPage from "./pages/main_page";
import DocumentationPage from "./pages/documentation_page"
import NotFoundPage from "./pages/notfound_page";


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
                path: "/documentation",
                element: <DocumentationPage />,
                errorElement: <NotFoundPage />,
            },
            {
                path: "/query",
                element: <QueryPage />,
                errorElement: <NotFoundPage />,
            },
        ],
    },
]);


function App() {
    return <RouterProvider router={router} />;
}

export default App;
