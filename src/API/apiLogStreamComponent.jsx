import React, { useEffect, useState } from "react";
import { baseUrl } from "./apiConsts";

// LogStreamComponent будет принимать функцию для обновления логов из родительского компонента
const LogStreamComponent = ({ onNewLog }) => {
    const [logs, setLogs] = useState([]);

    useEffect(() => {
        const eventSource = new EventSource(`${baseUrl}/stream-logs`);

        eventSource.onopen = () => {
            console.log("Connection to log stream established.");
        };

        eventSource.onmessage = (event) => {
            console.log("Received log entry:", event.data);
            setLogs((prevLogs) => [...prevLogs, event.data]);

            // Отправляем новый лог родительскому компоненту через onNewLog
            if (onNewLog) {
                onNewLog(event.data);
            }
        };

        eventSource.onerror = (error) => {
            console.error("EventSource failed:", error);
            eventSource.close();
        };

        return () => {
            eventSource.close();
        };
    }, [onNewLog]);

    return null;  // Этот компонент не рендерит ничего, так как логируем в родительский компонент
};

export default LogStreamComponent;
