import React, { useEffect, useState } from "react";
import { baseUrl } from "./apiConsts";

// LogStreamComponent ����� ��������� ������� ��� ���������� ����� �� ������������� ����������
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

            // ���������� ����� ��� ������������� ���������� ����� onNewLog
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

    return null;  // ���� ��������� �� �������� ������, ��� ��� �������� � ������������ ���������
};

export default LogStreamComponent;
