import { createContext, useContext } from "react";

export type SocketStatus = "connecting" | "connected" | "disconnected";

export type SocketContextType = {
  socketStatus: SocketStatus;
  sendMessage: (message: any) => void;
  socket?: WebSocket | null;
};

export const SocketContext = createContext<SocketContextType>({
  socketStatus: "disconnected",
  sendMessage: () => {},
});

export const useSocketContext = () => useContext(SocketContext);
