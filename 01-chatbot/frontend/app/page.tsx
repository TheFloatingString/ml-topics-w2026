"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface ToolCall {
  call_id: string;
  name: string;
  arguments: string;
  result?: string;
  status: "pending" | "completed";
}

interface Message {
  role: "user" | "assistant";
  content: string;
  toolCalls?: ToolCall[];
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [currentToolCalls, setCurrentToolCalls] = useState<ToolCall[]>([]);
  const [streamingText, setStreamingText] = useState("");
  const [isStreamingText, setIsStreamingText] = useState(false);
  const [darkMode, setDarkMode] = useState<boolean | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Initialize dark mode from localStorage or system preference
  useEffect(() => {
    const stored = localStorage.getItem("darkMode");
    if (stored !== null) {
      setDarkMode(stored === "true");
    } else {
      setDarkMode(window.matchMedia("(prefers-color-scheme: dark)").matches);
    }
  }, []);

  // Apply dark mode class to document and persist
  useEffect(() => {
    if (darkMode === null) return;
    document.documentElement.classList.toggle("dark", darkMode);
    localStorage.setItem("darkMode", String(darkMode));
  }, [darkMode]);

  // Auto-scroll to bottom when messages change or streaming
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentToolCalls, isLoading, streamingText]);

  const sendMessage = useCallback(async () => {
    if (!input.trim()) return;

    // Abort any existing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const currentRequestId = ++requestIdRef.current;
    const messageText = input;

    // Capture current messages for history before adding new one
    const history = messages.map((msg) => ({
      role: msg.role,
      content: msg.content,
    }));

    const userMessage: Message = { role: "user", content: messageText };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setCurrentToolCalls([]);
    setStreamingText("");
    setIsStreamingText(false);

    abortControllerRef.current = new AbortController();

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: messageText, history }),
        signal: abortControllerRef.current.signal,
      });

      // Check if this request is still the active one
      if (currentRequestId !== requestIdRef.current) return;

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) throw new Error("No reader");

      let buffer = "";
      let accumulatedText = "";
      const toolCallsMap = new Map<string, ToolCall>();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Check if this request is still the active one
        if (currentRequestId !== requestIdRef.current) {
          reader.cancel();
          return;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (let i = 0; i < lines.length; i++) {
          const line = lines[i];
          if (line.startsWith("event: ")) {
            const event = line.slice(7);
            const dataLine = lines[i + 1];
            if (dataLine?.startsWith("data: ")) {
              const data = JSON.parse(dataLine.slice(6));

              if (event === "tool_call_pending") {
                const tc: ToolCall = {
                  call_id: data.call_id,
                  name: data.name,
                  arguments: data.arguments,
                  status: "pending",
                };
                toolCallsMap.set(data.call_id, tc);
                setCurrentToolCalls(Array.from(toolCallsMap.values()));
              } else if (event === "tool_call_complete") {
                const existing = toolCallsMap.get(data.call_id);
                if (existing) {
                  existing.status = "completed";
                  existing.result = data.result;
                  setCurrentToolCalls(Array.from(toolCallsMap.values()));
                }
              } else if (event === "text_start") {
                setIsStreamingText(true);
              } else if (event === "text_delta") {
                accumulatedText += data.delta;
                setStreamingText(accumulatedText);
              } else if (event === "text_done") {
                setIsStreamingText(false);
              }
            }
          }
        }
      }

      // Final check before updating messages
      if (currentRequestId !== requestIdRef.current) return;

      const assistantMessage: Message = {
        role: "assistant",
        content: accumulatedText || "No response",
        toolCalls: Array.from(toolCallsMap.values()),
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setCurrentToolCalls([]);
      setStreamingText("");
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        console.error(err);
        if (currentRequestId === requestIdRef.current) {
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: "Error: Failed to get response" },
          ]);
        }
      }
    } finally {
      if (currentRequestId === requestIdRef.current) {
        setIsLoading(false);
        setIsStreamingText(false);
      }
    }
  }, [input, messages]);

  // Spinner component
  const Spinner = () => (
    <svg
      className="h-5 w-5 animate-spin"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );

  // Tool calls display component
  const ToolCallsDisplay = ({ toolCalls, isProcessing = false }: { toolCalls: ToolCall[]; isProcessing?: boolean }) => (
    <div className={isProcessing ? "" : "mt-3 border-t border-zinc-200 dark:border-zinc-600 pt-3"}>
      <p className="text-xs font-semibold text-zinc-500 dark:text-zinc-400 mb-2">
        {isProcessing ? "Processing Tool Calls:" : "Tool Calls:"}
      </p>
      <div className="space-y-2">
        {toolCalls.map((tc, j) => (
          <div
            key={tc.call_id || j}
            className="rounded bg-zinc-100 dark:bg-zinc-700 p-2 text-xs"
          >
            <div className="flex items-center justify-between">
              <span className="font-mono font-semibold text-blue-600 dark:text-blue-400">
                {tc.name}
              </span>
              <span
                className={`flex items-center gap-1 rounded px-2 py-0.5 text-[10px] font-medium ${
                  tc.status === "completed"
                    ? "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300"
                    : "bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300"
                }`}
              >
                {tc.status === "pending" && (
                  <span className="inline-block h-2 w-2 animate-spin rounded-full border border-yellow-600 border-t-transparent" />
                )}
                {tc.status === "completed" && "✓"} {tc.status}
              </span>
            </div>
            <p className="mt-1 text-zinc-600 dark:text-zinc-300">
              <span className="text-zinc-400">Args:</span> {tc.arguments}
            </p>
            {tc.result && (
              <p className="mt-1 text-zinc-600 dark:text-zinc-300">
                <span className="text-zinc-400">Result:</span> {tc.result}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="flex h-screen flex-col bg-zinc-50 dark:bg-zinc-900">
      <header className="flex-shrink-0 border-b border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-800">
        <div className="mx-auto flex max-w-3xl items-center justify-between">
          <h1 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
            Chatbot with Tool Calls
          </h1>
          {darkMode !== null && (
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="rounded-lg border border-zinc-300 bg-zinc-100 p-2 text-zinc-700 transition-colors hover:bg-zinc-200 dark:border-zinc-600 dark:bg-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-600"
              aria-label="Toggle dark mode"
            >
              {darkMode ? (
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
              ) : (
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                </svg>
              )}
            </button>
          )}
        </div>
      </header>

      <main className="flex-1 overflow-y-auto p-4">
        <div className="mx-auto max-w-3xl space-y-4">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[80%] rounded-lg p-3 ${
                  msg.role === "user"
                    ? "bg-blue-500 text-white"
                    : "bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700"
                }`}
              >
                {msg.role === "assistant" ? (
                  <>
                    {/* Tool calls first */}
                    {msg.toolCalls && msg.toolCalls.length > 0 && (
                      <div className="mb-3">
                        <ToolCallsDisplay toolCalls={msg.toolCalls} isProcessing />
                      </div>
                    )}
                    {/* Then the response text */}
                    <div className="prose prose-sm dark:prose-invert max-w-none text-zinc-900 dark:text-zinc-100">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                    </div>
                  </>
                ) : (
                  <p>{msg.content}</p>
                )}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="max-w-[80%] rounded-lg bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 p-3">
                {/* Tool calls first */}
                {currentToolCalls.length > 0 && (
                  <div className="mb-3">
                    <ToolCallsDisplay toolCalls={currentToolCalls} isProcessing />
                  </div>
                )}

                {/* Then streaming text or spinner below */}
                {streamingText ? (
                  <div className="prose prose-sm dark:prose-invert max-w-none text-zinc-900 dark:text-zinc-100">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{streamingText}</ReactMarkdown>
                    <span className="inline-block w-2 h-4 bg-zinc-400 dark:bg-zinc-500 animate-pulse ml-0.5" />
                  </div>
                ) : isStreamingText ? (
                  <div className="flex items-center gap-3 text-zinc-500 dark:text-zinc-400">
                    <Spinner />
                    <span>Generating response...</span>
                  </div>
                ) : currentToolCalls.length === 0 ? (
                  <div className="flex items-center gap-3 text-zinc-500 dark:text-zinc-400">
                    <Spinner />
                    <span>Thinking...</span>
                  </div>
                ) : null}
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      <footer className="flex-shrink-0 border-t border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-800">
        <div className="mx-auto flex max-w-3xl gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            placeholder="Type a message..."
            className="flex-1 rounded-lg border border-zinc-300 bg-white px-4 py-2 text-zinc-900 placeholder-zinc-400 focus:border-blue-500 focus:outline-none dark:border-zinc-600 dark:bg-zinc-700 dark:text-zinc-100"
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim()}
            className="rounded-lg bg-blue-500 px-4 py-2 font-medium text-white hover:bg-blue-600 disabled:opacity-50"
          >
            Send
          </button>
        </div>
      </footer>
    </div>
  );
}
