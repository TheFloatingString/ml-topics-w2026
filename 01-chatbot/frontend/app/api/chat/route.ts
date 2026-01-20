import Openai from "openai";
import { tools, executeToolCall, ToolResultStore } from "./tools";

const openai = new Openai({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: process.env.OPENROUTER_API_KEY,
  defaultHeaders: {
    "HTTP-Referer": process.env.SITE_URL || "http://localhost:3000",
    "X-Title": process.env.SITE_NAME || "AI Chatbot",
  },
});

const MODEL = "z-ai/glm-4.7";

const MAX_ARG_LENGTH = 500;

// Sanitize function call arguments to prevent context overflow
// When the model passes large data (like full book text), truncate it in the history
function sanitizeOutputForHistory(output: any[]): any[] {
  return output.map((item) => {
    if (item.type === "function_call") {
      try {
        const args = JSON.parse(item.arguments);
        let modified = false;
        for (const key of Object.keys(args)) {
          if (typeof args[key] === "string" && args[key].length > MAX_ARG_LENGTH) {
            args[key] = args[key].slice(0, MAX_ARG_LENGTH) + "...[truncated]";
            modified = true;
          }
        }
        if (modified) {
          return { ...item, arguments: JSON.stringify(args) };
        }
      } catch {
        // If parsing fails, return as-is
      }
    }
    return item;
  });
}

interface HistoryMessage {
  role: "user" | "assistant";
  content: string;
}

export async function POST(request: Request) {
  const body = await request.json();
  const userMessage = body.message || "Hello";
  const history: HistoryMessage[] = body.history || [];

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const send = (event: string, data: any) => {
        controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
      };

      try {
        // Build input from conversation history
        const input: any[] = [];
        for (const msg of history) {
          input.push({ role: msg.role, content: msg.content });
        }
        // Add the current user message
        input.push({ role: "user", content: userMessage });

        const store = new ToolResultStore();

        let response = await openai.responses.create({
          model: MODEL,
          input,
          tools,
        });

        let toolCalls = response.output.filter(
          (item): item is Openai.Responses.ResponseFunctionToolCall =>
            item.type === "function_call"
        );

        while (toolCalls.length > 0) {
          // Send pending tool calls immediately
          for (const toolCall of toolCalls) {
            send("tool_call_pending", {
              call_id: toolCall.call_id,
              name: toolCall.name,
              arguments: toolCall.arguments,
            });
          }

          input.push(...sanitizeOutputForHistory(response.output));

          // Execute each tool call and send completion
          for (const toolCall of toolCalls) {
            const args = JSON.parse(toolCall.arguments);
            const { summary } = await executeToolCall(toolCall.name, args, toolCall.call_id, store);

            send("tool_call_complete", {
              call_id: toolCall.call_id,
              name: toolCall.name,
              arguments: toolCall.arguments,
              result: summary,
            });

            // Send the summary to the model (not the full result)
            // The model can use $ref:call_id:path to reference full data in subsequent calls
            input.push({
              type: "function_call_output",
              call_id: toolCall.call_id,
              output: summary,
            });
          }

          response = await openai.responses.create({
            model: MODEL,
            input,
            tools,
          });

          toolCalls = response.output.filter(
            (item): item is Openai.Responses.ResponseFunctionToolCall =>
              item.type === "function_call"
          );
        }

        // Signal that we're starting the text response (for spinner)
        send("text_start", {});

        // Stream the final response
        input.push(...sanitizeOutputForHistory(response.output));

        const streamResponse = await openai.responses.create({
          model: MODEL,
          input,
          stream: true,
        });

        for await (const event of streamResponse) {
          if (event.type === "response.output_text.delta") {
            send("text_delta", { delta: event.delta });
          }
        }

        send("text_done", {});
        send("done", {});
      } catch (err) {
        console.error(err);
        send("error", { message: "Failed to get response" });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
