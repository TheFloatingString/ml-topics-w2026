import Openai from "openai";
import {
  getGutenbergText,
  extractSentencesWithKeyword,
  analyzeSentiment,
  searchBooks,
} from "./tool-handlers";

// Store for tool results that can be referenced by subsequent tool calls
export class ToolResultStore {
  private results: Map<string, any> = new Map();

  set(callId: string, result: any) {
    this.results.set(callId, result);
  }

  get(callId: string): any {
    return this.results.get(callId);
  }

  // Resolve a value that might be a reference ($ref:call_id:path) or a literal
  resolve(value: any): any {
    if (typeof value === "string" && value.startsWith("$ref:")) {
      const [, callId, ...pathParts] = value.split(":");
      const stored = this.results.get(callId);
      if (!stored) return value;

      // Navigate the path if provided (e.g., $ref:abc123:text)
      let result = stored;
      for (const part of pathParts) {
        if (result && typeof result === "object" && part in result) {
          result = result[part];
        }
      }
      return result;
    }
    return value;
  }
}

export const tools: Openai.Responses.Tool[] = [
  {
    type: "function",
    name: "get_text_from_project_Gutenberg_id",
    description: "Get the text of a book from Project Gutenberg given its ID.",
    parameters: {
      type: "object",
      properties: {
        id: {
          type: "number",
          description: "Project Gutenberg ID",
        },
      },
      required: ["id"],
      additionalProperties: false,
    },
    strict: true,
  },
  {
    type: "function",
    name: "extract_sentences_where_keyword_occurs",
    description: "Extract sentences where a keyword occurs in a text. For 'text', you can pass a reference to a previous tool result using $ref:call_id:path (e.g., $ref:abc123:text).",
    parameters: {
      type: "object",
      properties: {
        text: {
          type: "string",
          description: "Text to search in, or a reference like $ref:call_id:text",
        },
        keyword: {
          type: "string",
          description: "Keyword to search for",
        },
      },
      required: ["text", "keyword"],
      additionalProperties: false,
    },
    strict: true,
  },
  {
    type: "function",
    name: "extract_sentences_where_keyword_occurs",
    description: "Get the text of a book from Project Gutenberg given its URL.",
    parameters: {
      type: "object",
      properties: {
        url: {
          type: "string",
          description: "Project Gutenberg URL",
        },
      },
      required: ["url"],
      additionalProperties: false,
    },
    strict: true,
  },
  {
    type: "function",
    name: "run_sentiment_analysis_on_list_of_sentences",
    description: "Run sentiment analysis on a list of sentences. For 'sentences', you can pass a reference to a previous tool result using $ref:call_id:sentences.",
    parameters: {
      type: "object",
      properties: {
        sentences: {
          type: "string",
          description: "List of sentences to analyze, or a reference like $ref:call_id:sentences",
        },
      },
      required: ["sentences"],
      additionalProperties: false,
    },
    strict: true,
  },
  {
    type: "function",
    name: "list_book_ids_from_search_query",
    description: "Get a list of book IDs from a search query.",
    parameters: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Search query",
        },
      },
      required: ["query"],
      additionalProperties: false,
    },
    strict: true,
  },
];

export async function executeToolCall(
  name: string,
  args: Record<string, any>,
  callId: string,
  store: ToolResultStore
): Promise<{ result: any; summary: string }> {
  let result: any;
  let summary: string;

  if (name === "get_text_from_project_Gutenberg_id") {
    result = await getGutenbergText(args as { id: number });
    summary = JSON.stringify({
      success: true,
      charCount: result.text.length,
      preview: result.text.slice(0, 200) + "...",
      ref: `$ref:${callId}:text`,
    });
  } else if (name === "extract_sentences_where_keyword_occurs") {
    const text = store.resolve(args.text);
    result = extractSentencesWithKeyword({ text, keyword: args.keyword });
    summary = JSON.stringify({
      count: result.sentences.length,
      preview: result.sentences.slice(0, 3),
      ref: `$ref:${callId}:sentences`,
    });
  } else if (name === "run_sentiment_analysis_on_list_of_sentences") {
    const sentences = store.resolve(args.sentences);
    const sentencesArray = Array.isArray(sentences) ? sentences : [];
    result = analyzeSentiment({ sentences: sentencesArray });
    summary = JSON.stringify({
      analyzed: result.sentiments.length,
      breakdown: result.breakdown,
    });
  } else if (name === "list_book_ids_from_search_query") {
    result = await searchBooks(args as { query: string });
    summary = JSON.stringify(result);
  } else {
    result = { error: "Unknown function" };
    summary = JSON.stringify(result);
  }

  store.set(callId, result);
  return { result, summary };
}
