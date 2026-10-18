/**
 * Cohort AI Worker
 * Keeps the template's streaming /api/chat route and adds /api/cohort-ai,
 * a task-based JSON endpoint returning { text: string } for Cohort.
 */
import type { Env, ChatMessage } from "./types";

const MODEL_ID = "@cf/meta/llama-3.1-8b-instruct-fp8";
const MAX_INPUT_CHARS = 18000;
const SYSTEM_PROMPT =
  "You are Cohort Study Assistant, a helpful academic assistant for university students. " +
  "Explain concepts clearly, use accurate information, organize answers with headings or bullets when useful, " +
  "and never pretend to know information that was not provided. Be concise but useful.";

const ALLOWED_ORIGINS = new Set(["https://officialemzee24-hub.github.io"]);

function corsHeaders(origin: string | null): Record<string, string> {
  const headers: Record<string, string> = {
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Access-Control-Max-Age": "86400",
    Vary: "Origin",
  };
  if (origin && ALLOWED_ORIGINS.has(origin)) {
    headers["Access-Control-Allow-Origin"] = origin;
  }
  return headers;
}

function jsonResponse(data: unknown, status: number, cors: Record<string, string>): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      ...cors,
    },
  });
}

export default {
  async fetch(request: Request, env: Env, _ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    const cors = corsHeaders(request.headers.get("Origin"));

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: cors });
    }

    // Preserve the template frontend and static assets.
    if (url.pathname === "/" || !url.pathname.startsWith("/api/")) {
      return env.ASSETS.fetch(request);
    }

    if (url.pathname === "/api/chat") {
      if (request.method !== "POST") {
        return new Response("Method not allowed", {
          status: 405,
          headers: { ...cors, Allow: "POST, OPTIONS" },
        });
      }
      return handleChatRequest(request, env, cors);
    }

    if (url.pathname === "/api/cohort-ai" || url.pathname === "/api/assistant") {
      if (request.method !== "POST") {
        return jsonResponse({ error: "Method not allowed" }, 405, cors);
      }
      return handleCohortRequest(request, env, cors);
    }

    return jsonResponse({ error: "Not found" }, 404, cors);
  },
} satisfies ExportedHandler<Env>;

async function handleChatRequest(
  request: Request,
  env: Env,
  cors: Record<string, string>,
): Promise<Response> {
  try {
    const body = (await request.json()) as { messages?: ChatMessage[] };
    const messages = Array.isArray(body.messages) ? [...body.messages] : [];
    if (messages.length === 0) {
      return jsonResponse({ error: "Please provide a non-empty messages array." }, 400, cors);
    }
    if (!messages.some((message) => message.role === "system")) {
      messages.unshift({ role: "system", content: SYSTEM_PROMPT });
    }
    const inputs = {
      messages,
      max_tokens: 1024,
      stream: true,
    } satisfies AiTextGenerationInput & { stream: true };
    const stream = await env.AI.run<typeof MODEL_ID>(MODEL_ID, inputs);
    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        ...cors,
      },
    });
  } catch (error) {
    console.error("Error processing /api/chat:", error);
    return jsonResponse({ error: "Failed to process chat request." }, 500, cors);
  }
}

function taskInstructions(task: string): string {
  switch (task.toLowerCase()) {
    case "explain":
      return "Explain the topic step by step in student-friendly language. Define important terms and give an example where useful.";
    case "summary":
    case "summarize":
      return "Summarize the supplied material. Preserve its main points and terminology. Use headings and bullet points where helpful. Do not invent details absent from the material.";
    case "quiz":
      return "Create a practice quiz from the supplied topic or notes. Number the questions and include an answer key at the end. Follow the requested question count if provided.";
    case "catchup":
    case "catch-up":
      return "Help the student catch up on the supplied discussion or notes. Summarize what happened, identify key decisions or concepts, and list useful next steps. Do not invent missing details.";
    case "reminders":
    case "reminder":
      return "Extract dates, deadlines, tasks, and reminders explicitly stated in the supplied text. If none are present, say so. Do not invent dates.";
    case "moderate":
    case "moderation":
      return "Review the supplied message for clear harassment, threats, spam, or abusive content. Give a brief, neutral assessment and cite the textual reason. Do not infer intent without evidence.";
    case "ask":
    case "question":
    default:
      return "Answer the student's question clearly and accurately. If there is not enough information, say what is missing.";
  }
}

async function handleCohortRequest(
  request: Request,
  env: Env,
  cors: Record<string, string>,
): Promise<Response> {
  try {
    const body = (await request.json()) as {
      task?: string;
      type?: string;
      action?: string;
      text?: string;
      content?: string;
      prompt?: string;
      question?: string;
      context?: string;
      count?: number;
      messages?: ChatMessage[] | string;
    };

    const task = String(body.task || body.type || body.action || "ask").slice(0, 40);
    let userContent = [body.text, body.content, body.context, body.question, body.prompt]
      .filter((part): part is string => typeof part === "string" && part.trim().length > 0)
      .join("\n\n")
      .slice(0, MAX_INPUT_CHARS);

    // Cohort's current chatbox sends history as a formatted string. Other clients
    // may send a structured ChatMessage array, so support both formats.
    let historyContext = "";
    if (typeof body.messages === "string") {
      historyContext = body.messages.trim().slice(-10000);
    } else if (Array.isArray(body.messages) && body.messages.length > 0) {
      const validHistory = body.messages
        .filter((message) => message && ["user", "assistant"].includes(message.role) && typeof message.content === "string")
        .slice(-10);
      historyContext = validHistory
        .map((message) => `${message.role === "user" ? "Student" : "Cohort AI"}: ${message.content}`)
        .join("\n")
        .slice(-10000);
      const lastUserMessage = [...validHistory].reverse().find((message) => message.role === "user");
      if (!userContent && lastUserMessage) userContent = lastUserMessage.content.slice(0, MAX_INPUT_CHARS);
    }
    if (!userContent.trim()) {
      return jsonResponse(
        { error: "Missing text, question, prompt, content, context, or user message." },
        400,
        cors,
      );
    }

    const countInstruction =
      task.toLowerCase() === "quiz" && Number.isFinite(body.count) && Number(body.count) > 0
        ? ` Make exactly ${Math.min(20, Math.floor(Number(body.count)))} quiz questions.`
        : "";

    const messages: ChatMessage[] = [
      {
        role: "system",
        content: `${SYSTEM_PROMPT}\nTask: ${taskInstructions(task)}${countInstruction}`,
      },
      { role: "user", content: historyContext ? `Previous conversation (use this to understand follow-ups):\n${historyContext}\n\nCurrent request:\n${userContent}` : userContent },
    ];

    const result = await env.AI.run<typeof MODEL_ID>(MODEL_ID, {
      messages,
      max_tokens: 800,
      temperature: 0.4,
    });
    const output = typeof result === "string" ? result : (result as { response?: string }).response;
    if (!output || typeof output !== "string") {
      console.error("Unexpected Workers AI response shape.");
      return jsonResponse({ error: "The AI returned an empty response. Please try again." }, 502, cors);
    }
    return jsonResponse({ text: output }, 200, cors);
  } catch (error) {
    console.error("Error processing Cohort AI request:", error);
    return jsonResponse({ error: "Failed to process the Cohort AI request." }, 500, cors);
  }
}
