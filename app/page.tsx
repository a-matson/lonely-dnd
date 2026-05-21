"use client";

import { useEffect, useRef, useState } from "react";
import * as webllm from "@mlc-ai/web-llm";
import { addDocument, generateHyDE, getEmbedding, searchDocumentsHybrid } from "../lib/rag";
import { getSlidingWindow, summarizeOldMessages, MAX_CONTEXT_TOKENS } from "../lib/budget";

export type Message = { role: "user" | "assistant" | "system"; content: string };

const MODELS = [
  {
    label: "Phi-3 Mini (fast, recommended)",
    value: "Phi-3-mini-4k-instruct-q4f16_1-MLC",
  },
  {
    label: "Gemma 2B (balanced)",
    value: "gemma-2-2b-it-q4f16_1-MLC",
  },
];

export default function WebLLMChat() {
  const engineRef = useRef<any>(null);
  const [model, setModel] = useState(MODELS[0].value);
  const [loading, setLoading] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState<string>("Idle");
  const [conversationSummary, setConversationSummary] = useState("");

  async function initModel(selectedModel: string) {
    setLoading(true);
    setStatus("Loading model (first time may take a while)...");

    try {
      const engine = await webllm.CreateMLCEngine(selectedModel, {
        initProgressCallback: (p: any) => {
          setStatus(`${p.text || "Loading..."}`);
        },
      });

      engineRef.current = engine;
      setStatus("Model ready");
    } catch (err) {
      console.error(err);
      setStatus("Failed to load model");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    initModel(model);
  }, []);

  async function handleAddKnowledge() {
    const text = window.prompt("Paste knowledge text to embed:");
    if (!text) return;
    
    setStatus("Adding knowledge to vector DB...");
    try {
      await addDocument(text);
      setStatus("Knowledge added! Model ready");
    } catch (err) {
      console.error(err);
      setStatus("Failed to add knowledge");
    }
  }

  async function sendMessage() {
    if (!engineRef.current || !prompt.trim()) return;

    const userText = prompt;
    setPrompt("");

    // HyDE query transformation
    setStatus("Thinking about the query (HyDE)...");
    let searchEmbedding: number[];
    try {
      const hypotheticalAnswer = await generateHyDE(engineRef.current, userText);
      searchEmbedding = await getEmbedding(hypotheticalAnswer);
    } catch (e) {
      console.warn("HyDE failed, falling back to raw query embedding", e);
      searchEmbedding = await getEmbedding(userText);
    }

    // hybrid retrieval (RAG)
    setStatus("Searching knowledge base (Hybrid Search)...");
    const retrievedDocs = await searchDocumentsHybrid(userText, searchEmbedding, 4);
    
    // Token budgeting = MAX_CONTEXT_TOKENS - RAG - system - summary
    const historyBudget = MAX_CONTEXT_TOKENS - 1000 - 200 - 500;
    const windowMessages = getSlidingWindow(messages, historyBudget);

    const contextStr = retrievedDocs.map((d, idx) => `[Source Chunk ${idx + 1}]:\n${d.text}`).join("\n\n");
    
    const overflowMessages = messages.slice(0, messages.length - windowMessages.length);
    if (overflowMessages.length > 0) {
      summarizeOldMessages(engineRef.current, overflowMessages, conversationSummary)
        .then(newSummary => setConversationSummary(newSummary))
        .catch(console.error);
    }

    const displayMessage: Message = { role: "user", content: userText };
    setMessages([...messages, displayMessage, { role: "assistant", content: "" }]);

    const augmentedUserMessage: Message = {
      role: "user",
      content: `Use the provided context to answer the question. \n\nContext:\n${contextStr}\n\nQuestion: ${userText}`
    };

    const systemMessage: Message = {
      role: "system",
      content: `You are a helpful AI. \nPrevious conversation summary: ${conversationSummary}`
    };

    const payloadMessages = [
      systemMessage,
      ...windowMessages,
      augmentedUserMessage
    ];

    // generate response
    setStatus("Generating response...");
    try {
      const stream = await engineRef.current.chat.completions.create({
        messages: payloadMessages,
        stream: true,
      });

      let assistantText = "";
      for await (const chunk of stream) {
        assistantText += chunk.choices?.[0]?.delta?.content || "";
        setMessages((prev) => {
          const copy = [...prev];
          copy[copy.length - 1] = {
            role: "assistant",
            content: assistantText,
          };
          return copy;
        });
      }
      setStatus("Model ready");
    } catch (err) {
      console.error(err);
      setStatus("Error generating response");
    }
  }

  return (
    <div className="min-h-screen flex flex-col bg-gray-950 text-white">
      {/* Header */}
      <div className="p-4 border-b border-gray-800 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">WebLLM Starter (RAG)</h1>
          <p className="text-xs text-gray-400">Runs fully in browser (WebGPU)</p>
        </div>

        <div className="flex gap-2 items-center">
          <button 
             onClick={handleAddKnowledge}
             className="bg-green-700 hover:bg-green-600 px-3 py-1 mr-2 rounded text-sm transition-colors">
             + Add Knowledge
          </button>
          
          <select
            className="bg-gray-900 border border-gray-700 px-2 py-1 rounded"
            value={model}
            onChange={(e) => {
              setModel(e.target.value);
              initModel(e.target.value);
            }}
          >
            {MODELS.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Status */}
      <div className="px-4 py-2 text-xs text-gray-400 border-b border-gray-800 flex justify-between">
        <span>{status}</span>
        {conversationSummary && <span className="text-blue-400">Memory Active</span>}
      </div>

      {/* Chat */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`max-w-2xl p-3 rounded-lg whitespace-pre-wrap ${
              m.role === "user"
                ? "bg-blue-600 ml-auto"
                : "bg-gray-800 mr-auto"
            }`}
          >
            {m.content}
          </div>
        ))}
      </div>

      {/* Input */}
      <div className="p-4 border-t border-gray-800 flex gap-2">
        <input
          className="flex-1 bg-gray-900 border border-gray-700 px-3 py-2 rounded focus:outline-none focus:border-blue-500"
          value={prompt}
          placeholder="Ask something..."
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button
          onClick={sendMessage}
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-500 px-4 py-2 rounded disabled:opacity-50 transition-colors"
        >
          Send
        </button>
      </div>
    </div>
  );
}