"use client";

import * as webllm from "@mlc-ai/web-llm";
import { useEffect, useRef, useState } from "react";
import { getSlidingWindow, MAX_CONTEXT_TOKENS } from "../lib/budget";
import {
  archiveToEpisodicMemory,
  extractAndStoreFacts,
  retrieveRelevantMemory,
} from "../lib/memory";
import {
  addDocument,
  extractEntitiesAndIntent,
  getEmbedding,
  searchDocumentsHybrid,
} from "../lib/rag";
import { DocumentType } from "../lib/rag";

export type Message = {
  role: "user" | "assistant" | "system";
  content: string;
};

const MODELS = [
  {
    label: "Phi-3 Mini (Great for Story)",
    value: "Phi-3-mini-4k-instruct-q4f16_1-MLC",
  },
  {
    label: "Llama 3.2 1B (Great for Logic)",
    value: "Llama-3.2-1B-Instruct-q4f16_1-MLC",
  },
  {
    label: "Gemma 2B (Balanced)",
    value: "gemma-2-2b-it-q4f16_1-MLC",
  },
  {
    label: "SmolLM2 1.7B (Small & Fast)",
    value: "SmolLM2-1.7B-Instruct-q4f16_1-MLC",
  },
];

export default function WebLLMChat() {
  const logicEngineRef = useRef<webllm.MLCEngineInterface | null>(null);
  const storyEngineRef = useRef<webllm.MLCEngineInterface | null>(null);

  const [logicModel, setLogicModel] = useState(MODELS[1].value);
  const [storyModel, setStoryModel] = useState(MODELS[0].value);

  const [loading, setLoading] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState<string>("Idle");

  async function initDualEngines(logicId: string, storyId: string) {
    setLoading(true);
    setStatus("Loading dual engines (Warning: Requires ~4GB+ VRAM)...");

    try {
      setStatus(`Loading Logic Engine (${logicId})...`);
      logicEngineRef.current = await webllm.CreateMLCEngine(logicId, {
        initProgressCallback: (p) => setStatus(`Logic Engine: ${p.text}`),
      });

      setStatus(`Loading Story Engine (${storyId})...`);
      storyEngineRef.current = await webllm.CreateMLCEngine(storyId, {
        initProgressCallback: (p) => setStatus(`Story Engine: ${p.text}`),
      });

      setStatus("Both engines ready! Roll for initiative.");
    } catch (err) {
      console.error(err);
      setStatus("Failed to load one or both models. Check VRAM capacity.");
    } finally {
      setLoading(false);
    }
  }

  // biome-ignore lint/correctness/useExhaustiveDependencies: not needed
  useEffect(() => {
    initDualEngines(logicModel, storyModel);
  }, []);

  async function handleAddKnowledge() {
    const text = window.prompt("Paste campaign lore, rules, or world-building text:");
    if (!text) return;

    const typeInput = window.prompt(
      "Is this a strict rule constraint, or general lore?\nType 'rule', 'lore', 'npc', or 'location'", 
      "lore"
    );
    
    const validTypes = ["rule", "lore", "npc", "location"];
      const docType: DocumentType = validTypes.includes(typeInput?.toLowerCase() || "") 
        ? (typeInput?.toLowerCase() as DocumentType) 
        : "lore";

      setStatus(`Adding ${docType} to vector DB...`);
      try {
        await addDocument(text, docType);
        setStatus(`${docType.charAt(0).toUpperCase() + docType.slice(1)} added! Engines ready.`);
      } catch (err) {
        console.error(err);
        setStatus("Failed to add knowledge");
      }
    }

  async function sendMessage() {
    if (!logicEngineRef.current || !storyEngineRef.current || !prompt.trim())
      return;

    const userText = prompt;
    setPrompt("");

    const displayMessage: Message = { role: "user", content: userText };
    const newMessages = [...messages, displayMessage];
    setMessages([...newMessages, { role: "assistant", content: "" }]);

    await extractAndStoreFacts(logicEngineRef.current, userText);

    setStatus("Analyzing intent and entities...");
    let extractedData = { action: "", targets: [] as string[] };
    
    try {
      extractedData = await extractEntitiesAndIntent(logicEngineRef.current, userText);
      console.log("Extracted Intent:", extractedData);
    } catch {
      console.warn("Extraction failed, proceeding with raw query");
    }

    const searchEmbedding = await getEmbedding(userText);
    
    const keywordQuery = [
      extractedData.action, 
      ...extractedData.targets, 
      userText
    ].join(" ");
    
    setStatus("Searching campaign lore and memories...");
    const [retrievedDocs, relevantMemories] = await Promise.all([
      searchDocumentsHybrid(keywordQuery, searchEmbedding, 3),
      retrieveRelevantMemory(searchEmbedding, 3),
    ]);

    const historyBudget = MAX_CONTEXT_TOKENS - 1500;
    const windowMessages = getSlidingWindow(newMessages, historyBudget);

    const overflowMessages = newMessages.slice(
      0,
      newMessages.length - windowMessages.length,
    );
    if (overflowMessages.length > 0) {
      await archiveToEpisodicMemory(overflowMessages);
    }

    const memoryContext = `
    [Character Sheet / Known Facts]:
    ${relevantMemories.semantic.length ? relevantMemories.semantic.map((f) => `- ${f.text}`).join("\n") : "None relevant."}

    [Past Events / Episodic Memory]:
    ${relevantMemories.episodic.length ? relevantMemories.episodic.map((e) => `...\n${e.text}\n...`).join("\n") : "None relevant."}

    [Campaign Lore / RAG]:
    ${retrievedDocs.length ? retrievedDocs.map((d, i) => `Lore Chunk ${i + 1}:\n${d.text}`).join("\n\n") : "None relevant."}
    `;

    setStatus("Logic Engine is computing the outcome...");
    let logicOutcome = "";
    try {
      const logicPrompt = `
      You are the Dungeon Master's Logic Engine. 
      Analyze the player's action based on the context. 
      Determine if it succeeds, fails, or triggers a consequence. Keep your output to a brief, factual bulleted list of state changes. DO NOT write a story.
      
      Context: ${memoryContext}
      Player Action: ${userText}
      `;

      const logicResponse =
        await logicEngineRef.current.chat.completions.create({
          messages: [
            {
              role: "system",
              content:
                "You strictly output game mechanics and logical consequences.",
            },
            { role: "user", content: logicPrompt },
          ],
          temperature: 0.1,
        });
      logicOutcome =
        logicResponse.choices[0].message.content ||
        "The action resolves neutrally.";

      console.log("LOGIC ENGINE COMPUTATION:", logicOutcome);
    } catch (err) {
      console.error(err);
      setStatus("Logic Engine failed.");
      return;
    }

    setStatus("Story Engine is narrating...");
    try {
      const storyPrompt = `
      Context: ${memoryContext}
      Player Action: ${userText}
      Logical Outcome determined by the DM: ${logicOutcome}

      Using the Logical Outcome as your strict guide, write a highly descriptive, immersive paragraph narrating what happens next in the D&D campaign. Describe the environment, the action, and the result. Address the player in the second person ("You...").
      `;

      const storySystemMessage: webllm.ChatCompletionMessageParam = {
        role: "system",
        content:
          "You are the creative, descriptive Narrator of a D&D game. You bring the world to life with rich sensory details.",
      };

      const payloadMessages = [
        storySystemMessage,
        ...windowMessages.slice(0, -1),
        { role: "user", content: storyPrompt },
      ] as webllm.ChatCompletionMessageParam[];

      const stream = await storyEngineRef.current.chat.completions.create({
        messages: payloadMessages,
        stream: true,
        temperature: 0.7,
      });

      let assistantText = "";
      for await (const chunk of stream) {
        assistantText += chunk.choices?.[0]?.delta?.content || "";
        setMessages((prev) => {
          const copy = [...prev];
          copy[copy.length - 1] = { role: "assistant", content: assistantText };
          return copy;
        });
      }
      setStatus("Both engines ready! Awaiting your next move.");
    } catch (err) {
      console.error(err);
      setStatus("Error generating story response");
    }
  }

  return (
    <div className="min-h-screen flex flex-col bg-gray-950 text-white font-serif">
      <div className="p-4 border-b border-gray-800 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-amber-500">Lonely D&D</h1>
          <p className="text-xs text-gray-400">Powered by Dual-Engine WebLLM</p>
        </div>

        <div className="flex gap-2 items-center">
          <button
            type="button"
            onClick={handleAddKnowledge}
            className="bg-amber-700 hover:bg-amber-600 px-3 py-1 mr-2 rounded text-sm transition-colors shadow-md"
          >
            + Add Campaign Lore
          </button>

          <div className="flex flex-col gap-1">
            <select
              className="bg-gray-900 border border-gray-700 px-2 py-1 rounded text-xs"
              value={logicModel}
              onChange={(e) => setLogicModel(e.target.value)}
            >
              {MODELS.map((m) => (
                <option key={`logic-${m.value}`} value={m.value}>
                  Logic: {m.label}
                </option>
              ))}
            </select>
            <select
              className="bg-gray-900 border border-gray-700 px-2 py-1 rounded text-xs"
              value={storyModel}
              onChange={(e) => setStoryModel(e.target.value)}
            >
              {MODELS.map((m) => (
                <option key={`story-${m.value}`} value={m.value}>
                  Story: {m.label}
                </option>
              ))}
            </select>
          </div>
          <button
            type="button"
            onClick={() => initDualEngines(logicModel, storyModel)}
            className="bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded text-xs"
          >
            Reload Engines
          </button>
        </div>
      </div>

      <div className="px-4 py-2 text-xs text-amber-400 border-b border-gray-800 flex justify-between bg-gray-900">
        <span className="animate-pulse">{status}</span>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((m, i) => (
          <div
            key={`${m.role}-${i}`}
            className={`max-w-3xl p-4 rounded-lg whitespace-pre-wrap leading-relaxed shadow-lg ${
              m.role === "user"
                ? "bg-slate-800 ml-auto border-l-4 border-blue-500"
                : "bg-gray-900 mr-auto border-l-4 border-amber-500 text-gray-200"
            }`}
          >
            {m.content}
          </div>
        ))}
      </div>

      <div className="p-4 border-t border-gray-800 bg-gray-900 flex gap-2">
        <input
          className="flex-1 bg-gray-950 border border-gray-700 px-4 py-3 rounded focus:outline-none focus:border-amber-500 text-lg"
          value={prompt}
          placeholder="What do you do? (e.g. 'I cast fireball at the goblins')"
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button
          type="button"
          onClick={sendMessage}
          disabled={loading}
          className="bg-amber-600 hover:bg-amber-500 px-6 py-2 rounded font-bold disabled:opacity-50 transition-colors shadow-md text-black"
        >
          Roll
        </button>
      </div>
    </div>
  );
}
