"use client";

import * as webllm from "@mlc-ai/web-llm";
import { useEffect, useRef, useState } from "react";
import MemoryManager from "../components/MemoryManager";
import { generateLocalAvatar } from "../lib/avatar";
import { getSlidingWindow, MAX_CONTEXT_TOKENS } from "../lib/budget";
import {
  archiveToEpisodicMemory,
  extractAndStoreFacts,
  getCurrentGameState,
  retrieveRelevantMemory,
  saveGameState,
} from "../lib/memory";
import {
  addDocument,
  type DocumentType,
  extractEntitiesAndIntent,
  getEmbedding,
  searchDocumentsHybrid,
} from "../lib/rag";

export type Message = {
  role: "user" | "assistant" | "system";
  content: string;
};

const MODELS = [
  {
    label: "Mistral 7B (Best for Story - High VRAM)",
    value: "Mistral-7B-Instruct-v0.3-q4f16_1-MLC",
  },
  {
    label: "Phi-3.5 Mini (Great for Story - Low VRAM)",
    value: "Phi-3.5-mini-instruct-q4f16_1-MLC",
  },
  {
    label: "Hermes Llama (Uncensored)",
    value: "Hermes-3-Llama-3.1-8B-q4f16_1-MLC",
  },
  {
    label: "Qwen 2.5 3B (Best for Logic)",
    value: "Qwen2.5-3B-Instruct-q4f16_1-MLC",
  },
  {
    label: "Qwen 2.5 1.5B (Good for Logic - Low VRAM)",
    value: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
  },
  {
    label: "Llama 3.2 3B (Balanced Logic)",
    value: "Llama-3.2-3B-Instruct-q4f16_1-MLC",
  },
];

export default function WebLLMChat() {
  const engineRef = useRef<webllm.MLCEngineInterface | null>(null);

  const [selectedModel, setSelectedModel] = useState(MODELS[2].value);

  const [loading, setLoading] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState<string>("Idle");

  const [suggestedActions, setSuggestedActions] = useState<string[]>([]);

  const [isLoaded, setIsLoaded] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  useEffect(() => {
    const savedChat = localStorage.getItem("lonely-dnd-chat");
    if (savedChat) {
      try {
        setMessages(JSON.parse(savedChat));
      } catch (e) {
        console.error("Failed to parse saved chat:", e);
      }
    }
    setIsLoaded(true);
  }, []);

  useEffect(() => {
    if (isLoaded) {
      localStorage.setItem("lonely-dnd-chat", JSON.stringify(messages));
    }
  }, [messages, isLoaded]);

  function clearChat() {
    if (
      window.confirm(
        "Are you sure you want to clear the chat and start a new campaign?",
      )
    ) {
      setMessages([]);
      localStorage.removeItem("lonely-dnd-chat");
      setStatus("Chat cleared. Ready for a new adventure.");
    }
  }

  async function initEngine(modelId: string) {
    setLoading(true);
    setStatus("Loading AI engine...");
    try {
      engineRef.current = await webllm.CreateMLCEngine(modelId, {
        initProgressCallback: (p) => setStatus(`Engine: ${p.text}`),
      });
      setStatus("Engine ready! Roll for initiative.");
    } catch (err) {
      console.error(err);
      setStatus("Failed to load engine.");
    } finally {
      setLoading(false);
    }
  }

  // biome-ignore lint/correctness/useExhaustiveDependencies: not needed
  useEffect(() => {
    initEngine(selectedModel);
  }, []);

  async function handleAddKnowledge() {
    const text = window.prompt(
      "Paste campaign lore, rules, or world-building text:",
    );
    if (!text) return;

    const typeInput = window.prompt(
      "Is this a strict rule constraint, or general lore?\nType 'rule', 'lore', 'npc', or 'location'",
      "lore",
    );

    const validTypes = ["rule", "lore", "npc", "location"];
    const docType: DocumentType = validTypes.includes(
      typeInput?.toLowerCase() || "",
    )
      ? (typeInput?.toLowerCase() as DocumentType)
      : "lore";

    setStatus(`Adding ${docType} to vector DB...`);
    try {
      await addDocument(text, docType);
      setStatus(
        `${docType.charAt(0).toUpperCase() + docType.slice(1)} added! Engines ready.`,
      );
    } catch (err) {
      console.error(err);
      setStatus("Failed to add knowledge");
    }
  }

  async function sendMessage() {
    if (!engineRef.current || !prompt.trim()) return;

    const userText = prompt;
    setPrompt("");
    setSuggestedActions([]);

    const displayMessage: Message = { role: "user", content: userText };
    const newMessages = [...messages, displayMessage];
    setMessages([...newMessages, { role: "assistant", content: "" }]);

    setStatus("Analyzing intent...");
    let extractedData = { action: "", targets: [] as string[] };
    try {
      extractedData = await extractEntitiesAndIntent(
        engineRef.current,
        userText,
      );
      console.log("Extracted Intent:", extractedData);
    } catch {
      console.warn("Extraction failed, proceeding with raw query");
    }

    const searchEmbedding = await getEmbedding(userText);
    const keywordQuery = [
      extractedData.action,
      ...extractedData.targets,
      userText,
    ].join(" ");

    const currentState = await getCurrentGameState();

    setStatus("Searching campaign lore...");
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
    [CURRENT GAME STATE]:
    ${JSON.stringify(currentState, null, 2)}

    [Character Sheet / Known Facts]:
    ${relevantMemories.semantic.length ? relevantMemories.semantic.map((f) => `- ${f.text}`).join("\n") : "None relevant."}

    [Past Events / Episodic Memory]:
    ${relevantMemories.episodic.length ? relevantMemories.episodic.map((e) => `...\n${e.text}\n...`).join("\n") : "None relevant."}

    [Campaign Lore / Rules]:
    ${retrievedDocs.length ? retrievedDocs.map((d, i) => `Chunk ${i + 1}:\n${d.text}`).join("\n\n") : "None relevant."}
    `;

    setStatus("Logic Engine is computing outcome and updating game state...");
    let logicOutcome = "";

    const logicSchema = {
      type: "object",
      properties: {
        identified_constraints: {
          type: "array",
          items: { type: "string" },
          description:
            "Step 1: List any specific rules, physical boundaries, or NPC states from the context that restrict or affect this action.",
        },
        logic_outcome: {
          type: "string",
          description:
            "Step 2: Based on the constraints, determine if the action succeeds/fails and briefly list what happens.",
        },
        new_game_state: {
          type: "object",
          properties: {
            location: { type: "string" },
            time_and_weather: { type: "string" },
            player: {
              type: "object",
              properties: {
                physical_description: {
                  type: "string",
                  description:
                    "A concise visual description of the player character.",
                },
              },
            },
            npcs: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  name: { type: "string" },
                  state: {
                    type: "string",
                    description: "e.g., Healthy, Injured, Dead, Hidden",
                  },
                  mood: {
                    type: "string",
                    description: "e.g., Hostile, Terrified, Friendly",
                  },
                  current_action: {
                    type: "string",
                    description: "What the NPC is doing right now",
                  },
                  physical_description: {
                    type: "string",
                    description:
                      "A vivid physical description of the NPC for portrait generation.",
                  },
                },
                required: ["name", "state", "mood", "current_action"],
              },
            },
          },
          required: ["location", "npcs"],
        },
        suggested_actions: {
          type: "array",
          items: { type: "string" },
          description:
            "Provide exactly of at least 3 distinc possible, one-sentence actions the player could choose to take next based on this new outcome.",
        },
      },
      required: [
        "identified_constraints",
        "logic_outcome",
        "new_game_state",
        "suggested_actions",
      ],
    };

    try {
      const logicPrompt = `
      You are the Dungeon Master's Logic Engine. 
      Analyze the player's action based on the [CURRENT GAME STATE] and [Campaign Lore / Rules].
      
      Follow these steps strictly:
      1. Identify Constraints: Are there rules, locked doors, or NPC conditions that apply here?
      2. Resolve Action: Is it possible? Does it succeed or fail?
      3. Update State: Modify the game state (NPC moods, states, actions, location) to reflect the result.
      
      Context: ${memoryContext}
      Player Action: ${userText}
      `;

      const logicResponse = await engineRef.current.chat.completions.create({
        messages: [
          {
            role: "system",
            content:
              "You strictly output JSON representing game mechanics and state changes. Think step-by-step. Keep all text fields extremely brief and concise (1-2 sentences max). DO NOT write a story. DO NOT repeat the context back to me.",
          },
          { role: "user", content: logicPrompt },
        ],
        temperature: 0.1,
        max_tokens: 1000,
        response_format: {
          type: "json_object",
          schema: JSON.stringify(logicSchema),
        },
      });

      const rawContent = logicResponse.choices[0].message.content || "{}";
      let parsedLogic: any = {};

      try {
        parsedLogic = JSON.parse(rawContent);
      } catch {
        console.warn("DM Engine output invalid JSON. Raw output:", rawContent);
        parsedLogic = {
          identified_constraints: ["Failed to parse constraints."],
          logic_outcome:
            "The action resolves, but the exact consequences are unclear due to a DM error.",
        };
      }

      // save the suggested actions to state
      if (
        parsedLogic.suggested_actions &&
        Array.isArray(parsedLogic.suggested_actions)
      ) {
        setSuggestedActions(parsedLogic.suggested_actions);
      }

      console.log(
        "Constraints Identified by DM Engine:",
        parsedLogic.identified_constraints,
      );
      logicOutcome =
        parsedLogic.logic_outcome || "The action resolves neutrally.";

      if (parsedLogic.new_game_state) {
        let needsAvatarGeneration = false;

        if (
          parsedLogic.new_game_state.player?.physical_description &&
          !currentState.player?.avatar_url
        ) {
          needsAvatarGeneration = true;
        }
        for (const npc of parsedLogic.new_game_state.npcs) {
          const existingNPC = currentState.npcs.find(
            (n) => n.name === npc.name,
          );
          if (!existingNPC?.avatar_url && npc.physical_description) {
            needsAvatarGeneration = true;
            break;
          }
        }

        if (needsAvatarGeneration) {
          setStatus("Freeing VRAM for Image Generation...");

          await engineRef.current?.unload();

          if (
            parsedLogic.new_game_state.player?.physical_description &&
            !currentState.player?.avatar_url
          ) {
            setStatus(
              "Generating local player avatar (This may take a moment)...",
            );
            parsedLogic.new_game_state.player.avatar_data_url =
              await generateLocalAvatar(
                parsedLogic.new_game_state.player.physical_description,
              );
          } else if (currentState.player?.avatar_url) {
            parsedLogic.new_game_state.player.avatar_data_url =
              currentState.player.avatar_url;
            parsedLogic.new_game_state.player.physical_description =
              currentState.player.physical_description;
          }

          for (const npc of parsedLogic.new_game_state.npcs) {
            const existingNPC = currentState.npcs.find(
              (n) => n.name === npc.name,
            );

            if (existingNPC?.avatar_url) {
              npc.avatar_data_url = existingNPC.avatar_url;
              npc.physical_description = existingNPC.physical_description;
            } else if (npc.physical_description) {
              setStatus(`Generating local avatar for ${npc.name}...`);
              npc.avatar_data_url = await generateLocalAvatar(
                npc.physical_description,
              );
            }
          }

          setStatus("Reloading LLM engines for the story...");
          await initEngine(selectedModel);
        } else {
          if (
            currentState.player?.avatar_url &&
            parsedLogic.new_game_state.player
          ) {
            parsedLogic.new_game_state.player.avatar_data_url =
              currentState.player.avatar_url;
            parsedLogic.new_game_state.player.physical_description =
              currentState.player.physical_description;
          }
          for (const npc of parsedLogic.new_game_state.npcs) {
            const existingNPC = currentState.npcs.find(
              (n) => n.name === npc.name,
            );
            if (existingNPC?.avatar_url) {
              npc.avatar_data_url = existingNPC.avatar_url;
              npc.physical_description = existingNPC.physical_description;
            }
          }
        }

        // 4. Save the finalized state
        await saveGameState(parsedLogic.new_game_state);
        console.log("Game State Updated:", parsedLogic.new_game_state);
      }
    } catch (err) {
      console.error("Logic Engine failed:", err);
      logicOutcome = "The action resolves, but state update failed.";
    }

    setStatus("Story Engine is narrating...");
    try {
      const freshestState = await getCurrentGameState();
      const storyPrompt = `
      Context & Lore: ${memoryContext}
      Current World State: ${JSON.stringify(freshestState)}
      Player Action: ${userText}
      Logical Outcome determined by the DM: ${logicOutcome}

      Using the Logical Outcome and Current World State as your strict guide, write a highly descriptive, immersive paragraph narrating what happens in the campaign. Describe the NPC's moods/actions, and the result. Address the player in the second person ("You...").
      `;

      const windowMessages = getSlidingWindow(
        messages.slice(0, -1),
        MAX_CONTEXT_TOKENS - 1500,
      );

      const stream = await engineRef.current.chat.completions.create({
        messages: [
          {
            role: "system",
            content:
              "You are the creative, descriptive Narrator of a D&D game.",
          },
          ...windowMessages,
          { role: "user", content: storyPrompt },
        ],
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
      setStatus("Extracting narrative facts...");
      await extractAndStoreFacts(engineRef.current, userText, assistantText);
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
            onClick={() => setIsMenuOpen(true)}
            className="bg-gray-800 hover:bg-gray-700 border border-gray-700 px-3 py-1 mr-2 rounded text-sm transition-colors shadow-md"
          >
            🧠 Manage Memory
          </button>
          <button
            type="button"
            onClick={handleAddKnowledge}
            className="bg-amber-700 hover:bg-amber-600 px-3 py-1 mr-2 rounded text-sm transition-colors shadow-md"
          >
            + Add Lore
          </button>

          <div className="flex gap-1 hidden md:flex items-center">
            <select
              className="bg-gray-900 border border-gray-700 px-2 py-1 rounded text-xs"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              {MODELS.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={() => initEngine(selectedModel)}
              className="bg-gray-700 hover:bg-gray-600 px-3 py-1.5 rounded text-xs h-full"
            >
              Load Engine
            </button>
          </div>
        </div>
      </div>

      <div className="px-4 py-2 text-xs text-amber-400 border-b border-gray-800 flex justify-between bg-gray-900">
        <span className="animate-pulse">{status}</span>
      </div>

      <MemoryManager isOpen={isMenuOpen} onClose={() => setIsMenuOpen(false)} />

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

      {suggestedActions.length > 0 &&
        status === "Both engines ready! Awaiting your next move." && (
          <div className="px-4 py-2 flex flex-wrap gap-2 bg-gray-950">
            {suggestedActions.map((action, idx) => (
              <button
                // biome-ignore lint/suspicious/noArrayIndexKey: not needed
                key={idx}
                type="button"
                onClick={() => setPrompt(action)}
                className="bg-slate-800 hover:bg-slate-700 text-amber-400 border border-slate-700 px-4 py-2 rounded-full text-sm transition-colors text-left shadow-sm"
              >
                {action}
              </button>
            ))}
          </div>
        )}

      <div className="p-4 border-t border-gray-800 bg-gray-900 flex gap-2">
        <button
          type="button"
          onClick={clearChat}
          className="bg-red-900/50 hover:bg-red-800 text-red-200 px-4 py-2 rounded font-semibold transition-colors shadow-md"
          title="Clear Chat"
        >
          Reset
        </button>

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
